"""Data/layout/adapter tests only. No RF-DETR model, optimizer, download or GPU."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from src.layout import parse_small_equation
from src.data import SYMBOLS, eligible_symbols, letterbox_image, transform_box, validate_export
from src.model import (PIPELINE, SymbolPredictor, load_symbol_predictor,
                                map_detections, predict_symbol_image)
from src.train import main, read_config
from src.evaluate import match_symbols


PROJECT = Path(__file__).resolve().parents[1]


def symbol(text, x, y=30, width=20, height=40, confidence=.95):
    return {'text': text, 'bbox': [x, y, x + width, y + height], 'confidence': confidence}


class LayoutTests(unittest.TestCase):
    def parse(self, symbols, expected):
        result = parse_small_equation(symbols)
        self.assertTrue(result['layout_valid'], result['latex_metadata']['warnings'])
        self.assertEqual(result['latex'], expected)
        self.assertFalse(result['latex_metadata']['verified'])
        from src.validation import validate_hierarchy
        validate_hierarchy(result['nodes'])
        self.assertEqual(sum(n['terminal'] for n in result['nodes']), len(symbols))
        return result

    def test_linear_and_negative_rhs(self):
        self.parse([symbol(t, i * 28) for i, t in enumerate('2x-3=-7')], '2 x - 3 = -7')

    def test_polynomial_and_single_power(self):
        items = [symbol('x', 10), symbol('2', 31, 8, 14, 24), symbol('+', 53, 43, 24, 18),
                 symbol('2', 85), symbol('x', 109), symbol('+', 138, 43, 24, 18),
                 symbol('1', 175), symbol('=', 207, 45, 26, 14), symbol('0', 244)]
        result = self.parse(items, 'x^{2} + 2 x + 1 = 0')
        exponent = next(n for n in result['nodes'] if n['role'] == 'superscript')
        self.assertEqual(exponent['text'], '2')
        self.parse(items[:2], 'x^{2}')

    def test_multidigit_numbers_and_explicit_multiplication(self):
        self.parse([symbol('1', 0), symbol('2', 23), symbol('×', 60, 43, 24, 18),
                    symbol('3', 100), symbol('=', 140, 45, 26, 14), symbol('3', 180), symbol('6', 203)],
                   r'12 \times 3 = 36')

    def test_unsupported_or_ambiguous_layouts_are_rejected(self):
        cases = [[], [symbol('?', 0)], [symbol('x', 0), symbol('=', 30)],
                 [symbol('x', 0), symbol('2', 28)],
                 [symbol('x', 0), symbol('=', 30), symbol('1', 60), symbol('=', 90), symbol('2', 120)],
                 [symbol('2', 0, 0), symbol('3', 0, 100)],
                 [symbol('x', 0), symbol('z', 0, confidence=.92)],
                 [symbol('1', i * 24) for i in range(25)]]
        for items in cases:
            with self.subTest(items=items):
                result = parse_small_equation(items)
                self.assertFalse(result['layout_valid'])
                self.assertEqual(result['latex'], '')

    def test_same_label_duplicates_are_removed_without_inventing_text(self):
        result = parse_small_equation([symbol('z', 0), symbol('z', 1, confidence=.7)])
        self.assertTrue(result['layout_valid'])
        self.assertEqual(result['latex'], 'z')
        self.assertEqual(len(result['symbols']), 1)


class SymbolDataTests(unittest.TestCase):
    def test_letterbox_and_inverse_boxes_on_transparent_input(self):
        image = Image.new('RGBA', (600, 220))
        ImageDraw.Draw(image).rectangle((100, 70, 399, 129), fill='black')
        prepared, geometry = letterbox_image(image)
        self.assertEqual(prepared.size, (512, 512))
        original = [110., 75., 140., 125.]
        restored = transform_box(transform_box(original, geometry), geometry, inverse=True)
        np.testing.assert_allclose(restored, original)
        self.assertEqual(prepared.getpixel((0, 0)), (255, 255, 255))

    def test_whole_image_rejected_for_combined_tokens_or_complex_structure(self):
        base = {'id': 0, 'parent_id': None, 'kind': 'leaf', 'role': 'group', 'text': 'x', 'bbox': [0, 0, 20, 40]}
        self.assertEqual(eligible_symbols([base])[0]['text'], 'x')
        for change in ({'text': 'sin('}, {'text': '12'}, {'kind': 'fraction'}):
            with self.assertRaises(ValueError):
                eligible_symbols([{**base, **change}])

    def test_small_generated_coco_is_disjoint_complete_and_refuses_overwrite(self):
        from generate_data import generate
        with tempfile.TemporaryDirectory(prefix='eqocr_symbols_test_') as tmp:
            folder = Path(tmp) / 'symbols'
            report = generate(folder, count=100, seed=9)
            checked = validate_export(folder)
            self.assertEqual(sum(checked['splits'].values()), 100)
            self.assertFalse(report['missing_training_classes'])
            self.assertFalse(report['training_started'])
            for split in ('train', 'valid', 'test'):
                coco = json.loads((folder / split / '_annotations.coco.json').read_text())
                self.assertEqual([c['id'] for c in coco['categories']], list(range(len(SYMBOLS))))
                self.assertTrue(all(a['bbox'][2] > 0 and a['bbox'][3] > 0 for a in coco['annotations']))
            with self.assertRaises(FileExistsError):
                generate(folder, count=100)
            # Existing outputs are rejected before importing/constructing RF-DETR.
            from src.train import start_training, training_plan
            output = Path(tmp) / 'existing_checkpoints'
            output.mkdir()
            (output / 'keep.pth').write_bytes(b'protected fixture')
            with patch('src.train.require_backend', side_effect=AssertionError('No backend')):
                with self.assertRaises(FileExistsError):
                    start_training(read_config(), training_plan(read_config(), folder, output), 'cpu')
            self.assertEqual((output / 'keep.pth').read_bytes(), b'protected fixture')

    def test_kaggle_packaging_contains_only_symbol_data_and_verified_source(self):
        import ast
        import base64
        import hashlib
        import shutil
        import zipfile
        from generate_data import generate
        from prepare_kaggle import SOURCE_FILES, prepare
        with tempfile.TemporaryDirectory(prefix='eqocr_rf_package_test_') as tmp:
            project = Path(tmp) / 'project'
            project.mkdir()
            for name in SOURCE_FILES:
                destination = project / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(PROJECT / name, destination)
            data = Path(tmp) / 'data'
            generate(data, count=100, seed=11)
            (data / 'old_weights.pt').write_bytes(b'not part of the symbol dataset')
            report = prepare(data, project=project)
            self.assertFalse(report['uploaded'])
            self.assertFalse(report['training_started'])
            source = (project / 'kaggle_rfdetr_small/run.py').read_text(encoding='utf-8')
            compile(source, 'packaged_run.py', 'exec')
            values = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(source).body
                      if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                      and n.targets[0].id in {'SOURCE_HASH', 'SOURCE_B64'}}
            bundle = base64.b64decode(values['SOURCE_B64'])
            self.assertEqual(hashlib.sha256(bundle).hexdigest(), values['SOURCE_HASH'])
            with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
                self.assertEqual(set(archive.namelist()), set(SOURCE_FILES))
                for name in SOURCE_FILES:
                    self.assertEqual(archive.read(name), (PROJECT / name).read_bytes())
            with zipfile.ZipFile(project / 'kaggle_rfdetr_small_upload/dataset.zip') as archive:
                self.assertIsNone(archive.testzip())
                self.assertIn('dataset_symbols_small/train/_annotations.coco.json', archive.namelist())
                self.assertFalse(any(name.endswith(('.pt', '.pth', '.ckpt')) for name in archive.namelist()))

    def test_generated_equations_are_parseable_from_their_symbol_geometry(self):
        # This tests renderer/parser compatibility, NOT neural recognition.
        import random
        from generate_data import installed_fonts, render_symbols, template
        fonts, italic = installed_fonts()
        rng = random.Random(10)
        for index in range(3 * len(SYMBOLS), 3 * len(SYMBOLS) + 80):
            tokens, _, expected = template(index, rng)
            for font in fonts[:2]:
                _, symbols = render_symbols(tokens, font, size=54, math_italic=italic if index % 2 else None)
                result = parse_small_equation(symbols)
                self.assertTrue(result['layout_valid'], (index, expected, result['latex_metadata']['warnings']))
                self.assertEqual(result['latex'].replace(' ', ''), expected.replace(' ', ''))


class AdapterAndStartupTests(unittest.TestCase):
    def test_default_training_command_is_a_preview_without_backend_or_dataset(self):
        with tempfile.TemporaryDirectory(prefix='eqocr_plan_test_') as tmp:
            output = Path(tmp) / 'checkpoints'
            with patch('src.train.start_training', side_effect=AssertionError('No training')), contextlib.redirect_stdout(io.StringIO()):
                plan = main(['--data', str(Path(tmp) / 'missing'), '--output', str(output)])
            self.assertFalse(plan['training_started'])
            self.assertFalse(output.exists())
        config = read_config()
        self.assertEqual(config['aug_config'], {})
        self.assertFalse(config['scale_jitter'])
        self.assertFalse(config['multi_scale'])

    def test_missing_checkpoint_never_imports_or_constructs_rf_detr(self):
        with tempfile.TemporaryDirectory(prefix='eqocr_missing_test_') as tmp:
            with patch('src.model.require_backend', side_effect=AssertionError('No backend access')):
                with self.assertRaises(FileNotFoundError):
                    load_symbol_predictor(Path(tmp) / 'missing.pth')

    def test_mapping_uses_custom_class_ids_and_rejects_padding(self):
        image = Image.new('RGB', (300, 100), 'white')
        _, geometry = letterbox_image(image)
        mapped_box = transform_box([100, 20, 120, 60], geometry)
        detections = SimpleNamespace(xyxy=np.array([mapped_box, [0, 0, 10, 10]]),
                                     class_id=np.array([SYMBOLS.index('x'), SYMBOLS.index('0')]),
                                     confidence=np.array([.9, .9]), data={})
        symbols, rejected = map_detections(detections, geometry, image.size)
        self.assertEqual(symbols[0]['text'], 'x')
        np.testing.assert_allclose(symbols[0]['bbox'], [100, 20, 120, 60])
        self.assertEqual(rejected, 1)

    def test_fake_backend_result_is_not_replaced_by_a_reference_equation(self):
        image = Image.new('RGBA', (100, 80))
        ImageDraw.Draw(image).rectangle((20, 20, 39, 59), fill='black')
        _, geometry = letterbox_image(image)
        detections = SimpleNamespace(xyxy=np.array([transform_box([20, 20, 40, 60], geometry)]),
                                     class_id=np.array([SYMBOLS.index('z')]), confidence=np.array([.9]), data={})
        model = SimpleNamespace(predict=lambda image, threshold: detections)
        predictor = SymbolPredictor(model, {'pipeline': PIPELINE, 'resolution': 512}, Path('fake.pth'))
        result, colored = predict_symbol_image(image, predictor)
        self.assertEqual(result['latex'], 'z')
        self.assertFalse(result['latex_metadata']['verified'])
        self.assertEqual(colored.size, image.size)
        self.assertEqual(colored.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(result['metadata']['layout_source'], 'spatial grammar; not a learned container detector')
        from ui.inference import analyze_image
        sample = analyze_image(image, predictor, 'mocked_symbol_image')
        self.assertEqual(sample['prediction']['latex'], 'z')
        self.assertEqual(len(sample['steps']), 1)
        self.assertEqual(sample['source'], 'model prediction')

    def test_matching_penalizes_duplicate_predictions(self):
        truth = [symbol('x', 0)]
        predictions = [symbol('x', 0), symbol('x', 1), symbol('z', 70)]
        self.assertEqual(len(match_symbols(predictions, truth)), 1)


if __name__ == '__main__':
    unittest.main()
