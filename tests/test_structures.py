"""Backward compatibility and extended geometry; never starts training."""
import contextlib
import io
import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from generate_data import extended_example, generate, installed_fonts
from src.data import validate_export
from src.layout import parse_small_equation
from src.model import SymbolPredictor, map_detections, predict_symbol_image
from src.profiles import SYMBOLS, EXTENDED_SYMBOLS, EXTENDED_PIPELINE
from src.render import text, join, fraction, square_root, scripts
from src.train import main
from src.validation import validate_hierarchy


PROJECT = Path(__file__).resolve().parents[1]


class StructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fonts, _ = installed_fonts()
        cls.font = cls.fonts[0]

    def draw(self, value):
        return text(value, self.font, 54)

    def check_drawing(self, drawing):
        symbols = [dict(s, confidence=.95) for s in drawing.symbols]
        parsed = parse_small_equation(symbols, 48, extended=True)
        self.assertTrue(parsed['layout_valid'], parsed['latex_metadata']['warnings'])
        self.assertEqual(parsed['latex'].replace(' ', ''), drawing.latex.replace(' ', ''))
        validate_hierarchy(parsed['nodes'])
        self.assertEqual(sum(n['terminal'] for n in parsed['nodes']), len(symbols))
        return parsed

    def test_default_outputs_match_the_original_working_parser(self):
        examples = json.loads((PROJECT / 'tests/fixtures/baseline_layout.json').read_text(encoding='utf-8'))
        for example in examples:
            self.assertEqual(parse_small_equation(example['symbols']), example['expected'])

    def test_original_class_ids_and_default_training_are_unchanged(self):
        self.assertEqual(SYMBOLS, tuple('0123456789xyzab+-=×'))
        self.assertEqual(EXTENDED_SYMBOLS[:len(SYMBOLS)], SYMBOLS)
        with patch('src.train.start_training', side_effect=AssertionError('Training must not start')), \
                contextlib.redirect_stdout(io.StringIO()):
            current = main([])
            extended = main(['--config', str(PROJECT / 'configs/extended.json')])
        self.assertEqual(current['classes'], list(SYMBOLS))
        self.assertEqual(extended['classes'], list(EXTENDED_SYMBOLS))
        self.assertNotEqual(current['output_dir'], extended['output_dir'])
        self.assertFalse(extended['training_started'])

    def test_fraction_has_numerator_bar_and_denominator(self):
        expression = join(fraction(join(self.draw('x'), self.draw('+'), self.draw('3')), self.draw('2')),
                          self.draw('='), self.draw('7'))
        parsed = self.check_drawing(expression)
        self.assertTrue(any(n['kind'] == 'fraction' for n in parsed['nodes']))
        self.assertTrue(any(n['role'] == 'numerator' for n in parsed['nodes']))
        self.assertTrue(any(n['role'] == 'denominator' for n in parsed['nodes']))
        original = parse_small_equation([dict(s, confidence=.95) for s in expression.symbols])
        self.assertFalse(original['layout_valid'])

    def test_roots_and_nested_fractions(self):
        self.check_drawing(square_root(join(self.draw('x'), self.draw('+'), self.draw('1'))))
        self.check_drawing(square_root(fraction(self.draw('x'), self.draw('2'))))
        self.check_drawing(square_root(square_root(fraction(self.draw('x'), self.draw('2')))))
        self.check_drawing(fraction(fraction(self.draw('x'), self.draw('2')), self.draw('3')))

    def test_subscripts_and_combined_scripts(self):
        for sub, sup in [('i', None), ('12', None), ('n', 2)]:
            parsed = self.check_drawing(scripts(self.draw('x'), self.font, sub=sub, sup=sup))
            self.assertTrue(any(n['role'] == 'subscript' for n in parsed['nodes']))

    def test_minus_is_not_turned_into_a_fraction(self):
        drawing = join(self.draw('2'), self.draw('x'), self.draw('-'), self.draw('3'), self.draw('='), self.draw('-7'))
        parsed = self.check_drawing(drawing)
        self.assertFalse(any(n['kind'] == 'fraction' for n in parsed['nodes']))

    def test_missing_or_ambiguous_structure_is_rejected(self):
        examples = [
            [{'text': '√', 'bbox': [0, 0, 20, 40], 'confidence': .99}],
            [{'text': 'x', 'bbox': [0, 0, 20, 40], 'confidence': .99},
             {'text': '-', 'bbox': [0, 60, 80, 62], 'confidence': .99}],
            [{'text': 'x', 'bbox': [0, 0, 20, 40], 'confidence': .99},
             {'text': 'i', 'bbox': [0, 60, 10, 80], 'confidence': .99}],
        ]
        for symbols in examples:
            parsed = parse_small_equation(symbols, extended=True)
            self.assertFalse(parsed['layout_valid'])
            self.assertEqual(parsed['latex'], '')

    def test_extended_font_curriculum_geometry(self):
        for font in self.fonts:
            for index in range(72, 96):
                drawing, _ = extended_example(index, random.Random(index), font, 54)
                self.check_drawing(drawing)

    def test_separate_extended_coco_export_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'extended'
            generate(folder, count=150, profile='extended')
            report = validate_export(folder)
            self.assertEqual(report['classes'], list(EXTENDED_SYMBOLS))
            self.assertEqual(sum(report['splits'].values()), 150)
            self.assertFalse(report['training_started'])
            with self.assertRaises(FileExistsError):
                generate(folder, count=150, profile='extended')

    def test_extended_model_adapter_uses_appended_class_ids(self):
        image = Image.new('RGB', (512, 512), 'white')
        identity = {'crop': [0, 0, 512, 512], 'sx': 1, 'sy': 1, 'left': 0, 'top': 0,
                    'width': 512, 'height': 512, 'side': 512}
        boxes = [[10, 10, 100, 80], [50, 30, 70, 70]]
        detections = SimpleNamespace(xyxy=np.array(boxes), class_id=np.array([19, 10]),
                                     confidence=np.array([.99, .99]), data={})
        mapped, _ = map_detections(detections, identity, image.size, EXTENDED_SYMBOLS)
        self.assertEqual(mapped[0]['text'], '√')
        predictor = SymbolPredictor(SimpleNamespace(predict=lambda *a, **k: detections),
                                    {'pipeline': EXTENDED_PIPELINE, 'resolution': 512}, Path('mock.pth'))
        result, _ = predict_symbol_image(image, predictor, prepared_image=True)
        self.assertEqual(result['latex'], r'\sqrt{x}')
        self.assertTrue(result['metadata']['layout_valid'])

    def test_extended_kaggle_package_is_separate_and_never_executed(self):
        from prepare_kaggle import SOURCE_FILES, prepare
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            for name in SOURCE_FILES:
                path = project / name
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(PROJECT / name, path)
            data = project / 'dataset_extended'
            generate(data, count=150, profile='extended')
            result = prepare(data, project=project, config=project / 'configs/extended.json')
            job = Path(result['job'])
            self.assertEqual(job.name, 'kaggle_rfdetr_extended')
            metadata = json.loads((job / 'kernel-metadata.json').read_text())
            self.assertEqual(metadata['id'], 'nishitavedpathak03/equation-ocr-rfdetr-extended')
            self.assertEqual(metadata['title'], 'Equation OCR RFDETR Extended')
            self.assertEqual(metadata['dataset_sources'], ['nishitavedpathak03/equation-ocr-extended-symbols'])
            script = (job / 'run.py').read_text(encoding='utf-8')
            compile(script, str(job / 'run.py'), 'exec')
            self.assertIn('CONFIG_FILE = "configs/extended.json"', script)
            self.assertFalse(result['training_started'])
            self.assertFalse(result['uploaded'])


if __name__ == '__main__':
    unittest.main()
