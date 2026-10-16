"""Calculus geometry, isolated data and explicit training safeguards."""
import contextlib
import io
import json
import random
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from generate_data import generate, installed_fonts
from prepare_kaggle import SOURCE_FILES, prepare
from src.calculus_data import FAMILIES, calculus_example
from src.data import letterbox_image, transform_box, validate_export
from src.layout import parse_small_equation
from src.model import SymbolPredictor, predict_symbol_image
from src.profiles import CALCULUS_PIPELINE, CALCULUS_SYMBOLS, COMPLEX_SYMBOLS
from src.render import text, join, scripts, fraction, function, large_operator
from src.train import main
from src.validation import validate_hierarchy
from src.visualize import recursive_steps
from ui.steps import normalize_nodes, validate_steps


PROJECT = Path(__file__).resolve().parents[1]


class CalculusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fonts, cls.italic = installed_fonts()
        cls.font = cls.fonts[0]
        cls.fallback = next(f for f in cls.fonts if f.name == 'STIXGeneral.ttf')

    def draw(self, value, scale=1):
        return text(value, self.fallback if value in {'∫', '∑', '∂', '∞'} else self.font, round(54 * scale))

    def parse(self, drawing):
        result = parse_small_equation([dict(s, confidence=.95) for s in drawing.symbols], 128, calculus_layout=True)
        self.assertTrue(result['layout_valid'], (drawing.latex, result['latex_metadata']['warnings']))
        self.assertEqual(result['latex'].replace(' ', ''), drawing.latex.replace(' ', ''))
        validate_hierarchy(result['nodes'])
        terminals = [n for n in result['nodes'] if n['terminal']]
        self.assertEqual(len(terminals), len(drawing.symbols))
        self.assertEqual(len({n['detection_index'] for n in terminals}), len(terminals))
        return result

    def test_existing_class_ids_and_training_preview_are_preserved(self):
        self.assertEqual(CALCULUS_SYMBOLS[:len(COMPLEX_SYMBOLS)], COMPLEX_SYMBOLS)
        self.assertEqual(len(CALCULUS_SYMBOLS), 39)
        self.assertEqual(len(set(CALCULUS_SYMBOLS)), len(CALCULUS_SYMBOLS))
        with patch('src.train.start_training', side_effect=AssertionError('No training')), contextlib.redirect_stdout(io.StringIO()):
            plan = main(['--config', str(PROJECT / 'configs/calculus.json')])
        self.assertFalse(plan['training_started'])
        self.assertFalse(plan['model_constructed'])
        self.assertFalse(plan['weights_downloaded'])
        self.assertEqual(plan['training']['epochs'], 30)
        self.assertEqual(Path(plan['output_dir']).name, 'rfdetr_calculus')

    def test_all_families_across_fonts_and_sizes(self):
        styles = [(f, None) for f in self.fonts]
        if self.italic:
            styles.append((self.italic.parent / 'cmr10.ttf', self.italic))
        start = 3 * len(CALCULUS_SYMBOLS)
        for font, italic in styles:
            for size in (42, 54, 68):
                for index in range(start, start + len(FAMILIES) * 2):
                    with self.subTest(font=font.name, size=size, family=FAMILIES[(index - start) % len(FAMILIES)]):
                        drawing, _, _ = calculus_example(index, random.Random(index), font, size, italic, self.fallback)
                        self.parse(drawing)

    def test_integral_limits_differential_and_ui_steps(self):
        drawing = large_operator('∫', function('tan', self.draw('x'), self.font), self.fallback,
                                 lower=self.draw('0', .55), upper=self.draw('1', .55),
                                 differential=join(self.draw('d'), self.draw('x'), gap=3))
        result = self.parse(drawing)
        self.assertIn('lower_limit', {n['role'] for n in result['nodes']})
        self.assertIn('upper_limit', {n['role'] for n in result['nodes']})
        nodes = normalize_nodes(result['nodes'], drawing.image.size)
        steps = recursive_steps(nodes)
        self.assertEqual(validate_steps(steps, nodes), steps)

    def test_multiple_integrals_keep_separate_differentials(self):
        first = large_operator('∫', self.draw('x'), self.fallback,
                               differential=join(self.draw('d'), self.draw('x'), gap=3))
        second = large_operator('∫', self.draw('y'), self.fallback,
                                differential=join(self.draw('d'), self.draw('y'), gap=3))
        self.parse(join(first, self.draw('+'), second))

    def test_missing_limits_differential_and_sum_index_are_rejected(self):
        integral = large_operator('∫', self.draw('x'), self.fallback,
                                  lower=self.draw('0', .55), upper=self.draw('1', .55),
                                  differential=join(self.draw('d'), self.draw('x'), gap=3))
        summation = large_operator('∑', scripts(self.draw('i'), self.font, sup=2), self.fallback,
                                   lower=join(self.draw('i', .55), self.draw('=', .55), self.draw('1', .55), gap=1),
                                   upper=self.draw('n', .55))
        cases = [[s for s in integral.symbols if s['text'] != 'd'],
                 [s for s in integral.symbols if s['text'] != '0'],
                 [s for s in summation.symbols if s['text'] != '=']]
        for symbols in cases:
            result = parse_small_equation([dict(s, confidence=1.) for s in symbols], 128, calculus_layout=True)
            self.assertFalse(result['layout_valid'])
            self.assertEqual(result['latex'], '')
        old = parse_small_equation([dict(s, confidence=1.) for s in integral.symbols], 96, complex_layout=True)
        self.assertFalse(old['layout_valid'])

    def test_calculus_backend_dispatch_does_not_construct_a_model(self):
        drawing = function('tan', self.draw('x'), self.font)
        _, geometry = letterbox_image(drawing.image, 768)
        detections = SimpleNamespace(xyxy=np.array([transform_box(s['bbox'], geometry) for s in drawing.symbols]),
                                     class_id=np.array([CALCULUS_SYMBOLS.index(s['text']) for s in drawing.symbols]),
                                     confidence=np.full(len(drawing.symbols), .95), data={})
        backend = SimpleNamespace(predict=lambda image, threshold: detections)
        predictor = SymbolPredictor(backend, {'pipeline': CALCULUS_PIPELINE, 'resolution': 768}, Path('fake.pth'))
        result, _ = predict_symbol_image(drawing.image, predictor, max_symbols=128)
        self.assertEqual(result['latex'], drawing.latex)
        self.assertEqual(result['metadata']['layout_mode'], 'calculus')

    def test_export_padding_and_kaggle_package_never_start_training(self):
        with tempfile.TemporaryDirectory(prefix='calculus_test_') as temporary:
            project = Path(temporary)
            for name in SOURCE_FILES:
                target = project / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(PROJECT / name, target)
            data = project / 'dataset_calculus'
            with contextlib.redirect_stdout(io.StringIO()):
                generate(data, count=320, side=768, profile='calculus')
            self.assertEqual(sum(validate_export(data)['splits'].values()), 320)
            report = json.loads((data / 'preparation_report.json').read_text())
            self.assertEqual(set(report['families']) - {'calibration'}, set(FAMILIES))
            self.assertTrue(report['font_family_coverage_verified'])
            self.assertTrue(report['special_glyph_coverage_verified'])
            self.assertFalse(report['training_started'])
            for split in ('train', 'valid', 'test'):
                coco = json.loads((data / split / '_annotations.coco.json').read_text(encoding='utf-8'))
                present = {CALCULUS_SYMBOLS[a['category_id']] for a in coco['annotations']}
                self.assertTrue(set(CALCULUS_SYMBOLS[len(COMPLEX_SYMBOLS):]) <= present)
                refs = {r['image_id']: r for r in json.loads((data / split / 'references.json').read_text())}
                by_image = {i['id']: [] for i in coco['images']}
                for a in coco['annotations']:
                    self.assertIn('tight_bbox', a)
                    x, y, w, h = a['bbox']
                    by_image[a['image_id']].append({'text': CALCULUS_SYMBOLS[a['category_id']],
                                                   'bbox': [x, y, x + w, y + h], 'confidence': 1.})
                for identity, symbols in by_image.items():
                    if refs[identity]['calibration']:
                        continue
                    parsed = parse_small_equation(symbols, 128, calculus_layout=True)
                    self.assertTrue(parsed['layout_valid'], parsed['latex_metadata']['warnings'])
                    self.assertEqual(parsed['latex'].replace(' ', ''), refs[identity]['latex'].replace(' ', ''))
            packaged = prepare(data, project=project, config=project / 'configs/calculus.json')
            source = (Path(packaged['job']) / 'run.py').read_text()
            compile(source, 'calculus_job.py', 'exec')
            metadata = json.loads((Path(packaged['job']) / 'kernel-metadata.json').read_text())
            self.assertEqual(metadata['id'], 'nishitavedpathak03/equation-ocr-rfdetr-calculus')
            with zipfile.ZipFile(Path(packaged['upload']) / 'dataset.zip') as archive:
                self.assertIn('dataset_calculus/preparation_report.json', archive.namelist())
                self.assertFalse(any(n.endswith(('.pth', '.pt', '.ckpt')) for n in archive.namelist()))
            self.assertFalse(packaged['training_started'])
            self.assertFalse(packaged['uploaded'])


if __name__ == '__main__':
    unittest.main()
