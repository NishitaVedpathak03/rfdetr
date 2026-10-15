"""Complex geometry/data/package preparation, without constructing RF-DETR."""
import contextlib
import io
import json
import random
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from generate_data import generate, installed_fonts
from prepare_kaggle import SOURCE_FILES, prepare
from src.curriculum import complex_example, prepared_example, screenshot_variant, FAMILIES
from src.data import validate_export
from src.layout import parse_small_equation
from src.profiles import COMPLEX_SYMBOLS, EXTENDED_SYMBOLS
from src.render import text, fraction, join, scripts, parentheses, function
from src.train import main
from src.validation import validate_hierarchy


PROJECT = Path(__file__).resolve().parents[1]


class ComplexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fonts, cls.italic = installed_fonts()
        cls.font = cls.fonts[0]
        cls.fallback = next(f for f in cls.fonts if f.name == 'STIXGeneral.ttf')

    def check_drawing(self, drawing):
        symbols = [dict(s, confidence=1.) for s in drawing.symbols]
        parsed = parse_small_equation(symbols, 96, complex_layout=True)
        self.assertTrue(parsed['layout_valid'], parsed['latex_metadata']['warnings'])
        self.assertEqual(parsed['latex'].replace(' ', ''), drawing.latex.replace(' ', ''))
        validate_hierarchy(parsed['nodes'])
        self.assertEqual(sum(n['terminal'] for n in parsed['nodes']), len(symbols))
        return parsed

    def test_existing_ids_and_preview_are_preserved(self):
        self.assertEqual(COMPLEX_SYMBOLS[:len(EXTENDED_SYMBOLS)], EXTENDED_SYMBOLS)
        self.assertEqual(len(COMPLEX_SYMBOLS), 31)
        with patch('src.train.start_training', side_effect=AssertionError('No training')), contextlib.redirect_stdout(io.StringIO()):
            plan = main(['--config', str(PROJECT / 'configs/complex.json')])
        self.assertFalse(plan['training_started'])
        self.assertFalse(plan['model_constructed'])
        self.assertEqual(Path(plan['output_dir']).name, 'rfdetr_complex')

    def test_math_font_calibration_radical_uses_a_supported_font(self):
        if not self.italic:
            self.skipTest('Computer Modern is unavailable')
        drawing, _, calibration = complex_example(COMPLEX_SYMBOLS.index('√'), random.Random(1),
                                                  self.italic.parent / 'cmr10.ttf', 54, self.italic, self.fallback)
        expected = text('√', self.fallback, 54, self.italic)
        self.assertTrue(calibration)
        self.assertEqual(drawing.image.tobytes(), expected.image.tobytes())

    def test_all_families_with_regular_and_math_italic_fonts(self):
        styles = [(f, None) for f in self.fonts]
        if self.italic:
            styles.append((self.italic.parent / 'cmr10.ttf', self.italic))
        for font, italic in styles:
            for index in range(93, 153):
                with self.subTest(font=font.name, index=index):
                    drawing, _, _ = complex_example(index, random.Random(index), font, 54, italic, self.fallback)
                    self.check_drawing(drawing)

    def test_every_family_has_math_italic_and_alternate_font_examples(self):
        if not self.italic:
            self.skipTest('Computer Modern is unavailable')
        coverage = {family: set() for family in FAMILIES}
        rng = random.Random(20261014)
        for index in range(93 + 6 * len(FAMILIES)):
            drawing, family, calibration, style = prepared_example(index, rng, self.fonts, self.italic, self.fallback)
            if not calibration:
                coverage[family].add(style == 'computer_modern_math_italic')
                self.check_drawing(drawing)
            screenshot_variant(drawing, rng)
        self.assertTrue(all(styles == {False, True} for styles in coverage.values()))

    def test_parentheses_on_separate_fraction_rows(self):
        draw = lambda s: text(s, self.font)
        numerator = function('sin', draw('x'), self.font)
        denominator = function('log', join(scripts(draw('x'), self.font, sup=2), draw('+'), draw('1')), self.font, base=2)
        self.check_drawing(fraction(numerator, denominator))

    def test_ordinary_parentheses_include_a_slightly_taller_exponent(self):
        # Unstretched TeX parentheses can start below the superscript's top.
        values = [('c', [195.5, 111.4, 207.4, 124.8]),
                  ('o', [206.8, 111.4, 220.3, 124.9]),
                  ('s', [220., 111.5, 230.5, 124.5]),
                  ('(', [232.1, 103.2, 239.6, 131.3]),
                  ('x', [240.7, 111.4, 254.4, 124.8]),
                  ('2', [255.1, 101.6, 264.4, 114.8]),
                  (')', [265.9, 103.1, 273.7, 131.6])]
        parsed = parse_small_equation([{'text': s, 'bbox': b, 'confidence': .95}
                                       for s, b in values], 96, complex_layout=True)
        self.assertTrue(parsed['layout_valid'], parsed['latex_metadata']['warnings'])
        self.assertEqual(parsed['latex'], r'\cos\left(x^{2}\right)')
        self.assertEqual(sum(n['terminal'] for n in parsed['nodes']), len(values))

    def test_nested_parentheses_and_compound_power(self):
        draw = lambda s: text(s, self.font)
        body = parentheses(parentheses(join(draw('x'), draw('+'), draw('1')), self.font), self.font)
        self.check_drawing(scripts(body, self.font, sup=2))

    def test_missing_parenthesis_or_function_argument_is_rejected(self):
        for value in ('(x', 'sin', 'sinx', 'log', 'cosx'):
            drawing = text(value, self.font)
            result = parse_small_equation([dict(s, confidence=1.) for s in drawing.symbols], 96, complex_layout=True)
            self.assertFalse(result['layout_valid'])
            self.assertEqual(result['latex'], '')
        drawing = function('sin', text('x', self.font), self.font)
        self.assertFalse(parse_small_equation([dict(s, confidence=1.) for s in drawing.symbols], 48, extended=True)['layout_valid'])

    def test_math_italic_height_script_positions(self):
        values = [('√', [0, 0, 103, 64]), ('-', [26, 36.6, 100, 38.1]),
                  ('x', [28, 14, 42, 26.5]), ('2', [44, 17.6, 53, 29.8]),
                  ('+', [61, 9.7, 79, 27.7]), ('1', [88, 7.5, 97, 25.8]),
                  ('x', [28, 50.7, 42, 63.2]), ('2', [44, 42.9, 52, 54.8]),
                  ('-', [62, 54.9, 79, 57]), ('1', [88, 44.8, 98, 62.8]),
                  ('=', [111, 33.8, 129, 40]), ('2', [139, 25.7, 151, 43.8])]
        parsed = parse_small_equation([{'text': s, 'bbox': b, 'confidence': 1.} for s, b in values], 96, complex_layout=True)
        self.assertTrue(parsed['layout_valid'], parsed['latex_metadata']['warnings'])
        self.assertEqual(parsed['latex'], r'\sqrt{\frac{x_{2} + 1}{x^{2} - 1}} = 2')

    def test_degradation_keeps_transformed_symbol_coordinates(self):
        drawing, _, _ = complex_example(96, random.Random(96), self.font, 54)
        transformed = screenshot_variant(drawing, random.Random(42))
        self.check_drawing(transformed)
        for s in transformed.symbols:
            x1, y1, x2, y2 = s['bbox']
            self.assertTrue(0 <= x1 < x2 <= transformed.image.width)
            self.assertTrue(0 <= y1 < y2 <= transformed.image.height)

    def test_dataset_and_standalone_package_do_not_start_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            for name in SOURCE_FILES:
                target = project / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(PROJECT / name, target)
            data = project / 'dataset_complex'
            with contextlib.redirect_stdout(io.StringIO()):
                generate(data, count=180, side=768, profile='complex')
            validated = validate_export(data)
            self.assertEqual(validated['classes'], list(COMPLEX_SYMBOLS))
            self.assertEqual(sum(validated['splits'].values()), 180)
            provenance = json.loads((data / 'preparation_report.json').read_text())
            self.assertEqual(set(FAMILIES), set(provenance['families']) - {'calibration'})
            result = prepare(data, project=project, config=project / 'configs/complex.json')
            metadata = json.loads((Path(result['job']) / 'kernel-metadata.json').read_text())
            self.assertEqual(metadata['id'], 'nishitavedpathak03/equation-ocr-rfdetr-complex')
            self.assertEqual(metadata['dataset_sources'], ['nishitavedpathak03/equation-ocr-complex-symbols'])
            with zipfile.ZipFile(Path(result['upload']) / 'dataset.zip') as archive:
                self.assertIn('dataset_complex/preparation_report.json', archive.namelist())
            self.assertFalse(result['training_started'])
            self.assertFalse(result['uploaded'])


if __name__ == '__main__':
    unittest.main()
