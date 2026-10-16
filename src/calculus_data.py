"""Explicit calculus data preparation. No model import, download or training."""
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from .curriculum import complex_example, screenshot_variant
from .data import export_records, write_json
from .layout import parse_small_equation
from .profiles import CALCULUS_PIPELINE, CALCULUS_SYMBOLS, COMPLEX_SYMBOLS
from .render import text, join, scripts, fraction, function, parentheses, large_operator


FAMILIES = ('basic_replay', 'fraction_root_replay', 'complex_replay',
            'trig_single', 'trig_mixed', 'trig_fraction',
            'integral_power', 'integral_trig', 'definite_integral',
            'derivative_operator', 'derivative_fraction', 'second_derivative',
            'partial_derivative', 'summation_power', 'summation_expression', 'infinite_sum')


def calculus_example(index, rng, font, size, math_italic, fallback):
    draw = lambda value, scale=1: text(value, fallback if str(value) in {'×', '√', '∫', '∑', '∂', '∞'} else font,
                                      round(size * scale), math_italic)
    calibration = 3 * len(CALCULUS_SYMBOLS)
    if index < calibration:
        return draw(CALCULUS_SYMBOLS[index % len(CALCULUS_SYMBOLS)]), 'calibration', True
    family = FAMILIES[(index - calibration) % len(FAMILIES)]
    if family.endswith('_replay'):
        choices = {'basic_replay': (0, 1), 'fraction_root_replay': (2, 3, 4, 5),
                   'complex_replay': (6, 7, 8, 9, 10, 11)}
        replay_index = 3 * len(COMPLEX_SYMBOLS) + rng.choice(choices[family])
        drawing, _, _ = complex_example(replay_index, rng, font, size, math_italic, fallback)
        return drawing, family, False
    variable = rng.choice('xz' if family in {'derivative_fraction', 'second_derivative'} else 'xyz')
    power, constant, coefficient = rng.randint(2, 5), rng.randint(1, 19), rng.randint(2, 9)
    base = draw(variable)
    raised = lambda value=base, exponent=power: scripts(value, font, size, sup=exponent, math_italic=math_italic)
    trig = lambda name, body=base: function(name, body, font, size)
    differential = join(draw('d'), base, gap=3)
    integral = lambda body, lower=None, upper=None: large_operator(
        '∫', body, fallback, size, lower=lower, upper=upper, differential=differential)
    if family == 'trig_single':
        body = join(base, draw('+'), draw(constant))
        drawing = function(rng.choice(('sin', 'cos', 'tan')), body, font, size)
    elif family == 'trig_mixed':
        drawing = join(trig('sin'), draw('+'), trig('cos', raised()), draw('-'), trig('tan'))
    elif family == 'trig_fraction':
        drawing = function('tan', fraction(join(raised(), draw('+'), draw(constant)), draw(coefficient)), font, size)
    elif family == 'integral_power':
        answer = fraction(raised(exponent=power + 1), draw(power + 1))
        drawing = join(integral(raised()), draw('='), answer, draw('+'), draw('C'))
    elif family == 'integral_trig':
        name = rng.choice(('sin', 'cos', 'tan'))
        drawing = integral(trig(name))
        if name != 'tan':
            answer = trig('sin') if name == 'cos' else join(draw('-'), trig('cos'))
            drawing = join(drawing, draw('='), answer, draw('+'), draw('C'))
    elif family == 'definite_integral':
        bound = rng.randint(2, 9)
        drawing = join(integral(base, draw('0', .55), draw(bound, .55)), draw('='), fraction(draw(bound * bound), draw('2')))
    elif family == 'derivative_operator':
        operator = fraction(draw('d'), differential)
        argument = parentheses(join(raised(), draw('+'), draw(constant)), font, size)
        drawing = join(operator, argument, draw('='), draw(power), raised(exponent=power - 1))
    elif family == 'derivative_fraction':
        numerator = join(draw('d'), draw('y'), gap=3)
        drawing = join(fraction(numerator, differential), draw('='), draw(coefficient), raised(), draw('+'), draw(constant))
    elif family == 'second_derivative':
        numerator = join(raised(draw('d'), 2), draw('y'), gap=3)
        denominator = join(draw('d'), raised(exponent=2), gap=3)
        drawing = join(fraction(numerator, denominator), draw('='), draw(coefficient), base)
    elif family == 'partial_derivative':
        operator = fraction(draw('∂'), join(draw('∂'), base, gap=3))
        body = function('f', join(base, draw('+'), draw(constant)), font, size)
        drawing = join(operator, body)
    else:
        index_variable = rng.choice('ijkn')
        first_index = 1 if family == 'infinite_sum' else rng.randint(0, 3)
        lower = join(draw(index_variable, .55), draw('=', .55), draw(first_index, .55), gap=1)
        upper = draw('∞' if family == 'infinite_sum' else rng.choice(('n', str(coefficient))), .55)
        index_base = draw(index_variable)
        term = raised(index_base)
        if family == 'summation_expression':
            term = parentheses(join(term, draw('+'), draw(constant)), font, size)
        elif family == 'infinite_sum':
            term = parentheses(fraction(draw('1'), term), font, size)
        drawing = large_operator('∑', term, fallback, size, lower=lower, upper=upper)
    return drawing, family, False


def prepared_example(index, rng, fonts, math_italic, fallback):
    calibration = 3 * len(CALCULUS_SYMBOLS)
    use_math = bool(math_italic) and ((max(0, index - calibration) // len(FAMILIES)) % 3 != 2)
    font = math_italic.parent / 'cmr10.ttf' if use_math else fonts[(index // len(FAMILIES)) % len(fonts)]
    italic = math_italic if use_math else None
    drawing, family, isolated = calculus_example(index, rng, font, rng.randint(42, 68), italic, fallback)
    return drawing, family, isolated, 'computer_modern_math_italic' if use_math else font.name


def generate_calculus(out, count, seed, side, fonts, math_italic, box_padding=2):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite dataset: {out}')
    if count < 300:
        raise ValueError('Calculus preparation needs at least 300 images for all families and disjoint splits')
    fallback = next((f for f in fonts if f.name == 'STIXGeneral.ttf'), None)
    if fallback is None:
        raise RuntimeError('Install a STIXGeneral font before preparing calculus glyphs')
    # Check actual character coverage, rather than accepting a missing-glyph box.
    from fontTools.ttLib import TTFont
    with TTFont(str(fallback)) as font:
        cmap = font.getBestCmap()
        if any(ord(c) not in cmap for c in CALCULUS_SYMBOLS):
            raise ValueError('STIXGeneral does not cover the requested symbol vocabulary')
    rng, records = random.Random(seed), []
    families, styles, coverage = Counter(), Counter(), defaultdict(Counter)
    with tempfile.TemporaryDirectory(prefix='calculus_curriculum_') as temporary:
        for index in range(count):
            drawing, family, isolated, style = prepared_example(index, rng, fonts, math_italic, fallback)
            if not isolated:
                parsed = parse_small_equation([dict(s, confidence=1.) for s in drawing.symbols], 128, calculus_layout=True)
                if not parsed['layout_valid'] or parsed['latex'].replace(' ', '') != drawing.latex.replace(' ', ''):
                    raise ValueError(f'Calculus geometry failed ({family}, {index}): {parsed["latex_metadata"]["warnings"]}; '
                                     f'{parsed["latex"]!r} != {drawing.latex!r}')
            drawing = screenshot_variant(drawing, rng)
            path = Path(temporary) / f'{index:06d}.png'
            drawing.image.save(path)
            records.append({'path': str(path), 'symbols': drawing.symbols,
                            'source': f'calculus_{family}_{style}_{index:06d}', 'group': drawing.latex,
                            'calibration': isolated, 'latex': drawing.latex})
            families[family] += 1
            styles[style] += 1
            coverage[family][style] += 1
            if (index + 1) % 500 == 0:
                print(f'Prepared {index + 1}/{count} calculus images; no training.', flush=True)
        report = export_records(records, out, side, seed, CALCULUS_PIPELINE, box_padding=box_padding,
                                required_classes=CALCULUS_SYMBOLS[len(COMPLEX_SYMBOLS):])
    write_json(out / 'preparation_report.json', {
        'images': count, 'seed': seed, 'resolution': side, 'families': dict(families),
        'font_styles': dict(styles), 'family_font_styles': {k: dict(v) for k, v in coverage.items()},
        'geometry_reference_checks': count - families['calibration'],
        'font_family_coverage_verified': all('computer_modern_math_italic' in v and len(v) > 1
                                           for k, v in coverage.items() if k != 'calibration'),
        'special_glyph_font': fallback.name, 'special_glyph_coverage_verified': True,
        'heldout_new_class_coverage_verified': all(
            set(CALCULUS_SYMBOLS[len(COMPLEX_SYMBOLS):]) <= set(split['class_counts'])
            for split in report['splits'].values()),
        'annotation_box_padding_pixels': box_padding, 'image_pixels_changed_by_box_padding': False,
        'source_screenshot_added': False, 'training_started': False, 'uploaded': False,
        'note': 'OCR notation only; generated geometry checks are not neural-model accuracy.'})
    return report
