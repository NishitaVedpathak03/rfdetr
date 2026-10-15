"""Complex synthetic equations; explicit preparation only, never model execution."""
import io
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageFilter

from .data import export_records, write_json
from .layout import parse_small_equation
from .profiles import COMPLEX_PIPELINE, COMPLEX_SYMBOLS
from .render import Drawing, text, join, fraction, square_root, scripts, parentheses, function


FAMILIES = ('basic', 'script_contrast', 'scripted_fraction', 'root_fraction',
            'nested_fraction', 'nested_roots', 'parenthesized_power', 'trigonometry',
            'logarithm', 'mixed_functions', 'rational_expression', 'function_fraction')


def complex_example(index, rng, font, size, math_italic=None, fallback_font=None):
    # Every symbol has isolated calibration examples, confined to training.
    draw = lambda value: text(value, fallback_font if str(value) in {'×', '√'} and fallback_font else font,
                              size, math_italic)
    if index < 3 * len(COMPLEX_SYMBOLS):
        return draw(COMPLEX_SYMBOLS[index % len(COMPLEX_SYMBOLS)]), 'calibration', True
    variable = rng.choice('xyzab')
    number, divisor, result = rng.randint(1, 19), rng.randint(2, 9), rng.randint(-9, 19)
    sub = rng.choice(('i', 'j', 'k', 'n', '2', '3', '12'))
    power = rng.choice((2, 3, 4))
    scripted = lambda value=variable, lower=None, upper=None: scripts(
        draw(value), font, size, sub=lower, sup=upper, math_italic=math_italic)
    add = lambda first, op='+', value=number: join(first, draw(op), draw(value))
    family = FAMILIES[(index - 3 * len(COMPLEX_SYMBOLS)) % len(FAMILIES)]
    if family == 'basic':
        expression = add(join(draw(divisor), draw(variable)), rng.choice('+-×'))
    elif family == 'script_contrast':
        expression = join(scripted(lower=sub, upper=power), draw('+'), draw('x'), draw('-'), draw('a'), draw('+'), draw('n'))
    elif family in {'scripted_fraction', 'root_fraction'}:
        expression = fraction(add(scripted(lower=sub)), add(scripted(upper=power), '-', divisor))
        if family == 'root_fraction':
            expression = square_root(expression)
    elif family == 'nested_fraction':
        expression = fraction(add(fraction(add(draw(variable)), draw(divisor))), add(scripted(upper=power), '-', divisor))
    elif family == 'nested_roots':
        expression = square_root(add(square_root(add(scripted(lower=sub))), value=divisor))
    elif family == 'parenthesized_power':
        body = parentheses(add(join(scripted(upper=power), draw('+'), draw(divisor), draw(variable))), font, size)
        expression = scripts(body, font, size, sup=2)
    elif family == 'trigonometry':
        expression = function(rng.choice(('sin', 'cos')), fraction(add(scripted(lower=sub)), draw(divisor)), font, size)
    elif family == 'logarithm':
        expression = function('log', add(scripted(upper=power)), font, size, base=divisor)
    elif family == 'mixed_functions':
        expression = join(function('sin', draw(variable), font, size), draw('+'),
                          function('cos', scripted(upper=power), font, size), draw('-'),
                          fraction(scripted(lower=sub), draw(divisor)))
    elif family == 'rational_expression':
        expression = join(fraction(add(scripted(upper=power)), parentheses(add(draw(variable), '-', divisor), font, size)),
                          draw('+'), square_root(add(draw(variable))))
    else:
        expression = fraction(add(function('sin', draw(variable), font, size)),
                              function('log', add(scripted(upper=power), value=divisor), font, size, base=2))
    return join(expression, draw('='), draw(result)), family, False


def screenshot_variant(drawing, rng):
    """Generated-image degradation; keep every label under the exact transform."""
    factor = rng.choice((.45, .6, .8, 1.0))
    width, height = max(1, round(drawing.image.width * factor)), max(1, round(drawing.image.height * factor))
    image = drawing.image.resize((width, height), Image.Resampling.LANCZOS)
    if rng.random() < .35:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(.1, .4)))
    if rng.random() < .35:
        stream = io.BytesIO()
        image.save(stream, format='JPEG', quality=rng.randint(65, 95))
        with Image.open(io.BytesIO(stream.getvalue())) as jpeg:
            image = jpeg.convert('RGB')
    sx, sy = width / drawing.image.width, height / drawing.image.height
    symbols = [{'text': s['text'], 'bbox': [v * (sx if i % 2 == 0 else sy) for i, v in enumerate(s['bbox'])]}
               for s in drawing.symbols]
    return Drawing(image, symbols, drawing.latex, drawing.axis * sy)


def prepared_example(index, rng, fonts, math_italic, fallback):
    """Choose styles by curriculum round, so every family gets math italic fonts."""
    roman = math_italic.parent / 'cmr10.ttf' if math_italic else None
    available = bool(roman and roman.is_file())
    original_math = available and index % 3 != 2
    font, italic = (roman, math_italic) if original_math else (rng.choice(fonts), None)
    size = rng.randint(36, 72)
    formula_state = rng.getstate()
    drawing, family, calibration = complex_example(index, rng, font, size, italic, fallback)
    use_math = original_math if calibration else available and ((index - 3 * len(COMPLEX_SYMBOLS)) // len(FAMILIES)) % 3 != 2
    if use_math != original_math:
        font, italic = (roman, math_italic) if use_math else (fonts[(index // (3 * len(FAMILIES))) % len(fonts)], None)
        formula_rng = random.Random()
        formula_rng.setstate(formula_state)
        drawing, family, calibration = complex_example(index, formula_rng, font, size, italic, fallback)
    style = 'computer_modern_math_italic' if use_math else font.name
    return drawing, family, calibration, style


def generate_complex(out, count, seed, side, fonts, math_italic, box_padding=2):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite dataset: {out}')
    if count < 160:
        raise ValueError('Complex data needs at least 160 images for calibration and equation splits')
    rng, records, families, styles = random.Random(seed), [], Counter(), Counter()
    family_styles = defaultdict(Counter)
    fallback = next((f for f in fonts if f.name == 'STIXGeneral.ttf'), fonts[0])
    # Keep only provenance in RAM; original drawings are streamed through temporary PNGs.
    with tempfile.TemporaryDirectory(prefix='equation_curriculum_') as temporary:
        for index in range(count):
            drawing, family, calibration, style = prepared_example(index, rng, fonts, math_italic, fallback)
            if not calibration:
                symbols = [dict(s, confidence=1.) for s in drawing.symbols]
                parsed = parse_small_equation(symbols, 96, complex_layout=True)
                if not parsed['layout_valid'] or parsed['latex'].replace(' ', '') != drawing.latex.replace(' ', ''):
                    raise ValueError(f'Invalid generated geometry ({family}, {index}): {parsed["latex_metadata"]["warnings"]}; {parsed["latex"]!r} != {drawing.latex!r}')
            drawing = screenshot_variant(drawing, rng)
            path = Path(temporary) / f'{index:06d}.png'
            drawing.image.save(path)
            records.append({'path': str(path), 'symbols': drawing.symbols,
                            'source': f'complex_{family}_{style}_{index:06d}', 'group': drawing.latex,
                            'calibration': calibration, 'latex': drawing.latex})
            families[family] += 1
            styles[style] += 1
            family_styles[family][style] += 1
            if (index + 1) % 500 == 0:
                print(f'Prepared {index + 1}/{count} complex drawings; no training.', flush=True)
        report = export_records(records, out, side, seed, COMPLEX_PIPELINE, box_padding=box_padding)
    write_json(out / 'preparation_report.json', {
        'images': count, 'seed': seed, 'resolution': side, 'families': dict(families),
        'font_styles': dict(styles), 'geometry_reference_checks': count - families['calibration'],
        'family_font_styles': {k: dict(v) for k, v in family_styles.items()},
        'font_family_coverage_verified': all('computer_modern_math_italic' in v and len(v) > 1
                                           for k, v in family_styles.items() if k != 'calibration'),
        'radical_calibration_font': fallback.name,
        'annotation_box_padding_pixels': box_padding,
        'annotation_box_padding_units': 'per side, in prepared-image pixels; clipped to image boundaries',
        'image_pixels_changed_by_box_padding': False,
        'source_screenshot_added': False, 'training_started': False,
        'note': 'Generated labels and geometry verified before degradation; no recognition accuracy is claimed.'})
    return report
