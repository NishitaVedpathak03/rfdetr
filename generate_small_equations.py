"""Render a separate RF-DETR curriculum; no model, upload or training.

No data is generated on import. Run explicitly only when dataset generation is wanted.
"""
import argparse
import json
import random
from importlib.util import find_spec
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.symbol_data import SYMBOLS, export_records


def installed_fonts():
    candidates = ['C:/Windows/Fonts/times.ttf', 'C:/Windows/Fonts/timesi.ttf',
                  'C:/Windows/Fonts/cambria.ttc', 'C:/Windows/Fonts/cambriai.ttf',
                  'C:/Windows/Fonts/arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf',
                  '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    regular = [Path(p) for p in candidates if Path(p).is_file()]
    spec = find_spec('matplotlib')
    math_italic = None
    if spec and spec.submodule_search_locations:
        fonts = Path(next(iter(spec.submodule_search_locations))) / 'mpl-data/fonts/ttf'
        for name in ('DejaVuSerif.ttf', 'DejaVuSerif-Italic.ttf', 'STIXGeneral.ttf', 'STIXGeneralItalic.ttf'):
            if (fonts / name).is_file():
                regular.append(fonts / name)
        if (fonts / 'cmmi10.ttf').is_file():
            math_italic = fonts / 'cmmi10.ttf'
    if not regular:
        raise RuntimeError('No equation fonts found; supply a supported TrueType font.')
    return regular, math_italic


def integer_tokens(value):
    return [{'text': c} for c in str(value)]


def template(index, rng):
    # Every class gets three calibration images in training, including x/z.
    if index < 3 * len(SYMBOLS):
        text = SYMBOLS[index % len(SYMBOLS)]
        return [{'text': text}], 'glyph', text
    mode = index % 4
    variable = rng.choice('xyzab')
    if mode == 0:
        a, b = rng.randrange(21), rng.randrange(21)
        op = rng.choice('+-×')
        rhs = a + b if op == '+' else a - b if op == '-' else a * b
        tokens = integer_tokens(a) + [{'text': op}] + integer_tokens(b) + [{'text': '='}] + integer_tokens(rhs)
        latex = f'{a} ' + (r'\times' if op == '×' else op) + f' {b} = {rhs}'
        family = 'arithmetic'
    elif mode == 1:
        coefficient, constant, value = rng.randint(1, 9), rng.randrange(20), rng.randint(-9, 9)
        op = rng.choice('+-')
        rhs = coefficient * value + constant if op == '+' else coefficient * value - constant
        tokens = integer_tokens(coefficient) + [{'text': variable}, {'text': op}] + integer_tokens(constant) + [{'text': '='}] + integer_tokens(rhs)
        latex = f'{coefficient} {variable} {op} {constant} = {rhs}'
        family = 'linear'
    elif mode == 2:
        exponent, coefficient, constant = rng.choice((2, 3)), rng.randint(1, 9), rng.randint(0, 9)
        op = rng.choice('+-')
        tokens = [{'text': variable}, {'text': str(exponent), 'superscript_to': 0}, {'text': op}]
        tokens += integer_tokens(coefficient) + [{'text': variable}, {'text': '+'}] + integer_tokens(constant) + [{'text': '='}, {'text': '0'}]
        latex = f'{variable}^{{{exponent}}} {op} {coefficient} {variable} + {constant} = 0'
        family = 'polynomial'
    else:
        exponent, rhs = rng.choice((2, 3)), rng.randrange(31)
        tokens = [{'text': variable}, {'text': str(exponent), 'superscript_to': 0}, {'text': '='}] + integer_tokens(rhs)
        latex = f'{variable}^{{{exponent}}} = {rhs}'
        family = 'power'
    return tokens, family, latex


def render_symbols(tokens, font_path, size=54, math_italic=None, spacing=4):
    """Each visible character has its own tight box, including a two-stroke '='."""
    tiles, symbols, x = [], [], 10
    baseline = size + 14
    for index, token in enumerate(tokens):
        text = token['text']
        superscript = token.get('superscript_to') is not None
        font_size = round(size * .65) if superscript else size
        selected_font = math_italic if math_italic and text in 'xyzab' else font_path
        font = ImageFont.truetype(str(selected_font), font_size)
        bounds = font.getbbox(text)
        tile = Image.new('RGBA', (max(1, bounds[2] - bounds[0]) + 8, max(1, bounds[3] - bounds[1]) + 8))
        ImageDraw.Draw(tile).text((4 - bounds[0], 4 - bounds[1]), text, font=font, fill='black')
        ink = tile.getbbox()
        if ink is None:
            raise ValueError(f'Font did not render {text!r}')
        tile = tile.crop(ink)
        if superscript:
            base_index = token['superscript_to']
            if base_index >= index:
                raise ValueError('Exponent must follow its base')
            base = symbols[base_index]['bbox']
            x = max(x - spacing, base[2] + 1)
            y = max(3, base[1] - tile.height * .65)
        else:
            y = baseline - tile.height
            # Center operators relative to the bodies of regular glyphs.
            if text in '+-=×':
                y = baseline - size * .35 - tile.height / 2
            if text in '+-=×' and symbols:
                x += spacing + 3
        x, y = round(x), round(y)
        symbols.append({'text': text, 'bbox': [x, y, x + tile.width, y + tile.height]})
        tiles.append((tile, x, y))
        x += tile.width + spacing + (spacing + 3 if text in '+-=×' else 0)
    canvas = Image.new('RGB', (x + 10, baseline + 14), 'white')
    for tile, dx, dy in tiles:
        canvas.paste(tile, (dx, dy), tile)
    return canvas, symbols


def generate(out, count=2000, seed=20261013, side=512):
    if count < 100:
        raise ValueError('At least 100 images are needed for calibration and disjoint equation splits')
    if Path(out).exists() and any(Path(out).iterdir()):
        raise FileExistsError(f'Refusing to overwrite dataset: {out}')
    fonts, math_italic = installed_fonts()
    rng, records = random.Random(seed), []
    for index in range(count):
        tokens, family, latex = template(index, rng)
        font = rng.choice(fonts)
        use_math = math_italic if index % 3 == 0 else None
        image, symbols = render_symbols(tokens, font, rng.randint(40, 74), use_math, rng.randint(2, 6))
        key = json.dumps([(t['text'], t.get('superscript_to')) for t in tokens], ensure_ascii=False)
        records.append({'image': image, 'symbols': symbols, 'source': f'generated_{index:06d}',
                        'group': key, 'calibration': family == 'glyph', 'latex': latex})
    return export_records(records, out, side, seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='dataset_symbols_small')
    parser.add_argument('--count', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261013)
    args = parser.parse_args()
    print(json.dumps(generate(args.out, args.count, args.seed), ensure_ascii=True, indent=2))
