"""Small, composable synthetic equation drawings with per-symbol boxes."""
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont


@dataclass
class Drawing:
    image: Image.Image
    symbols: list
    latex: str
    axis: float


def paste(canvas, drawing, x, y):
    canvas.paste(drawing.image, (round(x), round(y)))
    return [{'text': s['text'], 'bbox': [v + (x if i % 2 == 0 else y)
             for i, v in enumerate(s['bbox'])]} for s in drawing.symbols]


def text(value, font_path, size=54, math_italic=None):
    font = ImageFont.truetype(str(font_path), size)
    italic = ImageFont.truetype(str(math_italic), size) if math_italic else None
    glyphs = []
    for letter in str(value):
        selected = italic if italic and letter in 'xyzabijkn' else font
        box = selected.getbbox(letter)
        tile = Image.new('RGBA', (box[2] - box[0] + 8, box[3] - box[1] + 8))
        ImageDraw.Draw(tile).text((4 - box[0], 4 - box[1]), letter, font=selected, fill='black')
        if tile.getbbox() is None:
            raise ValueError(f'Font cannot draw {letter}')
        glyphs.append((letter, tile.crop(tile.getbbox())))
    bottom = max(max(tile.height for _, tile in glyphs) + 5, round(size * .7) + 5)
    axis = bottom - size * .35
    placements, symbols, x = [], [], 5
    for letter, tile in glyphs:
        y = round(axis - tile.height / 2) if letter in '+-=×' else bottom - tile.height
        symbols.append({'text': letter, 'bbox': [x, y, x + tile.width, y + tile.height]})
        placements.append((tile, x, y))
        x += tile.width + 4
    canvas = Image.new('RGB', (x + 5, bottom + 8), 'white')
    for tile, x, y in placements:
        canvas.paste(tile, (x, y), tile)
    return Drawing(canvas, symbols, str(value).replace('×', r'\times'), axis)


def join(*parts, gap=6):
    axis = max(p.axis for p in parts)
    height = round(axis + max(p.image.height - p.axis for p in parts) + 2)
    width = sum(p.image.width for p in parts) + gap * (len(parts) - 1)
    canvas = Image.new('RGB', (width, height), 'white')
    symbols, x = [], 0
    for part in parts:
        y = round(axis - part.axis)
        symbols.extend(paste(canvas, part, x, y))
        x += part.image.width + gap
    return Drawing(canvas, symbols, ' '.join(p.latex for p in parts), axis)


def fraction(numerator, denominator):
    width = max(numerator.image.width, denominator.image.width) + 12
    line_y = numerator.image.height + 5
    lower_y = line_y + 7
    canvas = Image.new('RGB', (width, lower_y + denominator.image.height + 3), 'white')
    symbols = paste(canvas, numerator, (width - numerator.image.width) // 2, 0)
    ImageDraw.Draw(canvas).rectangle((3, line_y, width - 4, line_y + 1), fill='black')
    symbols.append({'text': '-', 'bbox': [3, line_y, width - 3, line_y + 2]})
    symbols.extend(paste(canvas, denominator, (width - denominator.image.width) // 2, lower_y))
    return Drawing(canvas, symbols, r'\frac{' + numerator.latex + '}{' + denominator.latex + '}', line_y + 1)


def square_root(body, stroke=2):
    width, height = body.image.width + 30, body.image.height + 12
    canvas = Image.new('RGB', (width, height), 'white')
    symbols = paste(canvas, body, 24, 8)
    ImageDraw.Draw(canvas).line([(2, round(body.axis + 12)), (7, round(body.axis + 8)),
                                (13, height - 4), (23, 3), (width - 3, 3)], fill='black', width=stroke)
    symbols.insert(0, {'text': '√', 'bbox': [1, 2, width - 2, height - 2]})
    return Drawing(canvas, symbols, r'\sqrt{' + body.latex + '}', body.axis + 8)


def scripts(base, font_path, size=54, sub=None, sup=None, math_italic=None):
    if sub is None and sup is None:
        return base
    sub_draw = text(sub, font_path, round(size * .55), math_italic) if sub is not None else None
    sup_draw = text(sup, font_path, round(size * .55), math_italic) if sup is not None else None
    last = base.symbols[-1]['bbox']
    top_y = round(last[1] - sup_draw.image.height * .65) if sup_draw else 0
    pad_top = max(0, -top_y)
    low_y = round(last[3] - sub_draw.image.height * .1) if sub_draw else 0
    width = base.image.width + max(p.image.width for p in (sub_draw, sup_draw) if p)
    height = pad_top + max(base.image.height, low_y + sub_draw.image.height if sub_draw else 0)
    canvas = Image.new('RGB', (width, height), 'white')
    symbols = paste(canvas, base, 0, pad_top)
    if sub_draw:
        symbols.extend(paste(canvas, sub_draw, base.image.width - 3, low_y + pad_top))
    if sup_draw:
        symbols.extend(paste(canvas, sup_draw, base.image.width - 3, top_y + pad_top))
    latex = base.latex + ('_{' + str(sub) + '}' if sub is not None else '')
    latex += '^{' + str(sup) + '}' if sup is not None else ''
    return Drawing(canvas, symbols, latex, base.axis + pad_top)


def parentheses(body, font_path, size=54):
    """Stretch two separately labelled parentheses around a compound body."""
    parts = []
    for mark in '()':
        glyph = text(mark, font_path, size)
        box = glyph.symbols[0]['bbox']
        tile = glyph.image.crop(tuple(round(v) for v in box))
        tile = tile.resize((tile.width, body.image.height), Image.Resampling.LANCZOS)
        parts.append(Drawing(tile, [{'text': mark, 'bbox': [0, 0, *tile.size]}], mark, body.axis))
    result = join(parts[0], body, parts[1], gap=3)
    result.latex = r'\left(' + body.latex + r'\right)'
    return result


def function(name, body, font_path, size=54, base=None):
    if name not in {'sin', 'cos', 'log'} or (base is not None and name != 'log'):
        raise ValueError('Supported functions are sin, cos and optionally base-labelled log')
    label = text(name, font_path, size)
    if base is not None:
        label = scripts(label, font_path, size, sub=base)
    argument = parentheses(body, font_path, size)
    result = join(label, argument)
    result.latex = '\\' + name + ('_{' + str(base) + '}' if base is not None else '') + argument.latex
    return result
