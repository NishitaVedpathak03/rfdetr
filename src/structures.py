"""Optional spatial grouping for fractions, radical bars and simple scripts.

The original parser remains the default. This module uses detected boxes only;
it never reads reference LaTeX or fills in missing symbols.
"""
import statistics

from .layout import LayoutError, _Parser, _container, _leaf, _number

OPERATORS = set('+-=×')
VARIABLES = set('xyzabijkn')
MAX_DEPTH = 4


def center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def leaf_height(ast):
    if not ast['children']:
        return 0 if ast['text'] in OPERATORS | {'√'} else ast['bbox'][3] - ast['bbox'][1]
    return max(leaf_height(child) for child in ast['children'])


def atom(ast, identity, anchor):
    return {'text': '@', 'bbox': ast['bbox'], 'detection_index': identity,
            '_ast': ast, '_anchor': anchor, '_height': max(1, leaf_height(ast))}


def fraction_groups(items, depth):
    """A bar needs horizontally covered operands both above and below it."""
    remaining = list(items)
    bars = sorted((s for s in items if s['text'] == '-'),
                  key=lambda s: s['bbox'][2] - s['bbox'][0], reverse=True)
    for bar in bars:
        if bar not in remaining:
            continue
        x1, y1, x2, y2 = bar['bbox']
        width, thickness = x2 - x1, y2 - y1
        if width < 3 * thickness:
            continue
        body_height = max((s.get('_height', s['bbox'][3] - s['bbox'][1])
                           for s in remaining if s['text'] not in OPERATORS | {'√'}), default=20)
        reach = max(width * 1.5, 6 * body_height)
        covered = [s for s in remaining if s is not bar and
                   x1 <= center(s['bbox'])[0] <= x2 and
                   abs(center(s['bbox'])[1] - center(bar['bbox'])[1]) < reach]
        above = [s for s in covered if s['bbox'][3] <= y1 + thickness * .5]
        below = [s for s in covered if s['bbox'][1] >= y2 - thickness * .5]
        if not above or not below:
            continue
        numerator = parse_region(above, depth + 1, allow_equality=False)
        denominator = parse_region(below, depth + 1, allow_equality=False)
        node = _container('fraction', [numerator, _leaf(bar), denominator])
        selected = {s['detection_index'] for s in above + below + [bar]}
        remaining = [s for s in remaining if s['detection_index'] not in selected]
        remaining.append(atom(node, bar['detection_index'], center(bar['bbox'])[1]))
    return remaining


def root_groups(items, depth):
    """The detected radical box must include its overbar and radicand span."""
    remaining = list(items)
    roots = sorted((s for s in items if s['text'] == '√'),
                   key=lambda s: s['bbox'][2] - s['bbox'][0], reverse=True)
    for root in roots:
        if root not in remaining:
            continue
        x1, y1, x2, y2 = root['bbox']
        height = y2 - y1
        sign_width = min(.18 * height, 20)
        bar_margin = min(.05 * height, 5)
        inside = [s for s in remaining if s is not root and
                  s['bbox'][0] >= x1 + sign_width and s['bbox'][2] <= x2 + .1 * height and
                  s['bbox'][1] >= y1 + bar_margin and s['bbox'][3] <= y2 + .15 * height]
        if not inside:
            raise LayoutError('Square root needs a detected sign/overbar covering its radicand')
        radicand = parse_region(inside, depth + 1, allow_equality=False)
        node = _container('sqrt', [_leaf(root), radicand])
        selected = {s['detection_index'] for s in inside + [root]}
        remaining = [s for s in remaining if s['detection_index'] not in selected]
        remaining.append(atom(node, root['detection_index'], radicand['baseline_center']))
    return remaining


def assign_scripts(items):
    """Separate small raised/lowered glyphs and require a nearby clear base."""
    operands = [s for s in items if s['text'] not in OPERATORS]
    if not operands:
        raise LayoutError('An expression needs an operand')
    heights = {s['detection_index']: s.get('_height', s['bbox'][3] - s['bbox'][1]) for s in operands}
    height = max(heights.values())
    bodies = [s for s in operands if heights[s['detection_index']] >= .9 * height]
    baseline = statistics.median(s.get('_anchor', center(s['bbox'])[1]) for s in bodies)
    bottom = baseline + height / 2
    main, small = [], []
    for item in items:
        box = item['bbox']
        if '_ast' not in item and item['text'] not in OPERATORS and box[3] - box[1] <= .9 * height:
            if box[3] < bottom - .2 * height and center(box)[1] < baseline - .2 * height:
                small.append((item, 'superscript'))
                continue
            if box[1] > baseline + .15 * height and box[3] > bottom + .12 * height:
                small.append((item, 'subscript'))
                continue
        main.append(item)
    for item in main:
        if abs(item.get('_anchor', center(item['bbox'])[1]) - baseline) > .6 * height:
            raise LayoutError('Unresolved rows or ambiguous vertical grouping')
    scripts = {}
    for item, role in small:
        if role == 'superscript' and not item['text'].isdigit():
            raise LayoutError('Only integer superscripts are supported')
        candidates = [base for base in main if base['text'] not in OPERATORS and
                      item['bbox'][0] >= base['bbox'][0] + .5 * (base['bbox'][2] - base['bbox'][0]) and
                      -.15 * height <= item['bbox'][0] - base['bbox'][2] <= 1.1 * height]
        candidates.sort(key=lambda s: abs(item['bbox'][0] - s['bbox'][2]))
        if not candidates:
            raise LayoutError(f'{role.capitalize()} has no nearby base')
        if len(candidates) > 1 and abs(item['bbox'][0] - candidates[1]['bbox'][2]) - abs(item['bbox'][0] - candidates[0]['bbox'][2]) < .1 * height:
            raise LayoutError(f'Ambiguous {role} base')
        group = scripts.setdefault(candidates[0]['detection_index'], {})
        group.setdefault(role, []).append(item)
    for group in scripts.values():
        for role, letters in group.items():
            letters.sort(key=lambda s: s['bbox'][0])
            if len(letters) > (2 if role == 'superscript' else 3):
                raise LayoutError(f'{role.capitalize()} is too long')
    return sorted(main, key=lambda s: s['bbox'][0]), scripts, height, baseline


class StructuredParser(_Parser):
    """Reuse the established operator grammar; add compound atoms and scripts."""
    def __init__(self, main, scripts, height):
        super().__init__(main, {}, height)
        self.scripts = scripts

    def term(self):
        result = self.signed_factor()
        while self.peek() == '×' or self.peek() in VARIABLES or (
                self.index < len(self.main) and '_ast' in self.main[self.index]):
            if self.peek() == '×':
                op = _leaf(self.take())
                result = _container('binary', [result, op, self.signed_factor()], '×')
            else:
                result = _container('product', [result, self.factor()])
        return result

    def factor(self):
        text = self.peek()
        if text is None or text in OPERATORS:
            raise LayoutError('Missing operand')
        item = self.take()
        identity = item['detection_index']
        if '_ast' in item:
            result = item['_ast']
        elif text.isdigit():
            digits = [item]
            while self.peek() is not None and self.peek().isdigit() and identity not in self.scripts:
                following = self.main[self.index]
                if following['bbox'][0] - digits[-1]['bbox'][2] > .45 * self.height:
                    raise LayoutError('Separated numbers need an operator')
                digits.append(self.take())
                identity = digits[-1]['detection_index']
            result = _number(digits)
        else:
            result = _leaf(item)
        group = self.scripts.get(identity, {})
        if set(group) == {'subscript', 'superscript'}:
            result = _container('scripts', [result, _number(group['subscript']), _number(group['superscript'])])
            result['script_roles'] = ['base', 'subscript', 'superscript']
        elif group:
            role, letters = next(iter(group.items()))
            result = _container('power' if role == 'superscript' else 'subscript', [result, _number(letters)])
        return result


def parse_region(items, depth=0, allow_equality=True):
    if depth > MAX_DEPTH:
        raise LayoutError('Structure nesting exceeds the supported depth')
    if not items:
        raise LayoutError('Missing structural operand')
    if not allow_equality and any(s['text'] == '=' for s in items):
        raise LayoutError('An equality cannot be a numerator, denominator or radicand')
    atoms = root_groups(fraction_groups(items, depth), depth)
    main, scripts, height, baseline = assign_scripts(atoms)
    result = StructuredParser(main, scripts, height).equation()
    result['baseline_center'] = baseline
    return result


def parse_structures(symbols):
    return parse_region(symbols)
