"""Optional spatial grouping for fractions, radical bars and simple scripts.

The original parser remains the default. This module uses detected boxes only;
it never reads reference LaTeX or fills in missing symbols.
"""
import statistics

from .layout import LayoutError, _Parser, _container, _leaf, _number

OPERATORS = set('+-=×()')
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


def fraction_groups(items, depth, functions=False):
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
        numerator = parse_region(above, depth + 1, allow_equality=False, functions=functions)
        denominator = parse_region(below, depth + 1, allow_equality=False, functions=functions)
        node = _container('fraction', [numerator, _leaf(bar), denominator])
        selected = {s['detection_index'] for s in above + below + [bar]}
        remaining = [s for s in remaining if s['detection_index'] not in selected]
        remaining.append(atom(node, bar['detection_index'], center(bar['bbox'])[1]))
    return remaining


def root_groups(items, depth, functions=False):
    """The detected radical box must include its overbar and radicand span."""
    remaining = list(items)
    roots = sorted((s for s in items if s['text'] == '√'),
                   key=lambda s: s['bbox'][2] - s['bbox'][0], reverse=True)
    for root in roots:
        if root not in remaining:
            continue
        x1, y1, x2, y2 = root['bbox']
        height = y2 - y1
        span = min(height, x2 - x1)
        sign_width = .04 * span if functions else min(.18 * height, 20)
        bar_margin = .01 * span if functions else min(.05 * height, 5)
        right_margin = .03 * span if functions else .1 * height
        inside = [s for s in remaining if s is not root and
                  s['bbox'][0] >= x1 + sign_width and s['bbox'][2] <= x2 + right_margin and
                  s['bbox'][1] >= y1 + bar_margin and s['bbox'][3] <= y2 + .15 * height]
        if not inside:
            raise LayoutError('Square root needs a detected sign/overbar covering its radicand')
        radicand = parse_region(inside, depth + 1, allow_equality=False, functions=functions)
        node = _container('sqrt', [_leaf(root), radicand])
        selected = {s['detection_index'] for s in inside + [root]}
        remaining = [s for s in remaining if s['detection_index'] not in selected]
        remaining.append(atom(node, root['detection_index'], radicand['baseline_center']))
    return remaining


def parenthesis_groups(items, depth):
    remaining = list(items)
    while any(s['text'] in '()' for s in remaining):
        marks = sorted((s for s in remaining if s['text'] in '()'), key=lambda s: s['bbox'][0])
        if marks[0]['text'] != '(':
            raise LayoutError('Unbalanced parentheses')
        opening = marks[0]
        height = opening['bbox'][3] - opening['bbox'][1]
        row_marks = [s for s in marks if
                     .65 * height <= s['bbox'][3] - s['bbox'][1] <= 1.5 * height and
                     abs(center(s['bbox'])[1] - center(opening['bbox'])[1]) <= .25 * height]
        balance, closing = 0, None
        for mark in row_marks:
            balance += 1 if mark['text'] == '(' else -1
            if balance == 0:
                closing = mark
                break
        if closing is None:
            raise LayoutError('Unbalanced parentheses')
        # Ordinary TeX parentheses need not stretch to cover script digits.
        # Require horizontal containment so this allowance cannot pull a wider
        # fraction bar from another row into the parenthesized argument.
        inside = [s for s in remaining if s is not opening and s is not closing and
                  s['bbox'][0] >= opening['bbox'][2] - .03 * height and
                  s['bbox'][2] <= closing['bbox'][0] + .03 * height and
                  s['bbox'][1] >= opening['bbox'][1] - .2 * height and
                  s['bbox'][3] <= opening['bbox'][3] + .2 * height]
        body = parse_region(inside, depth + 1, allow_equality=False, functions=True)
        node = _container('parentheses', [_leaf(opening), body, _leaf(closing)])
        selected = {s['detection_index'] for s in inside + [opening, closing]}
        remaining = [s for s in remaining if s['detection_index'] not in selected]
        remaining.append(atom(node, closing['detection_index'], body['baseline_center']))
    return remaining


def assign_scripts(items, functions=False):
    """Separate small raised/lowered glyphs and require a nearby clear base."""
    if functions:
        return assign_complex_scripts(items)
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
            lower_position = center(box)[1] if functions else box[1]
            if lower_position > baseline + .15 * height and box[3] > bottom + .12 * height:
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


def assign_complex_scripts(items):
    """Use local baseline offsets when italic bases are shorter than script digits."""
    operands = [s for s in items if s['text'] not in OPERATORS]
    proposed = {}
    for item in operands:
        if '_ast' in item:
            continue
        box = item['bbox']
        candidates = []
        for base in operands:
            if base is item:
                continue
            height = base.get('_height', base['bbox'][3] - base['bbox'][1])
            baseline = base.get('_anchor', center(base['bbox'])[1])
            bottom = baseline + height / 2
            gap = box[0] - base['bbox'][2]
            if not (box[0] >= base['bbox'][0] + .5 * (base['bbox'][2] - base['bbox'][0]) and
                    -.15 * height <= gap <= 1.5 * height and box[3] - box[1] <= 1.8 * height):
                continue
            role = None
            if box[3] < bottom - .2 * height and center(box)[1] < baseline - .2 * height:
                role = 'superscript'
            elif box[3] > bottom + .15 * height and center(box)[1] > baseline + .25 * height:
                role = 'subscript'
            if role:
                candidates.append((abs(gap), base, role))
        if candidates:
            candidates.sort(key=lambda c: c[0])
            proposed[item['detection_index']] = (item, candidates)
    scripts = {}
    for identity, (item, candidates) in proposed.items():
        candidates = [c for c in candidates if c[1]['detection_index'] not in proposed]
        if not candidates:
            raise LayoutError('Ambiguous chained script assignment')
        _, base, role = candidates[0]
        if role == 'superscript' and not item['text'].isdigit():
            raise LayoutError('Only integer superscripts are supported')
        scripts.setdefault(base['detection_index'], {}).setdefault(role, []).append(item)
    main = [s for s in items if s['detection_index'] not in proposed]
    bodies = [s for s in main if s['text'] not in OPERATORS]
    if not bodies:
        raise LayoutError('An expression needs an operand')
    height = max(s.get('_height', s['bbox'][3] - s['bbox'][1]) for s in bodies)
    baseline = statistics.median(s.get('_anchor', center(s['bbox'])[1]) for s in bodies
                                 if s.get('_height', s['bbox'][3] - s['bbox'][1]) >= .9 * height)
    for item in main:
        if abs(item.get('_anchor', center(item['bbox'])[1]) - baseline) > .6 * height:
            raise LayoutError('Unresolved rows or ambiguous vertical grouping')
    for group in scripts.values():
        for role, letters in group.items():
            letters.sort(key=lambda s: s['bbox'][0])
            if len(letters) > (2 if role == 'superscript' else 3):
                raise LayoutError(f'{role.capitalize()} is too long')
    return sorted(main, key=lambda s: s['bbox'][0]), scripts, height, baseline


class StructuredParser(_Parser):
    """Reuse the established operator grammar; add compound atoms and scripts."""
    def __init__(self, main, scripts, height, functions=False):
        super().__init__(main, {}, height)
        self.scripts = scripts
        self.functions = functions

    def term(self):
        result = self.signed_factor()
        while self.peek() == '×' or self.peek() in VARIABLES or (self.functions and self.peek() in {'s', 'c', 'l'}) or (
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
        if self.functions and text in {'s', 'c', 'l'}:
            name = {'s': 'sin', 'c': 'cos', 'l': 'log'}[text]
            letters = [item]
            for letter in name[1:]:
                if self.peek() != letter:
                    raise LayoutError('Incomplete or unsupported function name')
                following = self.take()
                if following['bbox'][0] - letters[-1]['bbox'][2] > .6 * self.height:
                    raise LayoutError('Function letters are too far apart')
                letters.append(following)
            label = _container('function_name', [_leaf(s) for s in letters], name)
            group = self.scripts.get(letters[-1]['detection_index'], {})
            if group and (name != 'log' or set(group) != {'subscript'} or
                          not all(s['text'].isdigit() for s in group['subscript'])):
                raise LayoutError('Only an integer logarithm base is supported on a function name')
            if self.index >= len(self.main) or self.main[self.index].get('_ast', {}).get('kind') != 'parentheses':
                raise LayoutError('Functions require a parenthesized argument')
            argument = self.take()['_ast']
            if group:
                return _container('logarithm', [label, _number(group['subscript']), argument])
            return _container('function', [label, argument])
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


def parse_region(items, depth=0, allow_equality=True, functions=False):
    if depth > (8 if functions else MAX_DEPTH):
        raise LayoutError('Structure nesting exceeds the supported depth')
    if not items:
        raise LayoutError('Missing structural operand')
    if not allow_equality and any(s['text'] == '=' for s in items):
        raise LayoutError('An equality cannot be a numerator, denominator or radicand')
    grouped = parenthesis_groups(items, depth) if functions else items
    atoms = root_groups(fraction_groups(grouped, depth, functions), depth, functions)
    main, scripts, height, baseline = assign_scripts(atoms, functions)
    result = StructuredParser(main, scripts, height, functions).equation()
    result['baseline_center'] = baseline
    return result


def parse_structures(symbols, functions=False):
    return parse_region(symbols, functions=functions)
