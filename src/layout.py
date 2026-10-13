"""Spatial grammar for small equations, using predicted symbols only.

Supports one baseline, signed integers, implicit coefficients, +, -, multiplication,
one equality, and integer superscripts. It never reads images or dataset labels.
"""
import math
import statistics

from .data import canonical_symbol


class LayoutError(ValueError):
    pass


def _iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    aa, bb = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(1e-9, aa + bb - inter)


def clean_symbols(symbols, max_symbols=24):
    warnings, candidates = [], []
    for index, raw in enumerate(symbols):
        text = canonical_symbol(raw['text'])
        box = list(map(float, raw['bbox']))
        confidence = float(raw.get('confidence', 1.))
        if len(box) != 4 or not all(math.isfinite(v) for v in [*box, confidence]) or not (box[0] < box[2] and box[1] < box[3] and 0 <= confidence <= 1):
            raise LayoutError('Invalid symbol coordinates/confidence')
        candidates.append({'text': text, 'bbox': box, 'confidence': confidence, 'detection_index': index})
    candidates.sort(key=lambda n: (-n['confidence'], n['detection_index']))
    kept = []
    for candidate in candidates:
        overlap = next((n for n in kept if _iou(n['bbox'], candidate['bbox']) >= .6), None)
        if overlap:
            if overlap['text'] != candidate['text'] and overlap['confidence'] - candidate['confidence'] < .15:
                warnings.append('Overlapping symbol predictions disagree without a clear confidence margin.')
            continue
        kept.append(candidate)
    if len(kept) > max_symbols:
        raise LayoutError(f'More than {max_symbols} symbols; outside the small-equation scope')
    return sorted(kept, key=lambda n: (n['bbox'][0], n['bbox'][1])), warnings


def _envelope(children):
    return [min(c['bbox'][0] for c in children), min(c['bbox'][1] for c in children),
            max(c['bbox'][2] for c in children), max(c['bbox'][3] for c in children)]


def _leaf(symbol):
    return {'kind': 'symbol' if symbol['text'] in '+-=×' else 'leaf', 'text': symbol['text'],
            'bbox': symbol['bbox'], 'confidence': symbol['confidence'], 'children': [],
            'detection_index': symbol['detection_index']}


def _container(kind, children, text=None):
    return {'kind': kind, 'text': text, 'bbox': _envelope(children), 'children': children,
            'confidence': None, 'geometry_derived': True}


def _number(symbols):
    leaves = [_leaf(s) for s in symbols]
    return leaves[0] if len(leaves) == 1 else _container('number', leaves)


def _layout(symbols):
    operands = [s for s in symbols if s['text'] not in '+-=×']
    if not operands:
        raise LayoutError('An expression needs an operand')
    heights = [s['bbox'][3] - s['bbox'][1] for s in operands]
    height = statistics.median(heights)
    bottom = statistics.median(s['bbox'][3] for s in operands)
    center = bottom - height / 2
    raised, main = [], []
    for symbol in symbols:
        b = symbol['bbox']
        if symbol['text'] not in '+-=×' and b[3] < bottom - .22 * height and (b[1] + b[3]) / 2 < center - .2 * height:
            raised.append(symbol)
        else:
            main.append(symbol)
    # Use the inferred baseline to reject a second row instead of flattening it.
    bodies = [s for s in main if s['text'] not in '+-=×']
    if not bodies:
        raise LayoutError('No baseline operands')
    body_height = statistics.median(s['bbox'][3] - s['bbox'][1] for s in bodies)
    body_bottom = statistics.median(s['bbox'][3] for s in bodies)
    body_center = statistics.median((s['bbox'][1] + s['bbox'][3]) / 2 for s in bodies)
    for s in main:
        b = s['bbox']
        if abs((b[1] + b[3]) / 2 - body_center) > .6 * body_height:
            raise LayoutError('Multiple rows or an unsupported vertical layout')
        if s['text'] == '-' and b[2] - b[0] > 2.5 * body_height:
            raise LayoutError('A wide bar may be a fraction; fractions are outside this stage')
    powers = {}
    for exponent in raised:
        if not exponent['text'].isdigit():
            raise LayoutError('Only integer superscripts are supported')
        b = exponent['bbox']
        possible = [s for s in main if s['text'] not in '+-=×' and
                    b[0] >= s['bbox'][0] + .5 * (s['bbox'][2] - s['bbox'][0]) and
                    -.15 * body_height <= b[0] - s['bbox'][2] <= 1.1 * body_height and
                    s['bbox'][1] - b[1] <= 1.1 * body_height and
                    b[3] <= body_bottom - .12 * body_height]
        if not possible:
            raise LayoutError('A raised symbol has no nearby unambiguous base')
        possible.sort(key=lambda s: abs(b[0] - s['bbox'][2]))
        if len(possible) > 1 and abs(b[0] - possible[1]['bbox'][2]) - abs(b[0] - possible[0]['bbox'][2]) < .1 * body_height:
            raise LayoutError('Ambiguous superscript base')
        base = possible[0]['detection_index']
        powers.setdefault(base, []).append(exponent)
    for exponent in powers.values():
        exponent.sort(key=lambda s: s['bbox'][0])
        if len(exponent) > 2:
            raise LayoutError('Superscript has too many digits for this stage')
    return main, powers, body_height


class _Parser:
    def __init__(self, main, powers, height):
        self.main, self.powers, self.height, self.index = main, powers, height, 0

    def peek(self):
        return self.main[self.index]['text'] if self.index < len(self.main) else None

    def take(self):
        token = self.main[self.index]
        self.index += 1
        return token

    def expression(self):
        result = self.term()
        while self.peek() in {'+', '-'}:
            op = _leaf(self.take())
            rhs = self.term()
            result = _container('binary', [result, op, rhs], op['text'])
        return result

    def term(self):
        result = self.signed_factor()
        while self.peek() == '×' or self.peek() in set('xyzab'):
            if self.peek() == '×':
                op = _leaf(self.take())
                rhs = self.signed_factor()
                result = _container('binary', [result, op, rhs], '×')
            else:
                result = _container('product', [result, self.factor()])
        return result

    def signed_factor(self):
        if self.peek() in {'+', '-'}:
            op = _leaf(self.take())
            if self.peek() in {'+', '-'}:
                raise LayoutError('Repeated signs are ambiguous')
            return _container('unary', [op, self.factor()], op['text'])
        return self.factor()

    def factor(self):
        text = self.peek()
        if text is None or text in '+-=×':
            raise LayoutError('Missing operand')
        symbol = self.take()
        digits = [symbol]
        if text.isdigit():
            while self.peek() is not None and self.peek().isdigit() and digits[-1]['detection_index'] not in self.powers:
                following = self.main[self.index]
                if following['bbox'][0] - digits[-1]['bbox'][2] > .45 * self.height:
                    raise LayoutError('Separated numbers need an operator')
                digits.append(self.take())
            result = _number(digits)
        else:
            result = _leaf(symbol)
        base_id = digits[-1]['detection_index']
        if base_id in self.powers:
            result = _container('power', [result, _number(self.powers[base_id])])
        return result

    def equation(self):
        left = self.expression()
        if self.peek() == '=':
            op = _leaf(self.take())
            result = _container('binary', [left, op, self.expression()], '=')
        else:
            result = left
        if self.peek() is not None:
            raise LayoutError('Unconsumed symbols, multiple equalities or unsupported adjacency')
        return result


def _latex(ast, parent_precedence=0):
    kind, children = ast['kind'], ast['children']
    if not children:
        return r'\times' if ast['text'] == '×' else ast['text']
    if kind == 'number':
        return ''.join(_latex(c) for c in children)
    if kind == 'power':
        return _latex(children[0], 4) + '^{' + _latex(children[1]) + '}'
    if kind == 'unary':
        return ast['text'] + _latex(children[1], 3)
    if kind == 'product':
        return ' '.join(_latex(c, 3) for c in children)
    if kind == 'binary':
        precedence = {'=': 1, '+': 2, '-': 2, '×': 3}[ast['text']]
        value = _latex(children[0], precedence) + ' ' + _latex(children[1]) + ' ' + _latex(children[2], precedence + (ast['text'] == '-'))
        return r'\left(' + value + r'\right)' if precedence < parent_precedence else value
    raise LayoutError('Unknown geometry-derived container')


def _nodes(ast):
    nodes = []

    def visit(item, parent=None, role='group'):
        node_id = len(nodes)
        node = {k: v for k, v in item.items() if k != 'children'}
        node.update({'id': node_id, 'parent_id': parent, 'role': role, 'children': [],
                     'terminal': not item['children'], 'relationship_source': 'spatial grammar',
                     'relationship_confidence': None})
        nodes.append(node)
        for index, child in enumerate(item['children']):
            child_role = 'operator' if child['kind'] == 'symbol' else 'group'
            if item['kind'] == 'power':
                child_role = 'base' if index == 0 else 'superscript'
            child_id = visit(child, node_id, child_role)
            node['children'].append(child_id)
        return node_id

    visit(ast)
    return nodes


def parse_small_equation(symbols, max_symbols=24):
    """Return a labelled spatial reconstruction or an explicit rejected result."""
    empty = {'nodes': [], 'root_id': None, 'latex': '',
             'latex_metadata': {'status': 'empty', 'verified': False, 'warnings': [],
                                'unresolved_node_ids': [], 'max_depth': 0,
                                'source': 'detected symbol identities and spatial grammar'},
             'symbols': [], 'layout_valid': False}
    if not symbols:
        return empty
    try:
        cleaned, warnings = clean_symbols(symbols, max_symbols)
        if warnings:
            raise LayoutError(' '.join(warnings))
        main, powers, height = _layout(cleaned)
        ast = _Parser(main, powers, height).equation()
        nodes = _nodes(ast)
        terminals = [n for n in nodes if n['terminal']]
        if len(terminals) != len(cleaned) or len({n['detection_index'] for n in terminals}) != len(cleaned):
            raise LayoutError('Not every detection has exactly one place in the expression')
        depths = {0: 0}
        for node in nodes[1:]:
            depths[node['id']] = depths[node['parent_id']] + 1
        return {'nodes': nodes, 'root_id': 0, 'latex': _latex(ast), 'symbols': cleaned,
                'layout_valid': True,
                'latex_metadata': {'status': 'parsed', 'verified': False, 'warnings': [],
                                   'unresolved_node_ids': [], 'max_depth': max(depths.values()),
                                   'source': 'detected symbol identities and spatial grammar'}}
    except (ValueError, KeyError, TypeError) as error:
        empty['latex_metadata'].update({'status': 'partial', 'warnings': [str(error)]})
        # Keep the raw detections for diagnosis; do not fabricate a replacement equation.
        empty['symbols'] = symbols
        return empty
