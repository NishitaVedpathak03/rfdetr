"""Bounded integral/summation geometry, using detected glyphs only.

No symbolic solving, reference lookup or insertion of missing glyphs occurs here.
"""
from .layout import LayoutError, _container, _leaf
from .structures import atom, center, parse_region


def region(items, depth, equality=False):
    return parse_region(items, depth + 1, allow_equality=equality, functions=True, calculus=True)


def limits_for(sign, items):
    x1, y1, x2, y2 = sign['bbox']
    height, width = y2 - y1, x2 - x1
    left = x1 - .6 * width if sign['text'] == '∑' else x1
    right = x2 + (.6 * width if sign['text'] == '∑' else .45 * height)
    nearby = [s for s in items if s is not sign and '_ast' not in s and
              left <= center(s['bbox'])[0] <= right and
              y1 - .8 * height <= center(s['bbox'])[1] <= y2 + .8 * height]
    above = [s for s in nearby if center(s['bbox'])[1] < y1 + .18 * height]
    below = [s for s in nearby if center(s['bbox'])[1] > y2 - .18 * height]
    if bool(above) != bool(below) or (sign['text'] == '∑' and not above):
        raise LayoutError('Integral/sum limits must be a complete upper/lower pair')
    if len(above) > 5 or len(below) > 7:
        raise LayoutError('Operator limits exceed the supported scope')
    if sign['text'] == '∑':
        text = ''.join(s['text'] for s in sorted(below, key=lambda s: s['bbox'][0]))
        if len(text) < 3 or text[0] not in 'ijkn' or text[1] != '=' or not text[2:].isdigit():
            raise LayoutError('A summation needs a detected index = integer lower limit')
    if any(s['text'] in {'∫', '∑', '(', ')', '√'} for s in above + below):
        raise LayoutError('Compound operator limits are outside this profile')
    return below, above


def sum_body(after):
    if not after:
        raise LayoutError('Summation needs a body')
    if after[0]['text'] == '(':
        balance = 0
        for end, item in enumerate(after):
            if item['text'] == '(':
                balance += 1
            elif item['text'] == ')':
                balance -= 1
                if balance == 0:
                    right = item['bbox'][2]
                    return [s for s in after if s['bbox'][0] < right]
        raise LayoutError('Unbalanced summation body parentheses')
    first = after[0]
    if first['text'] not in 'xyzabijkn' and not first['text'].isdigit():
        raise LayoutError('Use a parenthesized sum body or a single scripted variable/number')
    height = first['bbox'][3] - first['bbox'][1]
    body = [first]
    for item in after[1:]:
        if (item['text'] in '+-=×()' or item['bbox'][0] - body[-1]['bbox'][2] > height or
                not item['text'].isdigit()):
            break
        body.append(item)
    return body


def operator_groups(items, depth):
    remaining = list(items)
    # Inner/rightmost operators are consumed before an enclosing operator.
    signs = sorted((s for s in items if s['text'] in {'∫', '∑'}),
                   key=lambda s: s['bbox'][0], reverse=True)
    for sign in signs:
        if sign not in remaining:
            continue
        lower, upper = limits_for(sign, remaining)
        excluded = {s['detection_index'] for s in [sign, *lower, *upper]}
        after = sorted((s for s in remaining if s['detection_index'] not in excluded and
                        s['bbox'][0] >= sign['bbox'][2] - 1), key=lambda s: s['bbox'][0])
        differential = []
        if sign['text'] == '∫':
            anchor = center(sign['bbox'])[1]
            height = sign['bbox'][3] - sign['bbox'][1]
            pairs = [(a, b) for a, b in zip(after, after[1:]) if
                     a['text'] == 'd' and b['text'] in 'xyz' and
                     abs(center(a['bbox'])[1] - anchor) < .3 * height and
                     abs(center(b['bbox'])[1] - center(a['bbox'])[1]) < .4 * (a['bbox'][3] - a['bbox'][1]) and
                     0 <= b['bbox'][0] - a['bbox'][2] <= .8 * (a['bbox'][3] - a['bbox'][1])]
            if not pairs:
                raise LayoutError('Integral needs a detected differential such as d x')
            differential = list(pairs[0])
            body = [s for s in after if s['bbox'][0] < differential[0]['bbox'][0]]
        else:
            body = sum_body(after)
        parsed_body = region(body, depth)
        children, roles = [_leaf(sign)], ['operator']
        if lower:
            children += [region(lower, depth, equality=sign['text'] == '∑'), region(upper, depth)]
            roles += ['lower_limit', 'upper_limit']
        children.append(parsed_body)
        roles.append('body')
        if differential:
            children.append(_container('differential', [_leaf(s) for s in differential]))
            roles.append('differential')
        node = _container('integral' if sign['text'] == '∫' else 'summation', children)
        node['child_roles'] = roles
        selected = excluded | {s['detection_index'] for s in body + differential}
        remaining = [s for s in remaining if s['detection_index'] not in selected]
        remaining.append(atom(node, sign['detection_index'], parsed_body['baseline_center']))
    return remaining
