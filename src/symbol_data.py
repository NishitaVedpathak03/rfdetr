"""Small-equation symbol labels, COCO export and reversible image geometry.

Importing this module never generates data, imports RF-DETR or downloads weights.
"""
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


SYMBOLS = tuple('0123456789xyzab+-=×')
TERMINALS = {'leaf', 'symbol'}
SUPPORTED_KINDS = TERMINALS | {'binary', 'power', 'product', 'number', 'unary'}


def canonical_symbol(text):
    value = str(text or '').strip().replace('−', '-').replace('*', '×')
    if value not in SYMBOLS:
        raise ValueError(f'Unsupported or combined symbol: {text!r}')
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def composite_rgb(image):
    """Preserve transparent equation PNGs by compositing them on white."""
    rgba = image.convert('RGBA')
    original = Image.new('RGBA', rgba.size, 'white')
    original.alpha_composite(rgba)
    return original.convert('RGB')


def letterbox_image(image, side=512, crop=True):
    """Crop only outer whitespace, then preserve aspect ratio on a white square."""
    if side < 32:
        raise ValueError('side must be at least 32')
    original = composite_rgb(image)
    region = (0, 0, *original.size)
    if crop:
        ink = original.convert('L').point(lambda value: 255 if value < 200 else 0)
        bounds = ink.getbbox()
        if bounds:
            region = (max(0, bounds[0] - 12), max(0, bounds[1] - 12),
                      min(original.width, bounds[2] + 12), min(original.height, bounds[3] + 12))
    tile = original.crop(region)
    scale = min(side / tile.width, side / tile.height)
    width, height = max(1, round(tile.width * scale)), max(1, round(tile.height * scale))
    left, top = (side - width) // 2, (side - height) // 2
    canvas = Image.new('RGB', (side, side), 'white')
    canvas.paste(tile.resize((width, height), Image.Resampling.LANCZOS), (left, top))
    return canvas, {'crop': list(region), 'sx': width / tile.width, 'sy': height / tile.height,
                    'left': left, 'top': top, 'width': width, 'height': height, 'side': side}


def transform_box(box, geometry, inverse=False):
    x1, y1, x2, y2 = map(float, box)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        raise ValueError('Invalid symbol box')
    dx, dy = geometry['crop'][:2]
    sx, sy, left, top = (geometry[k] for k in ('sx', 'sy', 'left', 'top'))
    if inverse:
        return [(x1 - left) / sx + dx, (y1 - top) / sy + dy,
                (x2 - left) / sx + dx, (y2 - top) / sy + dy]
    return [(x1 - dx) * sx + left, (y1 - dy) * sy + top,
            (x2 - dx) * sx + left, (y2 - dy) * sy + top]


def eligible_symbols(nodes, max_symbols=24):
    """Reject whole images containing unsupported content; never leave it unlabelled."""
    if not nodes or any(n['kind'] not in SUPPORTED_KINDS for n in nodes):
        raise ValueError('Complex structure is outside the first-stage scope')
    terminals = [n for n in nodes if n['kind'] in TERMINALS]
    if not 1 <= len(terminals) <= max_symbols:
        raise ValueError('Symbol count is outside the first-stage scope')
    result = []
    for node in terminals:
        symbol = canonical_symbol(node.get('text'))
        if node.get('role') == 'superscript' and not symbol.isdigit():
            raise ValueError('Only integer superscripts are supported initially')
        result.append({'text': symbol, 'bbox': list(node['bbox'])})
    # Reject compound bases/exponents which require a richer spatial grammar.
    lookup = {n['id']: n for n in nodes}
    for node in nodes:
        if node['kind'] == 'power':
            children = [n for n in nodes if n.get('parent_id') == node['id']]
            if any(n['kind'] not in TERMINALS | {'number'} for n in children):
                raise ValueError('Compound power operands are outside this stage')
            if node.get('parent_id') is not None and lookup[node['parent_id']]['kind'] == 'power':
                raise ValueError('Nested powers are outside this stage')
    return result


def partition_records(records, seed=42):
    """Keep duplicate equations/fonts and identical source images in one split.

    Calibration glyphs go to training; validation/test assess complete equations.
    """
    groups, calibration = defaultdict(list), []
    for record in records:
        if record.get('calibration'):
            calibration.append(record)
        else:
            groups[record['group']].append(record)
    keys = sorted(groups)
    if len(keys) < 3:
        raise ValueError('Need at least three distinct equation groups')
    random.Random(seed).shuffle(keys)
    nv = nt = max(1, int(len(keys) * .1))
    key_splits = {'valid': keys[:nv], 'test': keys[nv:nv + nt], 'train': keys[nv + nt:]}
    result = {name: [r for key in group_keys for r in groups[key]] for name, group_keys in key_splits.items()}
    result['train'].extend(calibration)
    return result


def export_records(records, out, side=512, seed=42):
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty dataset: {out}')
    splits = partition_records(records, seed)
    categories = [{'id': i, 'name': symbol, 'supercategory': 'math_symbol'} for i, symbol in enumerate(SYMBOLS)]
    report = {'pipeline': 'rf-detr-small-equations-v1', 'classes': list(SYMBOLS), 'resolution': side,
              'split_seed': seed, 'split_policy': 'equation/content groups; glyph calibration in training only',
              'splits': {}, 'training_started': False}
    out.mkdir(parents=True, exist_ok=True)
    used_hashes = {}
    for split, examples in splits.items():
        folder = out / split
        folder.mkdir()
        coco = {'info': {'description': 'Small equations; individual glyphs only'},
                'licenses': [], 'categories': categories, 'images': [], 'annotations': []}
        references, counts = [], Counter()
        for index, record in enumerate(examples):
            image = record.get('image')
            if image is None:
                with Image.open(record['path']) as im:
                    image = composite_rgb(im)
            canvas, geometry = letterbox_image(image, side)
            digest = hashlib.sha256(canvas.tobytes()).hexdigest()
            if digest in used_hashes and used_hashes[digest] != split:
                raise ValueError('Identical image content appears in different splits')
            used_hashes[digest] = split
            filename = f'{index:06d}.png'
            annotations = []
            for glyph in record['symbols']:
                text = canonical_symbol(glyph['text'])
                x1, y1, x2, y2 = transform_box(glyph['bbox'], geometry)
                if not (0 <= x1 < x2 <= side + 1e-4 and 0 <= y1 < y2 <= side + 1e-4):
                    raise ValueError('Symbol extends outside the prepared image')
                annotation = {'id': len(coco['annotations']) + len(annotations) + 1, 'image_id': index,
                              'category_id': SYMBOLS.index(text), 'bbox': [x1, y1, x2 - x1, y2 - y1],
                              'area': (x2 - x1) * (y2 - y1), 'iscrowd': 0}
                annotations.append(annotation)
                counts[text] += 1
            canvas.save(folder / filename)
            coco['images'].append({'id': index, 'file_name': filename, 'width': side, 'height': side})
            coco['annotations'].extend(annotations)
            references.append({'image_id': index, 'source': record['source'], 'group': record['group'],
                               'image_sha256': digest, 'latex': record.get('latex'),
                               'calibration': bool(record.get('calibration'))})
        write_json(folder / '_annotations.coco.json', coco)
        # References are evaluation provenance, never inference inputs.
        write_json(folder / 'references.json', references)
        report['splits'][split] = {'images': len(examples), 'symbols': len(coco['annotations']),
                                    'class_counts': dict(counts)}
    report['missing_training_classes'] = sorted(set(SYMBOLS) - set(report['splits']['train']['class_counts']))
    write_json(out / 'symbol_dataset.json', report)
    return report


def export_existing(data, out, limit=2000, side=512, seed=42):
    data = Path(data).resolve()
    if Path(out).resolve() == data or data in Path(out).resolve().parents:
        raise ValueError('Export must be outside the source dataset')
    manifest = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    records, skipped = [], Counter()
    for entry in manifest:
        name = entry['sample']
        folder = (data / name).resolve()
        if folder.parent != data:
            raise ValueError('Invalid sample path in manifest')
        nodes = json.loads((folder / 'annotations.json').read_text(encoding='utf-8'))
        try:
            glyphs = eligible_symbols(nodes)
        except ValueError as error:
            skipped[str(error)] += 1
            continue
        content_hash = hashlib.sha256((folder / 'original.png').read_bytes()).hexdigest()
        # Normalized rendered hierarchy/text groups keep font variants together.
        signature = json.dumps([(n['kind'], n.get('role'), n.get('text'), n.get('parent_id')) for n in nodes])
        group = hashlib.sha256(signature.encode()).hexdigest()
        records.append({'path': str(folder / 'original.png'), 'source': name, 'symbols': glyphs,
                        'group': group, 'calibration': len(glyphs) == 1, 'content_hash': content_hash})
        if len(records) >= limit:
            break
    report = export_records(records, out, side, seed)
    report['source_dataset'] = str(data)
    report['skipped_images'] = dict(skipped)
    write_json(Path(out) / 'symbol_dataset.json', report)
    return report


def validate_export(data):
    """Read-only validation, including disjoint images and training vocabulary."""
    data = Path(data)
    metadata = json.loads((data / 'symbol_dataset.json').read_text(encoding='utf-8'))
    if metadata['classes'] != list(SYMBOLS) or metadata['pipeline'] != 'rf-detr-small-equations-v1':
        raise ValueError('Unsupported symbol dataset schema')
    hashes, groups, counts = {}, {}, {}
    training_classes = set()
    for split in ('train', 'valid', 'test'):
        folder = data / split
        coco = json.loads((folder / '_annotations.coco.json').read_text(encoding='utf-8'))
        if [(c['id'], c['name']) for c in coco['categories']] != list(enumerate(SYMBOLS)):
            raise ValueError('Class IDs differ between splits')
        images = {i['id']: i for i in coco['images']}
        if len(images) != len(coco['images']) or not images:
            raise ValueError('Duplicate image IDs or empty split')
        ids = set()
        per_image = Counter()
        for annotation in coco['annotations']:
            image = images.get(annotation['image_id'])
            cid = annotation['category_id']
            if not image or annotation['id'] in ids or cid not in range(len(SYMBOLS)):
                raise ValueError('Invalid COCO annotation reference')
            ids.add(annotation['id'])
            x, y, w, h = annotation['bbox']
            if not all(math.isfinite(v) for v in (x, y, w, h)) or not (0 <= x < x + w <= image['width'] + 1e-4 and 0 <= y < y + h <= image['height'] + 1e-4):
                raise ValueError('Invalid COCO bounding box')
            per_image[annotation['image_id']] += 1
            if split == 'train':
                training_classes.add(cid)
        for image_id, image in images.items():
            if not 1 <= per_image[image_id] <= 24:
                raise ValueError('Image has missing or excessive glyph labels')
            path = (folder / image['file_name']).resolve()
            if path.parent != folder.resolve():
                raise ValueError('Image path escapes its split')
            with Image.open(path) as im:
                if im.size != (image['width'], image['height']):
                    raise ValueError('COCO dimensions disagree with PNG')
                digest = hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
            if digest in hashes and hashes[digest] != split:
                raise ValueError('Image content overlaps between splits')
            hashes[digest] = split
        references = json.loads((folder / 'references.json').read_text(encoding='utf-8'))
        if {r['image_id'] for r in references} != set(images):
            raise ValueError('Evaluation references disagree with images')
        for reference in references:
            group = reference['group']
            if group in groups and groups[group] != split:
                raise ValueError('Equation groups overlap between splits')
            groups[group] = split
        counts[split] = len(images)
    if training_classes != set(range(len(SYMBOLS))):
        raise ValueError('Training split is missing symbol classes')
    return {'valid': True, 'splits': counts, 'classes': list(SYMBOLS), 'training_started': False}
