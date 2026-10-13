"""Lazy RF-DETR adapter. Imports/previews never instantiate or download a model."""
import json
import math
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .small_equation_layout import parse_small_equation
from .symbol_data import SYMBOLS, composite_rgb, letterbox_image, transform_box


PACKAGE_VERSION = '1.11.2'
PIPELINE = 'rf-detr-small-equations-v1'


def require_backend():
    try:
        installed = version('rfdetr')
    except PackageNotFoundError as error:
        raise RuntimeError('RF-DETR is not installed. Use the separate requirements-rfdetr.txt when setup is authorized.') from error
    if installed != PACKAGE_VERSION:
        raise RuntimeError(f'This adapter targets rfdetr {PACKAGE_VERSION}; installed version is {installed}.')


@dataclass
class SymbolPredictor:
    model: object
    metadata: dict
    checkpoint_path: Path
    backend: str = 'rf-detr'


def load_symbol_predictor(checkpoint_path, device='cpu'):
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'No trained RF-DETR checkpoint at {path}. No pretrained model was downloaded.')
    sidecar = path.parent / 'symbol_pipeline.json'
    if not sidecar.is_file():
        raise FileNotFoundError('Download symbol_pipeline.json together with the trained RF-DETR checkpoint.')
    metadata = json.loads(sidecar.read_text(encoding='utf-8'))
    if metadata.get('pipeline') != PIPELINE or metadata.get('classes') != list(SYMBOLS):
        raise ValueError('Checkpoint label metadata does not belong to the small-equation pipeline')
    require_backend()
    from rfdetr import RFDETR
    # Existing local weights only; from_checkpoint restores the trained model
    # variant, resolution and class names. It never substitutes COCO weights.
    model = RFDETR.from_checkpoint(str(path), device=device)
    if list(model.class_names or []) != list(SYMBOLS):
        raise ValueError('RF-DETR class names do not match symbol_pipeline.json')
    if model.model_config.resolution != metadata['resolution']:
        raise ValueError('RF-DETR resolution does not match preprocessing metadata')
    return SymbolPredictor(model, metadata, path)


def map_detections(detections, geometry, original_size, classes=SYMBOLS):
    symbols, padding_rejections = [], 0
    labels = getattr(detections, 'class_id', None)
    confidences = getattr(detections, 'confidence', None)
    if labels is None or confidences is None:
        raise ValueError('RF-DETR detections need class_id and confidence')
    if not (len(labels) == len(confidences) == len(detections.xyxy)):
        raise ValueError('Detection array lengths disagree')
    names = getattr(detections, 'data', {}).get('class_name')
    for index, (box, label, confidence) in enumerate(zip(detections.xyxy, labels, confidences)):
        cid = int(label)
        if not 0 <= cid < len(classes):
            raise ValueError('Predicted class ID is outside the symbol vocabulary')
        if names is not None and str(names[index]) != classes[cid]:
            raise ValueError('Predicted class name/ID mapping is inconsistent')
        box = [float(v) for v in box]
        if len(box) != 4 or not all(math.isfinite(v) for v in [*box, float(confidence)]):
            raise ValueError('Nonfinite detector output')
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if not (geometry['left'] <= cx <= geometry['left'] + geometry['width'] and
                geometry['top'] <= cy <= geometry['top'] + geometry['height']):
            padding_rejections += 1
            continue
        mapped = transform_box(box, geometry, inverse=True)
        mapped = [min(max(v, 0.), float(original_size[i % 2])) for i, v in enumerate(mapped)]
        if mapped[2] <= mapped[0] or mapped[3] <= mapped[1]:
            padding_rejections += 1
            continue
        symbols.append({'text': classes[cid], 'bbox': mapped, 'confidence': float(confidence)})
    return symbols, padding_rejections


def predict_symbol_image(image, predictor, threshold=.5, max_symbols=24, image_name='upload', prepared_image=False):
    if not 0 < threshold < 1 or max_symbols < 1:
        raise ValueError('Invalid inference threshold or symbol limit')
    image = composite_rgb(image)
    if prepared_image:
        side = predictor.metadata['resolution']
        if image.size != (side, side):
            raise ValueError('Prepared image dimensions disagree with the model')
        prepared = image.convert('RGB')
        geometry = {'crop': [0, 0, side, side], 'sx': 1., 'sy': 1., 'left': 0, 'top': 0,
                    'width': side, 'height': side, 'side': side}
    else:
        prepared, geometry = letterbox_image(image, predictor.metadata['resolution'])
    started = time.perf_counter()
    detections = predictor.model.predict(prepared, threshold=threshold)
    raw_symbols, padding_rejections = map_detections(detections, geometry, image.size)
    parsed = parse_small_equation(raw_symbols, max_symbols)
    elapsed = time.perf_counter() - started
    from .visualize import colored_equation
    colored, colors = colored_equation(image, parsed['nodes'])
    result = {'schema_version': 1, 'image': image_name, 'image_size': list(image.size),
              'coordinate_format': 'xyxy in original-image pixels; right/bottom exclusive',
              'nodes': parsed['nodes'], 'root_id': parsed['root_id'], 'raw_symbols': raw_symbols,
              'latex': parsed['latex'], 'latex_metadata': parsed['latex_metadata'],
              'metadata': {'backend': 'rf-detr', 'pipeline': PIPELINE,
                           'checkpoint_path': str(predictor.checkpoint_path),
                           'checkpoint_epoch': predictor.metadata.get('best_epoch'),
                           'inference_seconds': elapsed, 'threshold': threshold,
                           'max_symbols': max_symbols, 'raw_symbol_count': len(raw_symbols),
                           'node_limit_reached': len(raw_symbols) > max_symbols and not parsed['layout_valid'],
                           'layout_valid': parsed['layout_valid'], 'layout_source': 'spatial grammar; not a learned container detector',
                           'preprocessing': geometry, 'padding_rejections': padding_rejections,
                           'visualization': colors, 'confidence_note': 'uncalibrated symbol confidence; geometry has no learned confidence',
                           'limitations': ['Only small single-line equations and integer superscripts are supported.',
                                           'A parsed result does not establish correct symbol recognition.',
                                           'Fractions, roots, functions, parentheses and multiline layouts are outside this stage.']}}
    return result, colored
