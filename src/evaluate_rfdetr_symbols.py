"""Explicit held-out symbol and complete-equation evaluation; no training."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image

from .rfdetr_symbols import load_symbol_predictor, predict_symbol_image
from .small_equation_layout import _iou
from .symbol_data import SYMBOLS, validate_export, write_json


def match_symbols(predictions, truth, threshold=.5):
    """One-to-one, text-correct IoU matching; duplicates remain false positives."""
    if not predictions or not truth:
        return []
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    overlap = np.array([[_iou(p['bbox'], g['bbox']) for g in truth] for p in predictions])
    valid = (overlap >= threshold) & np.array([[p['text'] == g['text'] for g in truth] for p in predictions])
    rows, cols = linear_sum_assignment(1 - overlap + 100 * ~valid)
    return [(int(p), int(g)) for p, g in zip(rows, cols) if valid[p, g]]


def normalized_latex(text):
    return re.sub(r'\s+', '', text)


def evaluate(data, checkpoint, out='outputs/rfdetr_evaluation', split='test', limit=None, threshold=.5, device='cpu'):
    data, out = Path(data), Path(out)
    validate_export(data)
    coco = json.loads((data / split / '_annotations.coco.json').read_text(encoding='utf-8'))
    refs = {r['image_id']: r for r in json.loads((data / split / 'references.json').read_text(encoding='utf-8'))}
    annotations = {}
    for annotation in coco['annotations']:
        x, y, w, h = annotation['bbox']
        annotations.setdefault(annotation['image_id'], []).append(
            {'text': SYMBOLS[annotation['category_id']], 'bbox': [x, y, x + w, y + h]})
    images = coco['images'][:limit] if limit else coco['images']
    predictor = load_symbol_predictor(checkpoint, device)
    totals, records = Counter(), []
    for entry in images:
        with Image.open(data / split / entry['file_name']) as im:
            result, _ = predict_symbol_image(im.convert('RGB'), predictor, threshold,
                                             image_name=entry['file_name'], prepared_image=True)
        predictions, truth = result['raw_symbols'], annotations[entry['id']]
        pairs = match_symbols(predictions, truth)
        totals.update({'images': 1, 'true_positives': len(pairs), 'predicted': len(predictions),
                       'actual': len(truth), 'layout_valid': int(result['metadata']['layout_valid'])})
        reference = refs[entry['id']].get('latex')
        correct = None
        if reference is not None:
            correct = result['metadata']['layout_valid'] and normalized_latex(result['latex']) == normalized_latex(reference)
            totals.update({'equations_with_reference': 1, 'equation_correct': int(correct)})
        records.append({'file': entry['file_name'], 'predicted_latex': result['latex'], 'reference_latex': reference,
                        'exact_equation_correct': correct, 'symbol_tp': len(pairs),
                        'predicted_symbols': len(predictions), 'actual_symbols': len(truth),
                        'warnings': result['latex_metadata']['warnings']})
    precision = totals['true_positives'] / max(1, totals['predicted'])
    recall = totals['true_positives'] / max(1, totals['actual'])
    summary = {'split': split, 'scope': 'full_split' if len(images) == len(coco['images']) else 'fixed_subset',
               'images': totals['images'], 'symbol_precision': precision, 'symbol_recall': recall,
               'symbol_f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.,
               'complete_equation_accuracy': totals['equation_correct'] / totals['equations_with_reference'] if totals['equations_with_reference'] else None,
               'equations_with_reference': totals['equations_with_reference'],
               'layout_valid_rate': totals['layout_valid'] / max(1, totals['images']),
               'note': 'Layout validity alone is not correct OCR; exact comparison uses held-out references.',
               'training_started': False}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'metrics.json', summary)
    write_json(out / 'image_results.json', records)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='dataset_symbols_small')
    parser.add_argument('--checkpoint', default='checkpoints/rfdetr_small/checkpoint_best_total.pth')
    parser.add_argument('--out', default='outputs/rfdetr_evaluation')
    parser.add_argument('--split', choices=['valid', 'test'], default='test')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--threshold', type=float, default=.5)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('limit must be positive')
    print(json.dumps(evaluate(args.data, args.checkpoint, args.out, args.split, args.limit, args.threshold, args.device), indent=2))
