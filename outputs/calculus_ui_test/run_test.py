"""Ten predetermined held-out uploads through the real Streamlit inference path."""
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding='utf-8')
OUT = Path(__file__).resolve().parent
DATA = ROOT / 'dataset_calculus'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')


def main():
    refs = json.loads((DATA / 'test/references.json').read_text(encoding='utf-8'))
    coco = json.loads((DATA / 'test/_annotations.coco.json').read_text(encoding='utf-8'))
    entries = {entry['id']: entry for entry in coco['images']}
    train_groups = {r['group'] for r in json.loads((DATA / 'train/references.json').read_text(encoding='utf-8'))}
    families = ['trig_mixed', 'trig_fraction', 'fraction_root_replay', 'integral_trig',
                'definite_integral', 'derivative_operator', 'second_derivative',
                'partial_derivative', 'summation_expression', 'infinite_sum']
    selected = []
    for family in families:
        candidates = [r for r in refs if r['source'].startswith('calculus_' + family + '_')
                      and not r.get('calibration')]
        candidates.sort(key=lambda r: ('computer_modern_math_italic' not in r['source'], r['image_id']))
        ref = candidates[0]
        assert ref['group'] not in train_groups
        selected.append({'family': family, 'file': entries[ref['image_id']]['file_name'], **ref})
    write(OUT / 'selection.json', {'selection_rule': 'First test image per listed family, preferring Computer Modern; chosen before inference.',
                                 'split': 'test', 'images': selected})
    print('Selected 10 held-out formula groups before inference. Loading UI/model...', flush=True)

    from PIL import Image
    from streamlit.testing.v1 import AppTest
    from ui.inference import analyze_image
    from ui.images import render_overlay
    from src.evaluate import match_symbols, normalized_latex
    from src.profiles import profile_for

    classes = profile_for('rf-detr-calculus-equations-v4')['classes']
    truth_by_id = {}
    for ann in coco['annotations']:
        x, y, w, h = ann['bbox']
        truth_by_id.setdefault(ann['image_id'], []).append(
            {'text': classes[ann['category_id']], 'bbox': [x, y, x + w, y + h]})
    current_upload = [None]
    records, totals = [], Counter()
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=300)
    with patch('streamlit.file_uploader', side_effect=lambda *a, **kw: current_upload[0]), \
         patch('ui.inference.analyze_image', wraps=analyze_image) as inference:
        for index, ref in enumerate(selected, 1):
            image_path = DATA / 'test' / ref['file']
            upload = io.BytesIO(image_path.read_bytes())
            upload.name = image_path.name
            current_upload[0] = upload
            started = time.perf_counter()
            before = inference.call_count
            app.run()
            record = {'index': index, 'family': ref['family'], 'file': ref['file'],
                      'reference_latex': ref['latex'], 'ui_exceptions': [str(e.message) for e in app.exception],
                      'ui_errors': [str(e.value) for e in app.error]}
            if record['ui_exceptions'] or record['ui_errors']:
                records.append(record)
                print(json.dumps(record), flush=True)
                continue
            sample = app.session_state['prediction_sample']
            result = sample['prediction']
            assert app.selectbox(key='recognition_model').value == 'Calculus equations'
            assert result['metadata']['pipeline'] == 'rf-detr-calculus-equations-v4'
            assert len(app.get('imgs')) == 2
            assert inference.call_count == before + 1
            assert result['metadata']['threshold'] == .5
            target = OUT / f'{index:02d}_{ref["family"]}'
            target.mkdir(exist_ok=True)
            sample['original'].save(target / 'original.png')
            sample['colored'].save(target / 'colored.png')
            active = sample['steps'][0]['node_id'] if sample['steps'] else None
            render_overlay(sample['original'] if sample.get('rejection_reason') else sample['colored'],
                           sample['nodes'], {}, show_boxes=False, active_id=active,
                           display_size=(520, 1000)).save(target / 'ui_overlay.png')
            write(target / 'prediction.json', result)
            write(target / 'steps.json', sample['raw_steps'])
            (target / 'predicted_latex.txt').write_text(result['latex'] + '\n', encoding='utf-8')
            truth = truth_by_id[ref['image_id']]
            matches = match_symbols(result['raw_symbols'], truth)
            exact = bool(not sample.get('rejection_reason') and
                         normalized_latex(result['latex']) == normalized_latex(ref['latex']))
            navigation_checked = False
            if len(sample['steps']) > 1 and not sample.get('rejection_reason'):
                count = inference.call_count
                app.button(key='next').click().run()
                assert not app.exception and app.session_state['step_index'] == 1
                assert inference.call_count == count
                app.button(key='reset').click().run()
                assert not app.exception and app.session_state['step_index'] == 0
                assert inference.call_count == count
                navigation_checked = True
            totals.update(images=1, correct=int(exact), tp=len(matches),
                          predicted=len(result['raw_symbols']), actual=len(truth),
                          valid=int(result['metadata']['layout_valid']))
            record.update(predicted_latex=result['latex'], exact_equation_correct=exact,
                          layout_valid=result['metadata']['layout_valid'],
                          rejection_reason=sample.get('rejection_reason'),
                          warnings=result['latex_metadata']['warnings'],
                          symbol_tp=len(matches), predicted_symbols=len(result['raw_symbols']),
                          actual_symbols=len(truth), steps=len(sample['steps']),
                          next_reset_cache_verified=navigation_checked,
                          elapsed_seconds=round(time.perf_counter() - started, 2))
            records.append(record)
            write(OUT / 'image_results.json', records)
            print(json.dumps(record, ensure_ascii=True), flush=True)
        precision = totals['tp'] / max(1, totals['predicted'])
        recall = totals['tp'] / max(1, totals['actual'])
        summary = {'scope': '10 predetermined synthetic held-out test uploads through Streamlit AppTest; actual CPU neural inference',
                   'checkpoint': str(ROOT / 'checkpoints/rfdetr_calculus/checkpoint_best_total.pth'),
                   'threshold': .5, 'requested_images': 10, 'successful_ui_uploads': totals['images'],
                   'exact_equations': totals['correct'], 'layout_valid': totals['valid'],
                   'symbol_precision_iou_50': precision, 'symbol_recall_iou_50': recall,
                   'symbol_f1_iou_50': 2 * precision * recall / max(1e-10, precision + recall),
                   'inference_calls': inference.call_count,
                   'training_started': False,
                   'note': 'Synthetic test results do not establish accuracy on arbitrary screenshots or handwriting.'}
        write(OUT / 'metrics.json', summary)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
