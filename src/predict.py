"""Run RF-DETR prediction on an image and save symbols, hierarchy and LaTeX."""
import argparse
import json
from pathlib import Path

from PIL import Image

from .model import load_symbol_predictor, predict_symbol_image
from .data import composite_rgb, write_json


def predict(image_path, checkpoint='checkpoints/rfdetr_small/checkpoint_best_total.pth',
            out='outputs/rfdetr_prediction', threshold=.5, device='cpu'):
    predictor = load_symbol_predictor(checkpoint, device)
    with Image.open(image_path) as image:
        original = composite_rgb(image)
    result, colored = predict_symbol_image(original, predictor, threshold, image_name=str(image_path))
    from .visualize import hierarchy_overlay, recursive_steps
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    colored.save(folder / 'predicted_colored.png')
    hierarchy_overlay(original, result['nodes']).save(folder / 'predicted_regions.png')
    write_json(folder / 'predicted_symbols.json', result['raw_symbols'])
    write_json(folder / 'predicted_structure.json', result)
    write_json(folder / 'predicted_steps.json', {'schema_version': 1, 'root_id': result['root_id'],
                                               'steps': recursive_steps(result['nodes'])})
    (folder / 'predicted_latex.txt').write_text(result['latex'] + '\n', encoding='utf-8')
    print(json.dumps({'layout_valid': result['metadata']['layout_valid'], 'latex': result['latex'],
                      'warnings': result['latex_metadata']['warnings'], 'output': str(folder)}, ensure_ascii=True, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--checkpoint', default='checkpoints/rfdetr_small/checkpoint_best_total.pth')
    parser.add_argument('--out', default='outputs/rfdetr_prediction')
    parser.add_argument('--threshold', type=float, default=.5)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    predict(args.image, args.checkpoint, args.out, args.threshold, args.device)
