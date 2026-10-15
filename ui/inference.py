"""CPU RF-DETR inference adapter; no dataset or previous detector imports."""
import torch

from src.model import load_symbol_predictor, predict_symbol_image
from src.visualize import recursive_steps
from .steps import normalize_nodes, validate_steps


def load_predictor(checkpoint_path):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    return load_symbol_predictor(checkpoint_path, 'cpu')


def analyze_image(image, predictor, name, threshold=None, max_nodes=400, extended_layout=None):
    result, colored = predict_symbol_image(image, predictor,
        threshold=predictor.metadata.get('inference_threshold', .5) if threshold is None else threshold,
        max_symbols=predictor.metadata.get('max_symbols', 24), image_name=name,
        extended_layout=extended_layout)
    nodes = normalize_nodes(result['nodes'], image.size)
    raw_steps = {'schema_version': 1, 'root_id': result['root_id'], 'steps': recursive_steps(nodes)}
    return {'name': name, 'original': image, 'colored': colored, 'nodes': nodes,
            'tree': None, 'steps': validate_steps(raw_steps, nodes), 'raw_steps': raw_steps,
            'warnings': result['latex_metadata']['warnings'],
            'source': 'model prediction', 'prediction': result}
