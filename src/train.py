"""Preview training settings; start RF-DETR training only with --start-training."""
import argparse
import json
import math
from pathlib import Path

from .model import PACKAGE_VERSION, require_backend
from .data import validate_export, write_json
from .profiles import profile_for


PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / 'configs/training.json'
TRAIN_FIELDS = ('epochs', 'batch_size', 'grad_accum_steps', 'lr', 'lr_encoder', 'num_workers',
                'checkpoint_interval', 'early_stopping', 'early_stopping_patience', 'use_ema',
                'aug_config', 'scale_jitter', 'multi_scale', 'augmentation_backend')


def read_config(path=CONFIG):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    profile = profile_for(config['pipeline'])
    if config['package_version'] != PACKAGE_VERSION:
        raise ValueError('Unsupported pipeline/package configuration')
    if config['variant'] not in {'nano', 'small'}:
        raise ValueError('Use nano or small for the first-stage experiment')
    if not isinstance(config['resolution'], int) or config['resolution'] < 128 or config['resolution'] % 32:
        raise ValueError('Resolution must be a multiple of 32')
    if config['aug_config'] != {} or config['scale_jitter'] or config['multi_scale']:
        raise ValueError('First-stage text training requires no flips, crops, or scale jitter')
    if config['augmentation_backend'] != 'torchvision':
        raise ValueError('Pin the torchvision preprocessing backend')
    for key in ('epochs', 'batch_size', 'grad_accum_steps', 'checkpoint_interval', 'early_stopping_patience'):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if config['num_workers'] < 0 or not 1 <= config['max_symbols'] <= profile['max_symbols']:
        raise ValueError('Invalid worker count or first-stage symbol limit')
    if not 0 < config['inference_threshold'] < 1:
        raise ValueError('Inference threshold must lie between zero and one')
    for key in ('lr', 'lr_encoder'):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError('Learning rates must be finite and positive')
    return config


def training_plan(config, data=None, output=None):
    profile = profile_for(config['pipeline'])
    data = Path(data or PROJECT / config['dataset_dir']).resolve()
    output = Path(output or PROJECT / config['output_dir']).resolve()
    if output == data or data in output.parents or output in data.parents:
        raise ValueError('Training output and dataset must be separate directories')
    return {'pipeline': config['pipeline'], 'package_version': PACKAGE_VERSION, 'classes': list(profile['classes']),
            'variant': config['variant'], 'resolution': config['resolution'], 'scope': config['scope'],
            'dataset_dir': str(data), 'output_dir': str(output),
            'training': {key: config[key] for key in TRAIN_FIELDS},
            'best_checkpoint': 'checkpoint_best_total.pth', 'resume_checkpoint': 'last.ckpt',
            'training_started': False, 'model_constructed': False, 'weights_downloaded': False}


def start_training(config, plan, device='cuda'):
    dataset_report = validate_export(plan['dataset_dir'])
    data_metadata = json.loads((Path(plan['dataset_dir']) / 'symbol_dataset.json').read_text(encoding='utf-8'))
    if data_metadata['pipeline'] != plan['pipeline'] or dataset_report['classes'] != plan['classes']:
        raise ValueError('Dataset vocabulary/profile differs from the training plan')
    if config['resolution'] != data_metadata['resolution']:
        raise ValueError('Prepared image resolution differs from training configuration')
    output = Path(plan['output_dir'])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty output directory; existing trained checkpoints are protected')
    require_backend()
    import torch
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable; no model constructed')
    from rfdetr import RFDETRNano, RFDETRSmall
    print('START TRAINING: RF-DETR will now construct the model and may download its official pretrained weights.', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {'pipeline': plan['pipeline'], 'classes': plan['classes'], 'resolution': config['resolution'],
                'variant': config['variant'], 'package_version': PACKAGE_VERSION,
                'scope': config['scope'], 'dataset_validation': dataset_report,
                'inference_threshold': config['inference_threshold'], 'max_symbols': config['max_symbols'],
                'best_checkpoint': 'checkpoint_best_total.pth'}
    write_json(output / 'symbol_pipeline.json', metadata)
    write_json(output / 'requested_training_plan.json', plan)
    # This is the only construction/download point, reached only by explicit start.
    cls = RFDETRNano if config['variant'] == 'nano' else RFDETRSmall
    model = cls(resolution=config['resolution'], device=device)
    active_plan = {**plan, 'training_started': True, 'model_constructed': True,
                   'weights_downloaded': 'managed by RF-DETR; official weights may already be cached'}
    write_json(output / 'requested_training_plan.json', active_plan)
    options = dict(plan['training'])
    options.update({'dataset_dir': plan['dataset_dir'], 'output_dir': str(output),
                    'class_names': plan['classes'], 'device': device,
                    'amp_dtype': 'fp16' if device.startswith('cuda') else None})
    model.train(**options)
    best = output / 'checkpoint_best_total.pth'
    if not best.is_file():
        raise RuntimeError('Training returned without its selected best checkpoint')
    write_json(output / 'run_summary_symbols.json', {'pipeline': plan['pipeline'], 'training_finished': True,
               'best_checkpoint': str(best), 'scope': config['scope'],
               'note': 'Detection mAP selects this checkpoint. Run image-only equation evaluation before claiming OCR accuracy.'})
    return best


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(CONFIG))
    parser.add_argument('--data')
    parser.add_argument('--output')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--start-training', action='store_true', help='Actually construct/download the model and train')
    args = parser.parse_args(argv)
    config = read_config(args.config)
    if args.epochs is not None:
        if args.epochs < 1:
            parser.error('epochs must be positive')
        config['epochs'] = args.epochs
    plan = training_plan(config, args.data, args.output)
    print(json.dumps(plan, ensure_ascii=True, indent=2), flush=True)
    if not args.start_training:
        print('PREVIEW ONLY: no model, installation, download, training, or GPU work.', flush=True)
        return plan
    return start_training(config, plan, args.device)


if __name__ == '__main__':
    main()
