"""Preview or explicitly package RF-DETR's isolated Kaggle job; never submit it."""
import argparse
import base64
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

from src.data import validate_export, write_json
from src.train import read_config
from src.profiles import profile_for


PROJECT = Path(__file__).resolve().parent
SOURCE_FILES = ('src/data.py', 'src/layout.py', 'src/model.py',
                'src/train.py', 'src/evaluate.py', 'src/predict.py',
                'src/__init__.py', 'src/visualize.py', 'src/validation.py',
                'src/profiles.py', 'src/structures.py', 'src/calculus.py',
                'configs/training.json', 'configs/extended.json', 'configs/complex.json',
                'configs/calculus.json', 'requirements-rfdetr.txt')

RUN_TEMPLATE = '''"""Generated RF-DETR pilot; runs only in Kaggle, only when explicitly submitted."""
import base64
import hashlib
import importlib.metadata
import io
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

SOURCE_HASH = "__HASH__"
SOURCE_B64 = "__BUNDLE__"
DATASET_OWNER = "__OWNER__"
DATASET_SLUG = "__DATASET_SLUG__"
DATASET_FOLDER = "__DATASET_FOLDER__"
CONFIG_FILE = "__CONFIG_FILE__"
OUTPUT_FOLDER = "__OUTPUT_FOLDER__"

class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log
    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
        self.flush()
        return len(text)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def __getattr__(self, name):
        return getattr(self.terminal, name)

def locate_dataset():
    root = Path('/kaggle/input')
    for mount in [root / DATASET_SLUG, root / 'datasets' / DATASET_OWNER / DATASET_SLUG]:
        for candidate in [mount / DATASET_FOLDER, mount]:
            if (candidate / 'symbol_dataset.json').is_file():
                return candidate
    matches = [p.parent for p in root.rglob('symbol_dataset.json') if DATASET_SLUG in str(p)]
    if len(matches) != 1:
        raise FileNotFoundError('Attach the prepared symbol dataset; no unique dataset mount found.')
    return matches[0]

def main():
    working = Path('/kaggle/working')
    if not working.is_dir():
        raise RuntimeError('Kaggle-only entry point. Local training uses src.train --start-training explicitly.')
    original_out, original_err = sys.stdout, sys.stderr
    with (working / 'training.log').open('w', encoding='utf-8', buffering=1) as log:
        sys.stdout, sys.stderr = Tee(original_out, log), Tee(original_err, log)
        try:
            print('RF-DETR equation training starting:', CONFIG_FILE, flush=True)
            bundle = base64.b64decode(SOURCE_B64)
            if hashlib.sha256(bundle).hexdigest() != SOURCE_HASH:
                raise ValueError('Source bundle checksum mismatch')
            with tempfile.TemporaryDirectory(prefix='rfdetr_code_') as directory:
                code = Path(directory)
                with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
                    # The generated archive has a fixed allow-list of relative source paths.
                    for name in archive.namelist():
                        dest = (code / name).resolve()
                        if code.resolve() not in dest.parents:
                            raise ValueError('Unsafe source archive path')
                    archive.extractall(code)
                try:
                    installed = importlib.metadata.version('rfdetr')
                except importlib.metadata.PackageNotFoundError:
                    installed = None
                if installed != '1.11.2':
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', str(code / 'requirements-rfdetr.txt')])
                import torch
                print('CUDA available:', torch.cuda.is_available(), flush=True)
                if not torch.cuda.is_available():
                    raise RuntimeError('GPU required for this explicitly submitted Kaggle job')
                print('GPU:', torch.cuda.get_device_name(0), flush=True)
                data = locate_dataset()
                print('Resolved symbol dataset:', data, flush=True)
                sys.path.insert(0, str(code))
                from src.train import main as train
                best = train(['--config', str(code / CONFIG_FILE), '--data', str(data),
                              '--output', str(working / OUTPUT_FOLDER), '--device', 'cuda', '--start-training'])
                from src.evaluate import evaluate
                metrics = evaluate(data, best, working / 'equation_evaluation', device='cuda')
                print('Held-out equation evaluation:', metrics, flush=True)
        finally:
            sys.stdout, sys.stderr = original_out, original_err

if __name__ == '__main__':
    main()
'''


def prepare(data, project=PROJECT, account='nishitavedpathak03', config=None):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}', account):
        raise ValueError('Invalid Kaggle username')
    project, data = Path(project).resolve(), Path(data).resolve()
    report = validate_export(data)
    config_path = Path(config) if config else project / 'configs/training.json'
    config_path = config_path.resolve()
    config_name = config_path.relative_to(project).as_posix()
    if config_name not in SOURCE_FILES:
        raise ValueError('Kaggle packaging requires a registered training config')
    config = read_config(config_path)
    profile = profile_for(config['pipeline'])
    metadata = json.loads((data / 'symbol_dataset.json').read_text(encoding='utf-8'))
    if metadata['resolution'] != config['resolution']:
        raise ValueError('Dataset and model resolutions differ')
    if metadata['pipeline'] != config['pipeline'] or report['classes'] != list(profile['classes']):
        raise ValueError('Dataset and selected training profile differ')
    name = profile['name']
    slug = 'equation-ocr-' + name + '-symbols'
    job, upload = project / f'kaggle_rfdetr_{name}', project / f'kaggle_rfdetr_{name}_upload'
    job.mkdir(parents=True, exist_ok=True)
    upload.mkdir(parents=True, exist_ok=True)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
        for source_name in SOURCE_FILES:
            archive.writestr(source_name, (project / source_name).read_bytes())
    bundle = stream.getvalue()
    digest = hashlib.sha256(bundle).hexdigest()
    run = RUN_TEMPLATE.replace('__BUNDLE__', base64.b64encode(bundle).decode()).replace('__HASH__', digest).replace('__OWNER__', account)
    run = run.replace('__DATASET_SLUG__', slug).replace('__CONFIG_FILE__', config_name)
    dataset_folder = Path(config['dataset_dir']).name
    run = run.replace('__DATASET_FOLDER__', dataset_folder)
    run = run.replace('__OUTPUT_FOLDER__', Path(config['output_dir']).name)
    compile(run, str(job / 'run.py'), 'exec')
    (job / 'run.py').write_text(run, encoding='utf-8')
    write_json(job / 'kernel-metadata.json', {'id': account + '/equation-ocr-rfdetr-' + name,
        'title': 'Equation OCR RFDETR ' + name.title(), 'code_file': 'run.py', 'language': 'python',
        'kernel_type': 'script', 'is_private': True, 'enable_gpu': True,
        'machine_shape': 'NvidiaTeslaT4', 'enable_internet': True,
        'dataset_sources': [account + '/' + slug],
        'competition_sources': [], 'kernel_sources': [], 'model_sources': []})
    write_json(upload / 'dataset-metadata.json', {'id': account + '/' + slug,
        'title': 'Equation OCR ' + name.title() + ' Symbols', 'licenses': [{'name': 'CC0-1.0'}]})
    # Never package original dataset trees or old checkpoints in this job.
    data_files = [data / 'symbol_dataset.json']
    if (data / 'preparation_report.json').is_file():
        data_files.append(data / 'preparation_report.json')
    for split in ('train', 'valid', 'test'):
        folder = data / split
        coco_file = folder / '_annotations.coco.json'
        coco = json.loads(coco_file.read_text(encoding='utf-8'))
        data_files.extend([coco_file, folder / 'references.json'])
        data_files.extend(folder / entry['file_name'] for entry in coco['images'])
    with zipfile.ZipFile(upload / 'dataset.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(set(data_files)):
            if data not in file.resolve().parents or file.is_symlink() or (hasattr(file, 'is_junction') and file.is_junction()):
                raise ValueError('Links are not permitted in the symbol dataset archive')
            archive.write(file, dataset_folder + '/' + file.relative_to(data).as_posix())
    with zipfile.ZipFile(upload / 'dataset.zip') as archive:
        if archive.testzip() is not None:
            raise ValueError('Dataset ZIP integrity check failed')
    return {'prepared': True, 'job': str(job), 'upload': str(upload), 'dataset': report,
            'source_bundle_sha256': digest, 'training_started': False, 'uploaded': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=str(PROJECT / 'dataset_symbols_small'))
    parser.add_argument('--account', default='nishitavedpathak03')
    parser.add_argument('--config', help='Select a registered config, e.g. configs/complex.json or configs/calculus.json')
    parser.add_argument('--prepare', action='store_true', help='Write local job/ZIP files; does not upload or train')
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare(args.data, account=args.account, config=args.config), indent=2, ensure_ascii=True))
    else:
        print(json.dumps({'preview_only': True, 'data': args.data, 'dataset_exists': Path(args.data).is_dir(),
                          'account': args.account, 'training_started': False, 'uploaded': False,
                          'next_step': 'Generate/validate small equations, then explicitly prepare when wanted.'}, indent=2))
