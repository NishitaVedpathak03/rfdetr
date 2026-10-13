# RF-DETR Equation OCR

RF-DETR Nano detects individual symbols. Spatial rules reconstruct the equation
hierarchy and LaTeX. This repository contains the current pipeline, its trained
best checkpoint, a simple Streamlit UI and the 2,000-image symbol dataset.

## Setup and UI

Install Git LFS before cloning, then run:

```powershell
git clone https://github.com/NishitaVedpathak03/rfdetr.git
cd rfdetr
git lfs pull
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Upload a cropped equation image to see the original, colored symbols, hierarchy
steps and LaTeX. The UI uses CPU inference and the included best checkpoint.

## Terminal prediction

```powershell
python -m src.predict_rfdetr --image path/to/equation.png --device cpu
```

Outputs are written to `outputs/rfdetr_prediction/`, including
`predicted_latex.txt`, symbol detections, hierarchy, steps and annotated images.

## Model and supported equations

`checkpoints/rfdetr_small/checkpoint_best_total.pth` is the selected EMA model,
tracked with Git LFS. Keep `symbol_pipeline.json` alongside it. The adapter uses
RF-DETR 1.11.2 and 512-pixel aspect-preserving preprocessing.

Supported: one line, integers, variables `x`, `y`, `z`, `a`, `b`, addition,
subtraction, multiplication, equality and integer superscripts; at most 24
symbols after duplicate removal. Examples: `2x - 3 = -7`, `12 × 3 = 36`,
`x² + 2x + 1 = 0`. Fractions, roots, functions, parentheses and multiline
expressions are outside this stage. All predicted LaTeX should be checked
against the image; parsing validity does not prove recognition accuracy.

Training stopped after 28 epochs. The selected checkpoint is from epoch 20
(zero-based epoch 19), with validation detection mAP 0.9807. On 196 held-out
synthetic equations, symbol precision/recall and exact complete-equation
accuracy were 1.0. Those results do not establish accuracy on all real images.

## Included dataset

`dataset_symbols_small/` contains COCO annotations, PNG images and evaluation
references: 1,620 training images, 184 validation images and 196 test images.
There are 19 classes: digits 0–9, x/y/z/a/b, +, -, = and ×. Calibration glyphs
are restricted to training. Repeated equations/font variants and identical
pixels do not cross splits. References are used for evaluation only and are
never substituted for a prediction.

Validate the included data and evaluate the checkpoint:

```powershell
python prepare_symbol_dataset.py --out dataset_symbols_small --validate-only
python -m src.evaluate_rfdetr_symbols --data dataset_symbols_small
python -m unittest discover -s tests -p "test_*.py"
```

## Training

Preview settings without constructing a model or starting training:

```powershell
python -m src.train_rfdetr
```

For an explicitly requested new CUDA run, choose an empty output directory to
preserve the included checkpoint:

```powershell
python -m src.train_rfdetr --start-training --device cuda --output checkpoints/new_run
```

To generate additional synthetic data, choose a new directory:

```powershell
python generate_small_equations.py --out dataset_new --count 2000
```

`prepare_rfdetr_training.py` previews a Kaggle package by default. Its
`--prepare --account USERNAME` option packages source/data locally and does not
upload or start training. The Kaggle wrapper starts training only when submitted
as a GPU job. See the configuration in `configs/rfdetr_small_equations.json`.
