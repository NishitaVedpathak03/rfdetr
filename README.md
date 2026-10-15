# RF-DETR Equation OCR

RF-DETR Nano detects individual symbols. Spatial rules reconstruct the equation
hierarchy and LaTeX. This repository contains the current pipeline, its trained
best checkpoints, a simple Streamlit UI, the 2,000-image basic dataset and the
4,000-image extended dataset.

## Code guide

| File | What it does |
| --- | --- |
| `app.py` | Starts the upload, colored-image, steps and LaTeX UI. |
| `src/model.py` | Loads the trained RF-DETR model and detects symbols in an image. |
| `src/data.py` | Prepares images, transforms boxes and reads/writes the COCO dataset. |
| `src/layout.py` | Groups detected symbols into an equation hierarchy and produces LaTeX. |
| `src/structures.py` | Optional fraction, root and subscript grouping; reuses the same operator parser. |
| `src/profiles.py` | Keeps the original and extended vocabularies/settings separate. |
| `src/render.py` | Draws composable fractions, roots and scripts with exact symbol boxes. |
| `src/train.py` | Previews settings or starts an explicitly requested training run. |
| `src/predict.py` | Runs terminal prediction and saves its results. |
| `src/evaluate.py` | Scores symbol detection and complete-equation accuracy. |
| `src/visualize.py` | Colors symbols, draws regions and generates hierarchy steps. |
| `src/validation.py` | Checks parent/child consistency and creates the foreground mask. |
| `ui/inference.py` | Connects the UI to CPU model inference. |
| `ui/images.py` | Decodes uploads and highlights the current region. |
| `ui/steps.py` | Validates UI nodes/steps and describes each step. |
| `generate_data.py` | Generates a new synthetic small-equation dataset. |
| `prepare_data.py` | Validates data or exports an explicitly selected annotated dataset. |
| `prepare_kaggle.py` | Prepares the standalone Kaggle training package. |
| `configs/training.json` | Contains model, training and inference settings. |
| `tests/test_pipeline.py` | Checks the data, layout, model adapter and packaging. |

The neural-network architecture comes from the pinned RF-DETR library.
`src/model.py` is this project's loading/inference code; `src/layout.py` builds
the hierarchy from detected symbols using spatial rules.

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
python -m src.predict --image path/to/equation.png --device cpu
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
python prepare_data.py --out dataset_symbols_small --validate-only
python -m src.evaluate --data dataset_symbols_small
python -m unittest discover -s tests -p "test_*.py"
```

## Training

Preview settings without constructing a model or starting training:

```powershell
python -m src.train
```

For an explicitly requested new CUDA run, choose an empty output directory to
preserve the included checkpoint:

```powershell
python -m src.train --start-training --device cuda --output checkpoints/new_run
```

To generate additional synthetic data, choose a new directory:

```powershell
python generate_data.py --out dataset_new --count 2000
```

`prepare_kaggle.py` previews a Kaggle package by default. Its
`--prepare --account USERNAME` option packages source/data locally and does not
upload or start training. The Kaggle wrapper starts training only when submitted
as a GPU job. See the configuration in `configs/training.json`.

## Saved files

The repository also includes training metrics and the run summary beside the
checkpoint, full held-out results in `checkpoints/equation_evaluation/`, and
saved terminal predictions in `outputs/rfdetr_prediction/`. Their input image
is recorded in `predicted_structure.json`; these saved results are independent
of the current `image.png` test input. The current `image.png` contains a
fraction, which is outside this model's supported first-stage scope.

## Trained extended model

The original 19-class checkpoint, default commands, 2,000-image dataset and
simple-equation parser remain the default. New structure handling is opt-in:
stacked fractions, square roots with a detected radical/overbar box, digit or
letter subscripts, and combined subscripts/integer superscripts. Structural
nesting is bounded to four levels; parentheses, functions and arbitrary
multiline equations are still outside the scope.

`src/structures.py` first groups fraction bars and radical spans, then attaches
small raised/lowered symbols, then reuses the existing arithmetic parser.
Hierarchy/LaTeX serialization and visualization are shared with the original
pipeline. Every detected glyph must be used exactly once; missing or ambiguous
structures return a rejected result rather than a fabricated equation.

The UI has a collapsed **Advanced structures** option. Enable **Fractions,
square roots and subscripts** to select the trained extended checkpoint at
`checkpoints/rfdetr_extended/checkpoint_best_total.pth`. Its 24-class profile
appends `√`, `i`, `j`, `k` and `n` without reordering the original 19 IDs.
Keep `symbol_pipeline.json` alongside the weights. Both checkpoints use Git LFS.

Training completed all 30 epochs; the selected EMA checkpoint is from epoch 30,
with validation mAP@0.50:0.95 of **0.9813**. On all **381 held-out synthetic
test images**, symbol precision, recall and F1, exact equation accuracy and
layout validity were **1.0**. Full test results are in
`checkpoints/rfdetr_extended/equation_evaluation/`.

The included `dataset_extended/` contains 3,217 training images, 402 validation
images and 381 test images. Generated-font performance does not establish
real-image accuracy: the local `image.png` test was parsed as
`\frac{a + 3}{2} = 7`, while its actual numerator is `x + 3`.
`outputs/extended_prediction/` preserves this incorrect prediction for review.
The fraction structure was recognized, but the symbol `x` was misclassified.

Run the extended model in the terminal:

```powershell
python -m src.predict --image path/to/equation.png --checkpoint checkpoints/rfdetr_extended/checkpoint_best_total.pth --out outputs/extended_prediction --device cpu
```

Preview the extended training settings without constructing a model or starting
training:

```powershell
python -m src.train --config configs/extended.json
```

For a new dataset, use a separate directory; the generator refuses to overwrite
existing data:

```powershell
python prepare_data.py --out dataset_extended --validate-only
python generate_data.py --profile extended --out dataset_extended_new --count 4000
python prepare_kaggle.py --prepare --config configs/extended.json --data dataset_extended_new
```

These commands validate/generate/package data only; they do not upload or train.
The completed extended run is on Kaggle under
`nishitavedpathak03/equation-ocr-rfdetr-requirements-rfdetr-txt` because an earlier
packaging bug used a source filename for the notebook name. The bug is fixed for
future packages, which use `equation-ocr-rfdetr-extended`; the completed run has
not been renamed or restarted.

## Trained complex equation model

The separate 31-class complex profile adds parentheses and `sin`, `cos`, `log`
with parenthesized arguments, including integer logarithm bases. It combines
these with nested fractions, roots and scripts, and includes Computer Modern
math italic variables and generated screenshot-like degradation. The existing
24 class IDs and trained models are preserved.

The downloaded best checkpoint is
`checkpoints/rfdetr_complex/checkpoint_best_total.pth`, selected from epoch 30
with validation mAP@0.50:0.95 **0.9837**. On 588 held-out synthetic equations,
symbol F1 was **0.999828** and exact equation accuracy **1.0**. The previously
failing nested-root/fraction screenshot was recognized correctly in a CPU UI test.

The UI defaults to **Complex equations (latest)**. Older models are selectable
under the collapsed **Model** control. See [COMPLEX_TRAINING.md](COMPLEX_TRAINING.md)
for the 6,000-image dataset, configuration, saved evaluation and terminal commands.
Synthetic metrics and one screenshot do not establish arbitrary real-image accuracy.

Confidence filtering uses the checkpoint's inference setting (default `0.50`),
without a UI control. Ordinary parentheses retain slightly taller script digits.
