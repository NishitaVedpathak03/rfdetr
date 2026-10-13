# RF-DETR small-equation checkpoint

This repository contains the selected trained RF-DETR Nano checkpoint and the
metadata required by the Equation OCR project's inference adapter.

- `checkpoint_best_total.pth`: selected best EMA checkpoint, tracked with Git LFS.
- `symbol_pipeline.json`: symbol vocabulary, preprocessing and inference settings.

The detector recognizes 19 symbol classes at a resolution of 512 pixels. The
first-stage pipeline supports single-line integer equations, variables
`x`, `y`, `z`, `a`, `b`, addition, subtraction, multiplication, equality and
integer superscripts. Fractions, roots, functions, parentheses and multiline
expressions are outside this stage.

Training stopped after 28 epochs. The selected model reached validation
detection mAP of 0.9807. On 196 held-out synthetic equations, symbol
precision/recall and exact complete-equation accuracy were 1.0. These results
do not establish accuracy on all real-world images. A separate spatial grammar
in the Equation OCR project reconstructs hierarchy and LaTeX from detections.

## Download

Install Git LFS before cloning, then run `git lfs pull` in the cloned repository
to download the actual checkpoint rather than only its Git pointer.

## Use with the Equation OCR project

Copy both model files into `checkpoints/rfdetr_small/` in the project. Install
the project's `requirements-rfdetr.txt` (RF-DETR 1.11.2), then run:

```powershell
python -m src.predict_rfdetr --image image.png --checkpoint checkpoints/rfdetr_small/checkpoint_best_total.pth --device cpu
```

The project's UI also selects this checkpoint when both files are present.
Datasets, earlier checkpoints and project source code are not included here.
