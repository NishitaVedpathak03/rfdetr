# Trained calculus OCR model

The Kaggle run `nishitavedpathak/equation-ocr-rfdetr-calculus` completed all
30 epochs. The downloaded **best EMA checkpoint is from epoch 30** (CSV epoch
index 29), with validation mAP@0.50:0.95 **0.986153**.

Installed model: `checkpoints/rfdetr_calculus/checkpoint_best_total.pth`, together
with its required `symbol_pipeline.json` label/preprocessing metadata. Its SHA256
is `db3aab7a63a3b26396c64609ed91eb17f34801adfed31883e76f17a6b463497f`.
The UI automatically offers **Calculus equations** first when this model exists.

The saved Kaggle evaluation covers all **816 held-out synthetic test images**:
symbol precision **0.999897**, recall **1.0**, F1 **0.999949**, and exact equation
accuracy **1.0**. These results use prepared dataset images; real uploads use the
UI preprocessing path. The evaluation files are saved under
`checkpoints/rfdetr_calculus/equation_evaluation/`.

Local testing also passed **10/10 predetermined synthetic held-out uploads**
through Streamlit's actual inference path at confidence 0.50. The cases cover
trigonometry, nested roots, fractions, integrals, derivatives and finite/infinite
sums. Each produced the reference LaTeX, both image panels and working Next/Reset
controls; navigation reused the cached prediction. Symbol precision, recall and
F1 at IoU 0.50 were all 1.0 on these ten images. Saved images, predictions and
the report are in `outputs/calculus_ui_test/REPORT.md`.

This is a separate **39-class** RF-DETR Nano profile at **768 pixels**, configured
for up to **30 epochs**. The working 31-class complex checkpoint, its dataset,
and the current UI selection are preserved. The first 31 class IDs are unchanged.
The new IDs append `t`, `d`, `f`, `C`, `∫`, `∑`, `∂`, and `∞`.

“Derivation” is interpreted as differentiation notation. This remains an OCR
project: it reads symbols and rebuilds LaTeX; it does not solve calculus problems
or generate mathematical proofs.

## Prepared scope

- `sin`, `cos`, `tan`, `log`, and `f` with parenthesized arguments.
- Fractions, square roots, powers, subscripts, and nested parentheses from the
  existing profiles, mixed into the new curriculum.
- Single indefinite integrals with explicit `d x`, `d y`, or `d z` differentials.
- Definite integrals with simple paired upper/lower limits.
- Derivative fractions such as `d/dx`, `dy/dx`, `d²y/dx²`, and `∂/∂x`.
- Finite/infinite summations with an `i`, `j`, `k`, or `n` index, an integer start,
  and a simple upper limit such as a number, `n`, or infinity. The summand is a
  parenthesized expression or a single scripted variable/number.

Prime notation, double/triple/contour integrals, limits such as `x → 0`, Greek
variables, matrices, arbitrary multiline equations, and automatic solving are
outside this first calculus profile. Unsupported or incomplete geometry is
rejected rather than repaired with guessed symbols.

## Dataset

`dataset_calculus/` is a separate 8,000-image COCO dataset. It includes 16 families:
basic/fraction/root/complex replay, single/mixed/fraction trigonometry,
polynomial/trigonometric/definite integrals, derivative operators and fractions,
second and partial derivatives, and finite/infinite summations. Replayed older
families help retain the existing tasks.

The prepared split has **6,339 training**, **845 validation**, and **816 test**
images. All 39 classes occur in each split. Infinity coverage was repaired by
moving one complete 19-image formula group from training to test, without
changing image bytes or annotation boxes.

Computer Modern math italic and alternate fonts occur across all families.
STIXGeneral supplies the special mathematical glyphs; character-map coverage is
checked before generation. Isolated calibration glyphs stay in training only.
Formula groups and identical image pixels cannot cross splits.
New calculus classes are represented in every split by transferring whole
formula groups when a random split misses one; calibration glyphs stay in training.
Geometry and reference LaTeX are checked before image degradation; the padded
COCO boxes are also checked separately. Annotation boxes keep a **2-pixel margin per side**.
These checks establish dataset/parser consistency, not recognition accuracy.

Only generated examples are used. No supplied screenshot is added to training.
See `symbol_dataset.json` and `preparation_report.json` for the actual split,
class, font, and family counts.

## Preparation commands (no training)

The generation command refuses to overwrite an existing nonempty directory.
Once `dataset_calculus` exists, skip generation and validate it instead.

```powershell
cd D:\eq_ocr
python generate_data.py --profile calculus --out dataset_calculus --count 8000 --side 768 --seed 20261015
python prepare_data.py --out dataset_calculus --validate-only
python -m src.train --config configs/calculus.json
python prepare_kaggle.py --prepare --config configs/calculus.json --data dataset_calculus --account nishitavedpathak
```

The `src.train` command above is a **preview**. It does not construct RF-DETR,
download weights, or use a GPU. Packaging creates local files only:

- `kaggle_rfdetr_calculus/`: script and private GPU notebook metadata.
- `kaggle_rfdetr_calculus_upload/dataset.zip`: images, annotations and references.

These preparation commands do not upload or submit a kernel. The submitted run uses the separate `equation-ocr-calculus-symbols` Kaggle dataset
and `equation-ocr-rfdetr-calculus` notebook. New weights go to
`checkpoints/rfdetr_calculus/`; existing trained checkpoints are protected.
The trainer fine-tunes the official RF-DETR pretrained model with the new class
head, rather than resuming an incompatible 31-class optimizer checkpoint.

## Run the UI

```powershell
cd D:\eq_ocr
python -m streamlit run app.py
```

Select **Calculus equations** under **Model** if an existing browser session
retained an older model selection. The original models remain available.
Synthetic test metrics do not establish accuracy on arbitrary screenshots or
handwriting.
