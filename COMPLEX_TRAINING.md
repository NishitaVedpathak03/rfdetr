# Complex equation model and preparation

The complex Kaggle run completed all 30 epochs. The downloaded best EMA checkpoint
is installed at `checkpoints/rfdetr_complex/checkpoint_best_total.pth`, alongside
its `symbol_pipeline.json`, metrics and run summary. The UI selects **Complex
equations (latest)** by default; the older models remain available in the
collapsed **Model** selector. Preparation commands themselves do not start training.

## Downloaded model results

The selected checkpoint is from epoch 30, with validation mAP@0.50:0.95 of
**0.9837**. On the 588-image synthetic test set, symbol precision was **0.999656**,
recall **1.0**, F1 **0.999828**, and exact equation accuracy **1.0**. Full results
are in `checkpoints/rfdetr_complex/equation_evaluation/`.

The original screenshot with `sqrt((x_2 + 1)/(x^2 - 1)) = 2` was also recognized
correctly in a real CPU Streamlit test. This is one screenshot check and does not
establish accuracy on arbitrary images. Terminal prediction uses:

```powershell
python -m src.predict --image path/to/equation.png --checkpoint checkpoints/rfdetr_complex/checkpoint_best_total.pth --device cpu
```

## What is included

- Nested fractions and square roots, integer powers, letter/digit subscripts and combined scripts.
- Parenthesized expressions and powers of parenthesized expressions.
- `sin(...)`, `cos(...)`, `log(...)` and integer-base logarithms such as `log_2(...)`.
- Function arguments containing fractions/scripts; functions in numerators and denominators.
- Computer Modern Roman digits/operators with math italic variables, plus alternate fonts.
- Generated low-resolution, lightly blurred and JPEG-compressed variants.
- Basic equations and x/a/n contrast examples to retain simpler skills.

The dataset has 31 symbol classes. The existing 24 IDs are preserved; `(`, `)`,
`s`, `c`, `l`, `o` and `g` are appended. Function names are composed from separately
annotated letters. The square-root label includes its radical/overbar span;
fraction bars are labelled `-`.

Preparation uses only generated images. The supplied screenshot is not inserted
into the dataset or used as an inference reference. Every un-degraded generated
equation is reconstructed from its glyph boxes and checked against its intended
LaTeX. Formula/font variants and identical pixels cannot cross dataset splits;
isolated calibration glyphs are restricted to training.

## Files and checks

`dataset_complex/` contains 6,000 COCO images with annotations and evaluation
references. See `symbol_dataset.json` for split/class counts and
`preparation_report.json` for curriculum/font coverage and preparation status.
The split contains 4,821 training, 591 validation and 588 test images. Every
curriculum family contains both Computer Modern math italic and alternate-font
examples; font choice is varied by curriculum round to avoid coupling a family
to one font style.

Annotation boxes have a small **2-pixel margin on each side**, measured in the
prepared 768-pixel images and clipped at image boundaries. This changes training
targets only; image pixels and visual spacing are unchanged. Original tight
boxes are retained in `tight_bbox`, so reapplying padding does not accumulate.
Future complex generation uses the same 2-pixel default (`--box-padding` can
override it); original small/extended generation keeps its existing default.
All 5,907 non-calibration equations were also reconstructed correctly from the
padded ground-truth boxes. This verifies geometry compatibility, not model
recognition accuracy.

```powershell
python prepare_data.py --out dataset_complex --box-padding 2
```

This updates local annotations only. It does not update the already uploaded
Kaggle dataset or start training. The local upload ZIP must be refreshed after
annotation changes; upload it as a new dataset version when explicitly ready.

```powershell
python prepare_data.py --out dataset_complex --validate-only
python -m src.train --config configs/complex.json
```

The second command previews settings only. `configs/complex.json` selects RF-DETR
Nano at 768 pixels, a maximum of 96 symbols, batch size 1, accumulation 8 and up
to 30 epochs with early stopping. These are planned settings, not measured
training speed or guaranteed recognition accuracy.

The local Kaggle package is `kaggle_rfdetr_complex/`; the local upload bundle is
`kaggle_rfdetr_complex_upload/`. To regenerate the package without uploading or
training:

```powershell
python prepare_kaggle.py --prepare --config configs/complex.json --data dataset_complex --account nishitavedpathak03
```

Preparing this package does not submit or restart a run. The small/extended
datasets and weights are kept separate. Choose a new empty output directory for
any future training so the downloaded complex model is preserved.

## Scope and validation

Functions require parenthesized arguments. Structural parsing is bounded to
eight recursive levels. Greek letters, integrals, matrices, arbitrary functions,
arbitrary superscript expressions and general multiline layouts are outside
this prepared profile.

The small/extended checkpoints have not learned the new parentheses/function
labels. Select the complex checkpoint to use those classes. New model versions
need held-out symbol and exact-equation evaluation, followed by independent
screenshot testing. Geometry checks are not model accuracy measurements.
