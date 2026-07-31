# Detecting retinal damage with transfer learning

This project classified retinal optical coherence tomography (OCT) scans into four diagnostic classes using two ImageNet-pretrained backbones — **InceptionV3** and **ResNet50** — each frozen as a feature extractor under a small trainable head, trained and evaluated in Google Colab on the Kermany 2018 OCT dataset. The two notebooks run the same recipe at the two backbones' native input sizes (299×299 and 224×224) so the comparison is between the architectures rather than between two different pipelines.

OCT takes high-resolution cross-sections of the retina of a living patient. Roughly 30 million scans are performed each year and reading them is slow specialist work (Swanson & Fujimoto, 2017), which is what makes automated triage worth testing. The question here was narrower than "can this diagnose": it was how far a frozen ImageNet backbone plus two dense layers gets on medical imagery that looks nothing like ImageNet, and which of the two backbones carries further.

## The classes

| Class | Meaning |
|---|---|
| `CNV` | Choroidal neovascularization — abnormal new blood vessels growing under the retina, the hallmark of wet age-related macular degeneration |
| `DME` | Diabetic macular edema — fluid accumulation and retinal thickening caused by diabetes |
| `DRUSEN` | Yellow deposits beneath the retina; the early marker of age-related macular degeneration |
| `NORMAL` | Healthy retina |

## Results from the original runs

These are the numbers produced by the **original** version of the notebooks, before the pipeline fixes described in the next section. They are reported here as the historical record of what the project actually measured. They are **not** a clean benchmark of either architecture — see "Issues found and fixed" for why, particularly for ResNet50.

| | InceptionV3 | ResNet50 |
|---|---|---|
| Test accuracy | **0.86** | **0.71** |
| Evaluated on | 160 held-out images | 160 held-out images |
| Training images | 4,000 (1,000 per class) | 8,000 (2,000 per class) |
| Input size | 299×299 | 224×224 |
| GPU | Colab L4 | Colab T4 |
| Epochs | 20 (ran to completion) | early-stopped at 12 of 80 |

Per-class precision and recall:

| Class | InceptionV3 P | InceptionV3 R | ResNet50 P | ResNet50 R |
|---|---|---|---|---|
| NORMAL | 0.93 | 0.95 | 0.85 | 0.82 |
| CNV | 0.76 | 0.97 | 0.68 | 0.68 |
| DME | 0.96 | 0.68 | 0.57 | 0.65 |
| DRUSEN | 0.85 | 0.85 | 0.80 | 0.70 |

The InceptionV3 asymmetries are the interesting part: CNV recall of 0.97 against precision of 0.76, mirrored by DME precision of 0.96 against recall of 0.68, is a model biased toward calling things CNV and paying for it by missing DME cases. On a screening task that trade is not automatically wrong, but it is a property of the model rather than a choice anyone made.

## Issues found and fixed

Auditing the original pipeline turned up four defects, three of which materially distort the numbers above. The notebooks in this repository are the corrected versions.

**1. The best checkpoint was saved but never loaded.** `ModelCheckpoint` wrote the best-validation weights to disk — for InceptionV3 that was epoch 16 at `val_accuracy` 0.969 — and then evaluation ran straight off the in-memory model. `EarlyStopping(restore_best_weights=True)` only restores weights when it actually fires a stop, and the InceptionV3 run completed all 20 epochs, so nothing restored anything. The 0.86 was scored on the final-epoch, overfit weights while a better model sat unused in a file. The fixed notebooks reload the checkpoint before evaluating.

**2. ResNet50 was fed the wrong input distribution.** Images were scaled to `[0, 1]` by dividing by 255 and handed to ResNet50 as RGB. ResNet50's Keras weights are caffe-mode: they expect BGR channel order with the ImageNet per-channel means subtracted from 0–255 values, which is what `tensorflow.keras.applications.resnet50.preprocess_input` does. Every filter in the frozen backbone was therefore looking at a distribution it had never seen. Training accuracy never climbed past roughly 0.59, which is the tell. The 0.71 is a measurement of a broken input pipeline, not of the architecture — ResNet50 was never given a fair run. The fixed notebooks load raw 0–255 arrays and apply each model's own `preprocess_input`, so InceptionV3 gets its `[-1, 1]` scaling and ResNet50 gets caffe mode.

**3. The test set doubled as the validation set.** The same images that drove `ModelCheckpoint` and `EarlyStopping` were then used to report accuracy, so model selection was tuned on the reported test set and the headline numbers are optimistic by construction. The fixed notebooks use the dataset's own `val` split for callbacks and touch `test` exactly once, at the end. That split is only 32 images, which makes checkpoint selection noisy — a real limitation, but a smaller one than leaking the test set.

**4. Evaluation used 160 of about 968 available test images.** The per-class figures above rest on roughly 40 images per class, so a single flipped prediction moves a recall figure by 2–3 points and the differences between classes are inside the noise. The fixed notebooks evaluate on the full test split.

A fifth, milder issue: file lists were truncated in `os.listdir` order. Kermany filenames encode patient IDs, so taking the first N images clusters the sample on a handful of patients. The loader now shuffles with a fixed seed before truncating.

**The corrected notebooks have not yet been re-run on Colab.** The results table above is the original, defective run. Re-running is expected to improve both numbers materially — InceptionV3 because it will finally be scored on its best weights, ResNet50 because it will finally see the inputs its weights were trained for — and to shrink the gap between the two architectures, which the preprocessing bug had inflated.

## Getting the data

The dataset is [`paultimothymooney/kermany2018`](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) on Kaggle — 83,484 training images, 968 test, 32 validation, about 10.8 GB compressed. It is released under **CC BY-NC-SA 4.0**, which is a different and more restrictive license than this repository's code license: non-commercial use only, attribution required, share-alike on derivatives.

Download it with the Kaggle CLI. You need an API token from Kaggle → Settings → API → Create New Token:

```bash
pip install kaggle
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download -d paultimothymooney/kermany2018
unzip kermany2018.zip
```

The archive unpacks into a directory whose name ends in a space — `OCT2017 ` — which is not a typo in the notebooks. Inside it are `train/`, `val/` and `test/`, each holding one folder per class.

In Colab, the first notebook cell opens a file picker for `kaggle.json` and downloads the dataset through the Kaggle Python API instead; skip that cell when running locally.

## Running the notebooks

Two notebooks, one per architecture, each self-contained from download to classification report:

- [`inceptionv3-transfer-learning.ipynb`](inceptionv3-transfer-learning.ipynb) — 299×299 inputs, 1,000 images per class, 20 epochs
- [`resnet50-transfer-learning.ipynb`](resnet50-transfer-learning.ipynb) — 224×224 inputs, 2,000 images per class, up to 80 epochs with early stopping

Both carry an Open-in-Colab badge in their first cell, which is the path of least resistance: Colab already has TensorFlow, and a GPU runtime is strongly recommended. The whole training set is loaded into memory as float arrays, so a high-RAM runtime helps at the 2,000-per-class setting.

To run locally instead:

```bash
pip install -r requirements.txt
jupyter notebook
```

The sampling caps (`IMAGES_PER_CLASS`), batch size and epoch count are constants in each notebook's setup cell — lower them if you are memory-constrained.

## Limitations

**Patient-level grouping is not enforced.** Kermany filenames encode patient IDs, and images from one patient can land in both the training sample and the evaluation set. The official train/test split was built with that separation in mind, but the sampling in these notebooks selects images independently within each split rather than grouping by patient, so any leakage present in the source split is preserved rather than corrected. A rigorous version would parse patient IDs out of the filenames and split on them.

**The original evaluation used a small test subset** — 160 images, about 40 per class. Confidence intervals on those per-class figures are wide, and small differences between classes should not be read as real. The corrected notebooks use the full 968-image test split.

**Both backbones are frozen.** Only the head trains, so the features are whatever ImageNet taught the network about natural images. Fine-tuning the upper convolutional blocks is the obvious next step and is not attempted here.

**The 32-image validation split is very coarse.** At 8 images per class, `val_accuracy` moves in increments of about 3 points, which makes best-checkpoint selection noisier than the metric's precision suggests.

## References

Kermany, D. S., Goldbaum, M., Cai, W., Valentim, C. C. S., Liang, H., Baxter, S. L., et al. (2018). Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. *Cell*, 172(5), 1122–1131.e9. https://doi.org/10.1016/j.cell.2018.02.010

Swanson, E. A., & Fujimoto, J. G. (2017). The ecosystem that powered the translation of OCT from fundamental research to clinical and commercial impact. *Biomedical Optics Express*, 8(3), 1638–1664. https://doi.org/10.1364/BOE.8.001638

## License

The code in this repository is MIT licensed — see [LICENSE](LICENSE). The Kermany 2018 dataset is **not** covered by that license; it carries CC BY-NC-SA 4.0 and its terms apply independently to any use of the data.
