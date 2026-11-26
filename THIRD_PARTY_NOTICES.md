# Third-Party Licenses, Datasets, Models & Attributions

MouseSeg builds upon the following open-science datasets, pretrained teacher models, and open-source software libraries. Each third-party component retains its original license listed below.

## 1. Upstream Datasets & Sample Data

1. **Whole-Body Mouse Micro-CT Database (`Rosenhain et al., 2018`)**
   - Citation: Rosenhain, S., Magnuska, Z. A., Yamoah, G. G., Al Rawashdeh, W., Kiessling, F., & Gremse, F. (2018). *A preclinical micro-computed tomography database including 3D whole body organ segmentations*. Scientific Data, 5(1), 180294. https://doi.org/10.1038/sdata.2018.294
   - Data Repositories: Figshare native (`https://doi.org/10.6084/m9.figshare.7058615.v1`) and contrast-enhanced (`https://doi.org/10.6084/m9.figshare.7058618.v1`).
   - License: **CC0 1.0 Universal (`CC0-1.0`)** for the Figshare imaging datasets and masks (article text **CC BY 4.0**).
   - Sample CT Notice (`sample_mouse_ct.nii.gz` in the companion web application): Resampled from Rosenhain et al. (2018) contrast-enhanced mouse `M01_024h` (`CC0-1.0`).

2. **TumSeg Subcutaneous Tumor Micro-CT Database (`Jensen et al., 2024`)**
   - Citation: Jensen, M., Clemmensen, A., Hansen, J. G., van Krimpen Mortensen, J., Christensen, E. N., Kjaer, A., & Ripa, R. S. (2024). *3D whole body preclinical micro-CT database of subcutaneous tumors in mice with annotations from 3 annotators*. Scientific Data, 11(1), 1021. https://doi.org/10.1038/s41597-024-03814-y
   - Dataset Archive: University of Copenhagen Electronic Research Data Archive (ERDA), `https://doi.org/10.17894/UCPH.F7BCF864-BE18-4A16-95AD-6F22DEDB4265` (ERDA archive **CC BY 4.0**; published article text and figures **CC BY-NC-ND 4.0**).

## 2. Pretrained Teacher Models & Benchmark References

1. **AutoMICE (`Jiang et al., 2024`)**
   - Citation: Jiang, L., Xu, D., Xu, Q., Chatziioannou, A., Iwamoto, K. S., Hui, S., & Sheng, K. (2024). *Robust Automated Mouse Micro-CT Segmentation Using Swin UNEt TRansformers*. Bioengineering, 11(12), 1255. https://doi.org/10.3390/bioengineering11121255
   - Software & Hugging Face Weights (`github.com/namijiang/AutoMICE`, `huggingface.co/namijiang98/AutoMICE`): **Apache License 2.0 (`Apache-2.0`)**.
   - Zenodo Model Weights (`https://doi.org/10.5281/zenodo.13841139`): **Creative Commons Attribution 4.0 International (`CC BY 4.0`)**.

2. **TumSeg Pretrained 3D U-Net Models (`KjaerLab/TumSeg`)**
   - Repository: `https://github.com/KjaerLab/TumSeg`
   - License: **Creative Commons Attribution-NonCommercial 4.0 International (`CC BY-NC 4.0`)** (`https://creativecommons.org/licenses/by-nc/4.0/`).

3. **AIMOS (`Schoppe et al., 2020`, Reference Benchmark Only)**
   - Citation: Schoppe, O., Pan, C., Coronel, J., Mai, H., Rong, Z., Todorov, M. I., Müskes, A., Navarro, F., Li, H., Ertürk, A., & Menze, B. H. (2020). *Deep learning-enabled multi-organ segmentation in whole-body mouse scans*. Nature Communications, 11(1), 5626. https://doi.org/10.1038/s41467-020-19449-7 (article **CC BY 4.0**; no `OSchoppe/AIMOS` source code is bundled or redistributed).

## 3. Companion Browser Application Runtime

- **ONNX Runtime Web (`onnxruntime-web`)** (used in the companion web app at `https://mouseseg.joeyschulz.com`)
  - Copyright (c) Microsoft Corporation. Licensed under the **MIT License**.

## 4. Python Inference & Training Dependencies

- **NumPy & SciPy** (`requirements.txt`): **BSD-3-Clause** (binary SciPy wheels may bundle `libquadmath` under `LGPL-2.1-or-later`)
- **NiBabel** (`requirements.txt`): **MIT**
- **ONNX Runtime (`onnxruntime`)** (`requirements.txt`): **MIT**
- **PyTorch (`torch`)** (`requirements-train.txt`, optional for training and fine-tuning only): **BSD-3-Clause**
- **ONNX (`onnx`)** (`requirements-train.txt`, optional for model export only): **Apache-2.0**
- **MONAI & nnU-Net** (upstream teacher and methodological references, not required at runtime): **Apache-2.0**
