# FlowCIR: Semantic Transport via Flow Matching for Zero-Shot Composed Image Retrieval

**ECCV 2026**  
Zhenqi He, Ziqi Jiang, Yuanpei Liu, Yanghao Wang, Teng Wang, Long Chen  
LONG Group, HKUST

FlowCIR learns semantic transport with flow matching for zero-shot composed image retrieval.

## Setup

```bash
conda create --name flowcir python=3.10 -y
conda activate flowcir
pip install torch torchvision numpy pillow tqdm datasets transformers
pip install git+https://github.com/openai/CLIP.git
```

Run the following data preparation, training, validation, and submission commands from the code directory:

```bash
cd code
```

## Data Preparation

Download HQ-Edit and the target CIR datasets (CIRR, CIRCO, and Fashion-IQ). Configure the CIRR and CIRCO locations in `config.py`, or use environment variables:

```bash
export FLOWCIR_HQEDIT_DIR=/path/to/HQ-Edit
export FLOWCIR_CIRR_DIR=/path/to/CIRR
export FLOWCIR_CIRCO_DIR=/path/to/CIRCO
```

Before extracting test features, copy `captions/cap.rc2.test1_posneg.json` to `<CIRR_DIR>/cirr/captions/` and `captions/test_posneg.json` to `<CIRCO_DIR>/annotations/`, replacing the corresponding non-posneg test annotations when necessary.

Use `extract_features/extracted_features_clip_posneg.py` to extract the positive and negative text features from the posneg annotations.

## Feature Extraction

Extract HQ-Edit training features:

```bash
bash myScripts/extract_hqedit_feat.sh
```

Extract CIRR or CIRCO retrieval features:

```bash
DATASET=cirr bash myScripts/extract_cir_feat.sh
DATASET=circo bash myScripts/extract_cir_feat.sh
```

## Training and Validation

```bash
bash myScripts/train.sh
MODEL_DIR=/path/to/checkpoint_directory \
  EPOCHS="10 15 20" END_POINTS="0.4 0.5 0.6" \
  bash myScripts/val.sh
```

Checkpoints and logs are written to the directories configured by `FLOWCIR_SAVE_ROOT` and `FLOWCIR_LOG_DIR`. Empirically, training for about 200 epochs is usually sufficient. Select the checkpoint epoch and inference end time according to validation performance; end times between 0.4 and 0.6 generally work well.

## Submission

Generate the final submission with the epoch and end time selected on the validation set:

```bash
MODEL_DIR=/path/to/checkpoint_directory \
  DATASET=cirr EPOCH=15 END_POINT=0.5 \
  bash myScripts/submit.sh
```

Use `DATASET=circo` to generate a CIRCO submission. Submission files are saved under `FLOWCIR_SUBMISSION_DIR`.

## Acknowledgement

This repository is developed in part based on *Exploring Cross-Modal Flows for Few-Shot Learning (FMA)* (ICLR 2026). It also uses [OpenAI CLIP](https://github.com/openai/CLIP) and the publicly available HQ-Edit, CIRR, and CIRCO resources. We thank their authors for making these projects available.

## Citation

If you find this work useful, please cite [flowcir.bib](code/flowcir.bib):

```bibtex
@inproceedings{he2026flowcir,
  title     = {FlowCIR: Semantic Transport via Flow Matching for Zero-Shot Composed Image Retrieval},
  author    = {He, Zhenqi and Jiang, Ziqi and Liu, Yuanpei and Wang, Yanghao and Wang, Teng and Chen, Long},
  booktitle = {European Conference on Computer Vision},
  year      = {2026}
}
```
