# MedViLL — Modernized

A fully modernized reimplementation of **MedViLL** (Medical Vision-Language Learner) that runs on current environments including **Google Colab** and **Kaggle**, with no legacy build dependencies.

## What Changed

The [original MedViLL codebase](https://github.com/SuperSupermoon/MedViLL) requires `pytorch-pretrained-bert==0.6.2` (deprecated/unmaintained), `transformers==3.5.1` (2020), and PyTorch 1.7 — a combination that can no longer be built on modern cloud platforms. This repo replaces all of that with standard pip-installable packages.

| Original | This repo |
|----------|-----------|
| `pytorch-pretrained-bert==0.6.2` | removed — `transformers>=4.40` |
| `transformers==3.5.1` | `transformers>=4.40` |
| `torch==1.7.0` + `pytorch-lightning==1.0.5` | `torch>=2.1` + `accelerate>=0.29` |
| Hardcoded `.cuda()` everywhere | `torch.device()` abstraction (CPU / CUDA / MPS) |
| Custom `BertLayerNorm` | `torch.nn.LayerNorm` |
| LM labels not shifted (generation bug) | Proper next-token prediction |

## Experiments

1. **Retrieval** — Image→Text (I2T) and Text→Image (T2I) with Recall@1/5/10 and MRR  
2. **Diagnosis Classification** — Multi-label (MIMIC-CXR 14 labels) with AUC, F1  
3. **VQA on VQA-RAD** — Accuracy + BLEU-4 on answer strings  
4. **Report Generation** — Seq2seq BLEU-4 and Perplexity  

## Project Structure

```
medvill/
├── config.py                  # Typed dataclass configs
├── models/
│   ├── image_encoder.py       # ResNet-50 / ViT patch embedding
│   ├── medvill_model.py       # Pre-training + 4 task wrappers
│   └── heads.py               # MLM, ITM, Cls, VQA, Gen heads
├── data/
│   ├── pretrain_dataset.py    # MLM + ITM pre-training
│   ├── classification_dataset.py
│   ├── retrieval_dataset.py
│   ├── vqa_dataset.py         # VQA-RAD loader
│   └── generation_dataset.py
├── tasks/
│   ├── classification.py
│   ├── retrieval.py
│   ├── vqa.py
│   └── generation.py
├── metrics/
│   ├── bleu.py                # sacrebleu BLEU-4
│   ├── perplexity.py          # running + batch perplexity
│   ├── retrieval_metrics.py   # R@K, MRR, median rank
│   └── classification_metrics.py  # AUC-ROC, F1
└── utils/
    ├── seed.py
    ├── checkpoint.py
    └── logging_utils.py
scripts/
├── train_pretrain.py
├── train_classification.py
├── train_retrieval.py
├── train_vqa.py
├── train_generation.py
└── evaluate.py
configs/
├── pretrain.yaml
├── classification.yaml
├── retrieval.yaml
├── vqa.yaml
└── generation.yaml
```

## Quick Start

```bash
pip install -r requirements.txt

# Fine-tune classification (adjust config paths first)
python scripts/train_classification.py \
  --config configs/classification.yaml \
  --pretrained outputs/pretrain/pretrain_best.pt

# Evaluate retrieval
python scripts/evaluate.py \
  --task retrieval \
  --config configs/retrieval.yaml \
  --checkpoint outputs/retrieval/ret_best.pt \
  --data_path data/mimic/retrieval_test.jsonl

# VQA (includes BLEU-4 on answer strings)
python scripts/train_vqa.py \
  --config configs/vqa.yaml \
  --vqa_dir data/vqa_rad/ \
  --pretrained outputs/pretrain/pretrain_best.pt
```

## Data Format

All datasets use **JSONL** with one record per line:

```jsonl
{"img": "path/to/image.jpg", "text": "Findings: ...", "label": "Cardiomegaly,Edema"}
```

VQA-RAD uses its original JSON format. See `medvill/data/vqa_dataset.py` for expected directory layout.

## Citations

### Original MedViLL Paper

```bibtex
@inproceedings{moon2022medvill,
  title     = {Multi-modal Understanding and Generation for Medical Images and Text via Vision-Language Pre-Training},
  author    = {Moon, Jong Hak and Lee, Hyungyung and Shin, Woncheol and Kim, Young-Hak and Choi, Edward},
  journal   = {IEEE Journal of Biomedical and Health Informatics},
  year      = {2022},
  doi       = {10.1109/JBHI.2022.3162690}
}
```

### OpenI Dataset

```bibtex
@article{demner2016openi,
  title   = {Preparing a collection of radiology examinations for distribution and retrieval},
  author  = {Demner-Fushman, Dina and Kohli, Marc D and Rosenman, Marc B and
             Shooshan, Sonya E and Rodriguez, Laritza and Antani, Sameer and
             Thoma, George R and McDonald, Clement J},
  journal = {Journal of the American Medical Informatics Association},
  volume  = {23},
  number  = {2},
  pages   = {304--310},
  year    = {2016},
  doi     = {10.1093/jamia/ocv080}
}
```

### MIMIC-CXR

```bibtex
@article{johnson2019mimic,
  title   = {MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports},
  author  = {Johnson, Alistair E W and Pollard, Tom J and Berkowitz, Seth J and
             Greenbaum, Nathaniel R and Lungren, Matthew P and Deng, Chih-ying and
             Mark, Roger G and Horng, Steven},
  journal = {Scientific Data},
  volume  = {6},
  number  = {1},
  pages   = {317},
  year    = {2019},
  doi     = {10.1038/s41597-019-0322-0}
}
```

### VQA-RAD

```bibtex
@article{lau2018vqa_rad,
  title   = {A dataset of clinically generated visual questions and answers about radiology images},
  author  = {Lau, Jason J and Gayen, Soumya and Ben Abacha, Asma and Demner-Fushman, Dina},
  journal = {Scientific Data},
  volume  = {5},
  pages   = {180251},
  year    = {2018},
  doi     = {10.1038/sdata.2018.251}
}
```

## License

This reimplementation follows the [MIT License](LICENSE). The underlying model design and pre-training approach are credited entirely to the original MedViLL authors.
