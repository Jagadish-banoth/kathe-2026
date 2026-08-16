
---
base_model: facebook/nllb-200-distilled-600M
library_name: peft
pipeline_tag: translation
tags:
- lora
- peft
- nllb
- machine-translation
- english-kashmiri
- kashmiri
- kathe-2026
---
# NLLB-200 English → Kashmiri LoRA

A LoRA fine-tuned adapter for **English-to-Kashmiri machine translation**, based on
`facebook/nllb-200-distilled-600M`.

This model was developed for the **KATHE 2026: AI Challenge for Kashmiri Language Translation**.

> **Important:** This repository contains the trained LoRA adapter, not the full
> NLLB-200 base model. The base model is loaded separately from Hugging Face.

---

## Model Details

### Model Description

- **Model type:** NLLB-200 Transformer with LoRA adapter
- **Base model:** `facebook/nllb-200-distilled-600M`
- **Task:** English → Kashmiri translation
- **Source language:** English (`eng_Latn`)
- **Target language:** Kashmiri Arabic script (`kas_Arab`)
- **Fine-tuning method:** LoRA
- **PEFT version:** 0.19.1
- **Training precision:** FP16
- **Training epochs:** 1

### LoRA Configuration

| Parameter         | Value                  |
| ----------------- | ---------------------- |
| LoRA rank (`r`) | 16                     |
| LoRA alpha        | 32                     |
| LoRA dropout      | 0.1                    |
| Target modules    | `q_proj`, `v_proj` |
| Bias              | `none`               |
| Task type         | `SEQ_2_SEQ_LM`       |

---

## Intended Use

This adapter is intended for:

- English-to-Kashmiri machine translation
- Research and experimentation in low-resource language translation
- Kashmiri NLP research
- Evaluation in the KATHE 2026 competition

### Out-of-Scope Use

This model should not be treated as a general-purpose multilingual model or as
a source of authoritative translations.

Translations should be reviewed by a fluent Kashmiri speaker when accuracy is
important, particularly for legal, medical, financial, or other high-stakes
content.

---

## Training Data

The model was fine-tuned using English-Kashmiri parallel sentence data derived
from the **Bharat Parallel Corpus Collection (BPCC)**.

The training data consists of English source sentences paired with Kashmiri
target sentences written in the Arabic script.

The project used separate training, validation, and test splits.

---

## Training Procedure

The model was fine-tuned using **Parameter-Efficient Fine-Tuning (PEFT) with
LoRA**.

Instead of updating the complete NLLB-200 model, LoRA adapters were attached to
selected attention projection layers.

======================================

The trained LoRA adapter is hosted on Hugging Face:

**`JagadishBanoth/nllb-en-kas-lora`**

The inference scripts automatically download the adapter from Hugging Face.

---

## Repository Structure


kathe-2026/
│
├── README.md
├── requirements.txt
├── load_model.py
├── inference.py
└── batch_inference.py

### Training Configuration

```text
Base model:
facebook/nllb-200-distilled-600M

Fine-tuning:
LoRA

Epochs:
1

Learning rate:
2e-4

LoRA rank:
16

LoRA alpha:
32

LoRA dropout:
0.1

Target modules:
q_proj, v_proj


