# Amharic Capability Distillation

Response-level knowledge distillation of a multilingual instruction-following model for Amharic.

## Overview

This project distills Amharic language capabilities from a high-capacity teacher model into a compact student model using **response-level knowledge distillation**.

The teacher generates responses to task-specific Amharic prompts, and the student is fine-tuned to reproduce those target outputs. The student model is trained using **LoRA (Low-Rank Adaptation)** while its base pretrained weights remain frozen.

## Models & Environment

| Component | Specification |
|---|---|
| **Teacher Model** | Google Gemini API |
| **Student Model** | `Qwen/Qwen2.5-0.5B-Instruct` |
| **Fine-Tuning Method** | LoRA (PEFT) |
| **Hardware** | NVIDIA Tesla T4 GPU |
| **Frameworks** | PyTorch, Hugging Face Transformers, PEFT, TRL |

## Tasks

The dataset covers four primary capabilities:

1. **Closed-Book Knowledge** (factual QA without reference context)
2. **Instruction Following & Reading Comprehension** (context-grounded QA)
3. **Translation**
   - English to Amharic
   - Amharic to English
4. **Summarization** (extractive and abstractive Amharic news summaries)

## Dataset

An initial prompt pool was constructed with a target of **2,800 prompts** (700 per task). Due to teacher API quota constraints, generation completed for **1,144 prompts**.

The final training dataset contains **1,144 prompt and teacher response pairs**:

| Task | Examples |
|---|---:|
| Translation | 294 |
| Closed-book knowledge | 293 |
| Instruction following | 283 |
| Summarization | 274 |
| **Total** | **1,144** |

### Data Format

The training CSV (`final_training_1144.csv`) adheres to the following schema:

```text
prompt
teacher_response
task
source
```

> **Note:** The 60 held-out evaluation prompts were strictly isolated from the training pool to prevent data leakage.

## Training Configuration

The student model was fine-tuned for 2 epochs using LoRA with response-only loss masking.

### LoRA Hyperparameters
- **Rank ($r$):** 8
- **Alpha ($\alpha$):** 16
- **Dropout:** 0.05
- **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`

### Training Hyperparameters
- **Per-Device Batch Size:** 1
- **Gradient Accumulation Steps:** 4 (effective batch size: 4)
- **Learning Rate:** 2e-4
- **Max Sequence Length:** 256
- **Optimizer:** AdamW
- **Precision:** FP16
- **Gradient Checkpointing:** Enabled
- **Loss Masking:** Response-only loss masking (loss computed solely on teacher response tokens, ignoring prompt tokens with index `-100`)

## Evaluation

The base model and the LoRA-distilled model were evaluated on the exact same 60 held-out prompts under identical greedy decoding conditions.

### Evaluation Parameters
- **Test Set Size:** 60 prompts (15 per task)
- **Batch Size:** 1
- **Decoding Strategy:** Greedy (`do_sample=False`)
- **Max New Tokens:** 256
- **Environment:** NVIDIA Tesla T4 GPU

Teacher outputs were not used during evaluation.

## Experimental Results

| Metric | Base Model (`Qwen2.5-0.5B-Instruct`) | Distilled Model (LoRA) |
|---|---:|---:|
| **Average time / answer** | 6.485 s | 8.325 s |
| **Average generated tokens** | 187.7 | 163.2 |
| **256-token outputs (hit cap)** | 31 / 60 | 32 / 60 |
| **Repetition score** | 0.0981 | 0.2234 |
| **Consecutive-word repetition** | 9 / 60 | 20 / 60 |

### Key Observations & Analysis

- The distilled model did not outperform the base model qualitatively.
- **Teacher Response Quality:** Analysis of the training corpus revealed quality issues in some teacher generations (403 of 1,144 responses were shorter than 10 words, with occasional off-target or hallucinated answers).
- **Model Capacity:** At 0.5B parameters, capacity constraints made learning high-fidelity Amharic generation from limited data challenging.
- **Repetition:** The distilled model exhibited a higher repetition rate under greedy decoding.

## Repository Structure

```text
amharic-capability-distillation/
├── README.md
├── amharic_capability_distillation_master.ipynb
├── final_training_1144.csv
├── predictions_qwen25_05b_lora.csv
└── adapter_qwen25_05b_lora/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── chat_template.jinja
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── training_args.bin
```

## Reproducibility

The end-to-end distillation pipeline is contained in:
[`amharic_capability_distillation_master.ipynb`](amharic_capability_distillation_master.ipynb)

The notebook includes:
1. Public data preprocessing and deduplication.
2. Teacher prompt construction and API generation loop.
3. Dataset quality auditing.
4. LoRA fine-tuning with response-only loss masking.
5. Baseline vs. Distilled model evaluation and metrics computation.

### Steps to Reproduce:
1. Open [`amharic_capability_distillation_master.ipynb`](amharic_capability_distillation_master.ipynb) in Google Colab (GPU runtime).
2. Mount Google Drive or upload the project files.
3. (Optional) Provide a Gemini API key via Colab Secrets if regenerating teacher data.
4. Load `final_training_1144.csv`.
5. Run the LoRA training cell.
6. Execute the evaluation cells to generate prediction files and benchmark statistics.

## Limitations

- **Dataset Size:** Training was limited to 1,144 teacher-generated examples due to API quotas.
- **Teacher Quality:** Presence of brief or noisy teacher responses in the synthetic data pool.
- **Student Model Size:** Small 0.5B parameter architecture.
- **Outcome:** The experiment should be viewed as an empirical study of response-level distillation constraints on low-resource languages rather than a production-ready model deployment.

## License

This repository and model artifacts follow the licenses and usage terms of the underlying base models (`Qwen2.5`) and dataset sources.