# Dermatology Image Captioning with Retrieval-Augmented Generation and Efficient VLM Fine-Tuning

Generating clinical-style captions for dermatology images from the Fitzpatrick17k dataset, in two stages: a retrieval-augmented captioner builds a caption corpus, and two small vision-language models are then LoRA fine-tuned on it and evaluated — including a fairness audit across Fitzpatrick skin-tone groups.

> **Research code, not a medical device.** Nothing here has been clinically validated. See [Limitations](#limitations).

## Pipeline

```
Stage 1                          Stage 2                    Evaluation
─────────────────────            ──────────────────         ──────────────────
CLIP embeddings                  SmolVLM-500M  ─┐           BERTScore / ROUGE
   ↓                                            ├─ LoRA →  BLEURT
per-label FAISS index    →  caption dataset   │           entailment (AlignScore)
   ↓                          (16,577 images)   │           tone-group fairness
top-12 same-label NN                            │
   ↓                             MedGemma-4B  ──┘
Qwen2-VL-2B captions
```

| Notebook | Stage |
| --- | --- |
| [`notebooks/01_rag_caption_generation.ipynb`](notebooks/01_rag_caption_generation.ipynb) | Retrieval-augmented caption generation |
| [`notebooks/02_smolvlm_finetune_eval.ipynb`](notebooks/02_smolvlm_finetune_eval.ipynb) | SmolVLM-500M LoRA fine-tune + evaluation |
| [`notebooks/03_medgemma_finetune_eval.ipynb`](notebooks/03_medgemma_finetune_eval.ipynb) | MedGemma-4B LoRA fine-tune + evaluation |

Run them in order — Stage 2 consumes the caption CSVs Stage 1 writes.

Each notebook opens with a single **CONFIGURATION** cell holding every path. The notebooks were developed in Google Colab with the dataset on Google Drive, so paths default to a `/content/drive/MyDrive/...` layout; override the `DERM_DRIVE_ROOT` / `DERM_WORK_DIR` environment variables or edit that one cell to run elsewhere.

## Stage 1 — Retrieval-augmented caption generation

The notebook contains **two distinct retrieval paths**, and only the second produced the 16,577-caption corpus. Keep them apart when citing this work.

**Path A — per-image visual retrieval (pilot only, ~1,010 captions)**

1. **Embed** every training image with CLIP ViT-L/14 (`clip-ViT-L-14`).
2. **Index** the embeddings into a **per-label** FAISS `IndexFlatIP` — one index per disease label, so retrieval never crosses a diagnosis boundary.
3. **Retrieve** the **top-12 nearest neighbours** for each target image, restricted to its own label (`top_k_visual_keywords(..., k_nn=12)`).
4. **Generate** with Qwen2-VL-2B-Instruct twice per image — with and without the retrieved context — and pick between them.

This path runs only in the preview cells (a 10-image A/B and a 1,000-image always-RAG run). It never wrote the training corpus.

**Path B — per-label context (production, all 16,577 captions)**

1. **Aggregate** the `concepts` column already present in `train.csv` per disease label, ranked by frequency, into one context string per label — **114 labels, 114 context strings**.
2. **Look up** that context by label at generation time: `ctx = label2ctx.get(lbl, f"Label: {lbl}")`. No FAISS call, no image embedding. Every image sharing a label receives byte-identical context.
3. **Filter** diagnostic vocabulary out of the context (a banned-term list covering `melanoma`, `carcinoma`, `malignant`, `benign`, and the label tokens themselves).
4. **Generate** with Qwen2-VL-2B-Instruct under a system prompt restricting output to visible morphology, colour, border, distribution and surface features. A post-filter drops any sentence containing diagnosis, treatment or prognosis language.

So the corpus the fine-tuned models learned from is conditioned on **label-level** retrieved concepts, not per-image visual neighbours.

Captions completed, after a resume-and-patch pass that filled every gap:

| Split | Captions |
| --- | --- |
| train | 11,787 |
| val | 1,474 |
| test | 3,316 |
| **total** | **16,577** |

## Stage 2 — LoRA fine-tuning

Both models were trained on the Stage-1 captions. Note the two runs differ in precision strategy — each uses one of 4-bit quantization or Flash Attention 2, not both:

| | SmolVLM | MedGemma |
| --- | --- | --- |
| Base model | `HuggingFaceTB/SmolVLM-500M-Instruct` | `google/medgemma-4b-it` |
| LoRA | r=32, α=32, RS-LoRA, dropout 0.05, targeted projections | r=16, α=16, `all-linear` |
| Precision | bf16 + **Flash Attention 2** | **4-bit NF4** (double quant) + eager attention |
| Epochs | 2 | 1 |
| Optimiser steps | 5,894 | 737 |
| Wall-clock | 13.4 h (48,144 s) | 3.9 h (14,189 s) |
| Batch × grad-accum | 1 × 4 | 4 × 4 |
| LR / schedule | 2e-4, constant | 2e-4, linear |
| Trainable params | 114.8 M / 622.3 M (18.4 %) | 1.381 B / 5.681 B (24.3 %) |
| Final training loss | 0.544 | 0.484 |

Both keep `lm_head` and `embed_tokens` trainable, which is most of the trainable-parameter count.

## Evaluation

3,316 test captions, scored with BERTScore, ROUGE, BLEURT-20 and an NLI entailment score (`AlignScore`). Each notebook contains **two evaluation passes** over different prediction files. They are not two measurements of the same thing:

- **Pass A** reads predictions generated with the **LoRA adapter attached** (verified for SmolVLM by matching its rows against the LoRA inference output, and for MedGemma by an explicit `trainer.model` call).
- **Pass B (SmolVLM)** reads `smolvlm_test_predictions.csv`, which is written by an inference cell that loads `HuggingFaceTB/SmolVLM-500M-Instruct` and **never attaches the adapter** — it is a **zero-shot base-model** run.

That is what the 3× ROUGE-1 gap measures. Cite **pass A** for fine-tuned performance; pass B is a useful base-model baseline. Both merges are inner joins on normalised image id with no empty-caption filtering, so the 3,316 row count is identical across passes.

| Metric | SmolVLM (A: LoRA / B: base) | MedGemma (pass A / pass B) |
| --- | --- | --- |
| Overall | 0.362 / 0.223 | 0.389 / 0.319 |
| Relevance average | — / 0.317 | — / 0.456 |
| BERTScore (recall) | 0.621 / 0.435 | 0.635 / 0.529 |
| ROUGE-1 | 0.496 / 0.155 | 0.522 / 0.308 |
| BLEURT-20 (norm) | — / 0.331 | — / 0.574 |
| Image–caption similarity | 0.375 / 0.348 | 0.406 / 0.412 |
| Keyword concept F1 | — / 0.113 | — / 0.223 |
| AlignScore (entailment) | — / 0.143 | — / 0.140 |
| Factuality average | — / 0.128 | — / 0.182 |

MedGemma outperforms SmolVLM on every relevance metric in both passes, which is the expected ordering given 4 B vs 0.5 B parameters.

### Fairness audit — BERTScore by Fitzpatrick tone

Per-caption BERTScore for the SmolVLM predictions, grouped by Fitzpatrick scale (1–2 Light, 3–4 Medium, 5–6 Dark; rows with an invalid scale of −1 excluded, leaving 3,209 of 3,316):

| Tone group | n | Precision | Recall | **F1** |
| --- | --- | --- | --- | --- |
| Dark | 457 | 0.845 | 0.846 | **0.846** |
| Medium | 1,168 | 0.843 | 0.843 | **0.843** |
| Light | 1,584 | 0.840 | 0.840 | **0.840** |

The spread is 0.006 F1 — no tone group is meaningfully disadvantaged on this metric. Two caveats keep this from being a clean bill of health: darker tones are the smallest group (457 images, 14 %), and BERTScore measures wording overlap with the reference, not clinical correctness. A model can score evenly across tones while being uniformly wrong.


## Setup

```bash
pip install -r requirements.txt
```

A CUDA GPU is required. Flash Attention 2 needs an Ampere-or-newer card; MedGemma's 4-bit path needs `bitsandbytes`. Set `HF_TOKEN` in your environment (or as a Colab secret) before running — MedGemma is a gated model. No notebook contains a credential.

### Pinned versions

These are the versions the notebooks pin explicitly. Everything else (`torch`, `peft`, `trl`, `datasets`, `faiss-cpu`, …) was installed unpinned in Colab and is listed without a version in `requirements.txt`.

```
flash_attn==2.7.3           bert-score==0.3.13         pandas==2.2.2
tokenizers==0.15.2          numpy==2.1.3               sentence-transformers==2.7.0
transformers==4.38.2  (evaluation)   transformers==4.44.2  (--no-deps, some eval cells)
transformers>=4.45.0  (Stage 1)      accelerate>=0.33.0    qwen-vl-utils>=0.0.8
torch — unpinned, from https://download.pytorch.org/whl/cu121
BLEURT-20 — git+https://github.com/google-research/bleurt.git (no PyPI release)
en_core_sci_md 0.5.4 — installed from the scispacy release URL
```

The three `transformers` versions are not a mistake in this list: the notebooks really do reinstall it between stages. See [`requirements.txt`](requirements.txt) for which block goes with which notebook.

## Data

[Fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k) — clinical dermatology images annotated with Fitzpatrick skin-tone scale and disease labels. Not redistributed here; obtain it from the source under its own terms. The notebooks expect `train.csv` / `valid.csv` / `test.csv` with `image_id`, `label` and `concepts` columns, alongside `Train-image/`, `Valid-image/` and `Test-image/` folders.

## Repository layout

```
notebooks/
  01_rag_caption_generation.ipynb    Stage 1: CLIP + FAISS retrieval → Qwen2-VL-2B captions
  02_smolvlm_finetune_eval.ipynb     Stage 2a: SmolVLM-500M LoRA + evaluation + tone audit
  03_medgemma_finetune_eval.ipynb    Stage 2b: MedGemma-4B LoRA + evaluation
requirements.txt
```

A fourth exploratory notebook (Qwen2.5-VL-3B) was excluded: it has no completed training run.

## Attribution

Joint work by **Mahmudul Hoque** and **Raisa Nusrat Chowdhury**.

The SmolVLM notebook was adapted from a third-party script; the MedGemma notebook was adapted from Google's MedGemma fine-tuning notebook (Apache-2.0, copyright header retained in the file). Base models are used under their respective licences: SmolVLM (Apache-2.0), MedGemma (Health AI Developer Foundations terms), Qwen2-VL (Apache-2.0).
