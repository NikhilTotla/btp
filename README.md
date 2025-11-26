# Machine Translation for Bhojpuri, Magahi Maithili using low resource monolingual and bilingual corpus

This repository documents the progression of fine-tuning machine translation models for Indic languages, moving from high-resource settings (Hindi-English) to low-resource scenarios (Bhojpuri, Magahi). The project involves creating custom tokenizers to handle unseen languages and leveraging advanced techniques like LoRA and back-translation for extremely low-resource dialects.

## Project Roadmap

1.  **Preliminary Stage**:
    * Fine-tuning IndicTrans on **Hindi -> English**.
    * Building a **custom Bhojpuri tokenizer** to address vocabulary limitations in the base IndicTrans model.
2.  **IndicTrans Fine-tuning**: Training on **Hindi -> Bhojpuri** using the custom vocabulary.
3.  **NLLB Fine-tuning**: Full fine-tuning of NLLB-200 for **Hindi <-> Bhojpuri**.
4.  **Low-Resource Optimization**: Fine-tuning NLLB-200 for **Hindi <-> Magahi** using LoRA and synthetic data.

---

## 1. Preliminary Work: Hi-En & Custom Vocabulary

**Notebooks:** `hindi_to_eng.ipynb`, `vocab_bpe_bhoj.ipynb`

Before tackling low-resource languages, we established a baseline training pipeline and addressed vocabulary gaps for Bhojpuri (which is not natively supported by IndicTrans).

* **Hindi -> English Fine-tuning**:
    * We fine-tuned the [IndicTrans](https://github.com/AI4Bharat/indicTrans) model (based on `fairseq`) on the **CVIT-PIB** corpus (part of WAT2021).
    * This step verified the `fairseq` training pipeline, including preprocessing (normalization, script conversion), BPE application, and model optimization.
* **Custom Bhojpuri Tokenizer**:
    * The native IndicTrans vocabulary was insufficient for Bhojpuri.
    * We trained a new Byte Pair Encoding (BPE) tokenizer on monolingual Bhojpuri data (`monoloresmt-2020.bho`) using `subword-nmt` (specifically `learn-bpe` with 32k operations) to generate a dedicated vocabulary (`vocab.bho`).

---

## 2. Hindi -> Bhojpuri (IndicTrans)

**Notebook:** `final_hindi_to_bhoj.ipynb`

This stage utilized the pipeline and custom tokenizer developed in the previous step to fine-tune IndicTrans for a language direction it was not originally trained for.

### Approach
* **Model**: `indicTrans` (En-Indic/Indic-Indic checkpoint).
* **Data Preparation**:
    * Data was preprocessed using `IndicNLP` for normalization and transliteration to Devanagari.
    * Applied the **custom BPE codes** generated in Stage 1.
    * Added source/target tags (e.g., `__src__hi__`, `__tgt__bho__`).
* **Training**:
    * Framework: `fairseq`
    * Architecture: `transformer_4x`
    * Loss: Label Smoothed Cross Entropy.

---

## 3. Hindi <-> Bhojpuri (NLLB)

**Notebook:** `finetuningnllbBhojpuri.ipynb`

We shifted to the Meta [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M) model to leverage its massive multilingual pre-training capabilities.

### Approach
* **Framework**: Hugging Face `transformers`
* **Optimization**: Used `Adafactor` optimizer with gradient accumulation and checkpointing to manage memory.
* **Results**:
    * **Baseline BLEU (Zero-shot)**: 7.63
    * **Fine-tuned BLEU**: **26.31**

---

## 4. Hindi <-> Magahi (Low-Resource NLLB with LoRA)

**Notebook:** `finetuning_magahi_low_corpus.ipynb`

The final stage addressed **Magahi**, a language with extremely limited parallel data. To overcome data scarcity and prevent overfitting, we employed a combination of synthetic data generation and parameter-efficient fine-tuning.

### Data Augmentation Strategy
Since high-quality parallel Hindi-Magahi pairs were scarce (~800 gold pairs), we augmented the dataset using **Back-Translation**:
1.  **Monolingual Data**: We utilized ~15,000 lines each of monolingual Magahi and Hindi text.
2.  **Synthetic Generation**:
    * **Mag -> Hi**: We used the base NLLB model to translate monolingual *Magahi* sentences into *Hindi*.
    * **Hi -> Mag**: We translated monolingual *Hindi* sentences into *Magahi*.
3.  **Filtering**: Generated pairs were filtered based on length ratios (0.5 to 2.0) to remove poor-quality translations, retaining approximately 4,000-5,000 synthetic pairs per direction.
4.  **Upsampling**: The original "gold" parallel data was upsampled **3x** to ensure the model prioritized high-quality human translations during training.

### Model Architecture (LoRA)
We used **Low-Rank Adaptation (LoRA)** to fine-tune the model efficiently on a single GPU.
* **Base Model**: `facebook/nllb-200-distilled-600M` loaded in **8-bit precision** (via `bitsandbytes`) to reduce memory footprint.
* **LoRA Config**:
    * **Rank (r)**: 16
    * **Alpha**: 32
    * **Dropout**: 0.05
    * **Target Modules**: `q_proj` (Query Projection) and `v_proj` (Value Projection).

### Training & Results
* **Hyperparameters**: Trained for **4 epochs** with a learning rate of `1e-4`.
* **Performance**: The model demonstrated strong performance on the held-out test set, significantly outperforming baselines.

| Metric | Score |
| :--- | :--- |
| **sacreBLEU** | **37.24** |
| **chrf** | **65.10** |
| **Test Loss** | 1.268 |

---

## Requirements

* `fairseq` (for IndicTrans)
* `transformers`, `peft`, `bitsandbytes`, `accelerate` (for NLLB/LoRA)
* `subword-nmt` (for BPE tokenization)
* `indic-nlp-library`
* `sacrebleu`, `evaluate`
