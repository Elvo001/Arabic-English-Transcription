# 🧠 Arabic-English ASR Fine-Tuning with Wav2Vec2 + PEFT

This notebook demonstrates how to fine-tune a Wav2Vec2 transformer-based model for automatic speech recognition (ASR) using Hugging Face Transformers. It focuses on multilingual (Arabic-English) speech data using the Common Voice dataset and integrates parameter-efficient fine-tuning (PEFT) using LoRA.

---

## 🔧 Features

- Uses `facebook/wav2vec2-large-xlsr-53` as the pretrained model.
- Loads Arabic Common Voice data using the 🤗 `datasets` library.
- Applies LoRA-based PEFT for lightweight fine-tuning.
- Computes Word Error Rate (WER) as evaluation metric.
- Utilizes `Trainer` API for training and evaluation.

---

## 📦 Dependencies

Install necessary packages:

```bash
pip install transformers datasets accelerate peft evaluate
```

---

## 🗂️ Dataset

Common Voice 13.0 Arabic is used:

```python
load_dataset("mozilla-foundation/common_voice_13_0", "ar")
```

---

## 🚀 Training Flow

1. **Load model and processor**
2. **Apply preprocessing to dataset**
3. **Attach PEFT config (LoRA)**
4. **Compute WER via Hugging Face `evaluate`**
5. **Use Hugging Face `Trainer` for training**
6. **Save the fine-tuned model**

---

## 🧪 Metrics

Tracks WER (Word Error Rate):

```python
wer_metric.compute(predictions=pred_str, references=label_str)
```

---

## 💾 Saving

Saves final model and processor to disk using `.save_pretrained()`.

---
