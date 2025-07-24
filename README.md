# Arabic-English-Transcription
# Arabic-English ASR Fine-Tuning with Wav2Vec2

This demo fine-tunes a multilingual Wav2Vec2 model (`facebook/wav2vec2-large-xlsr-53`) on the Common Voice dataset for Arabic-English automatic speech recognition (ASR) using Hugging Face Transformers.

## 🚀 Highlights
- Supports Arabic (`ar`) and English (`en`) Common Voice data
- Uses `Trainer` API with `Wav2Vec2CTC` for fast iteration
- Tracks WER (Word Error Rate) to measure transcription quality
- Leverages FP16 mixed-precision training (when available)

## 📦 Requirements
```bash
pip install transformers datasets evaluate torchaudio
```

## 🧠 Training
```bash
python train.py
```

## 🔍 Results
Evaluated per epoch with WER:
```
{'wer': 0.145}  # Example
```

## 📝 Notes
- Use `split='train[:1%]'` for quick testing.
- Hugging Face token recommended for full dataset access.

