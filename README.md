# IMDb Sentiment Neural Network

Neural network sentiment classifier trained on IMDb reviews.

## Features

- TF-IDF Vectorization
- Feedforward Neural Network
- PyTorch
- GitHub Actions CI/CD
- Automatic Hugging Face Deployment

---

## Dataset

IMDb movie review dataset.

---

## Install

```bash
pip install -r requirements.txt
```

---

## Train Model

```bash
python train.py
```

---

## Predict

```bash
python predict.py
```

---

## CI/CD Pipeline

On every push to `main`:

1. Install dependencies
2. Train model
3. Save artifacts
4. Upload to Hugging Face Hub automatically

---

## Hugging Face

Artifacts deployed automatically using GitHub Actions.
