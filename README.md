# Transformer Conversation Model

A TensorFlow encoder-decoder Transformer with explicit multi-head attention, positional encoding, padding masks and causal decoder masks, connected to a Streamlit interface.

## Architecture

| Component | Configuration |
|---|---|
| Encoder / decoder | 2 layers each |
| Embedding width | 256 |
| Attention heads | 8 |
| Feed-forward width | 512 |
| Dropout | 0.1 |
| Maximum sequence length | 100 tokens |

## Run locally

Use Python 3.10 or 3.11:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --data data/Conversation.csv --output artifacts --epochs 10
streamlit run app.py
```

See [conversation input requirements](data/README.md). The UI can start before training and explains which assets are missing.

## Training and inference

The training entry point cleans and deduplicates pairs, creates a seeded 90/10 training/validation split, builds its tokenizer only on training text and uses teacher forcing. Padding tokens are excluded from the loss denominator. Early stopping selects validation loss. Weights, tokenizer, architecture configuration and loss history are saved together. Inference uses greedy decoding and the same text cleaning.

The original code rebuilt a tokenizer and loaded unpublished weights on import using personal filesystem paths. These side effects are removed from the current modules. Originals remain under `legacy/` for reference only.

## Results and limits

No pretrained model, original conversation corpus or measured response-quality results were supplied. Training on your own corpus is required before the chat demo can generate responses. Small-data runs validate mechanics only; they do not establish conversational quality. A validation split is not an independent final test set. No full training or response-quality benchmark was completed in this refresh.

The architecture preserves the original grouped sine/cosine positional encoding. This is an educational implementation rather than a claim of a novel architecture.
