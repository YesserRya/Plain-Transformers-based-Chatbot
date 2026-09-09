"""Train and load the original encoder-decoder architecture with paired assets."""
import argparse
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from transformer import transformer, loss_function, MAX_LENGTH

CONFIG = dict(num_layers=2, units=512, d_model=256, num_heads=8, dropout=0.1)

def clean(text):
    table = str.maketrans("ąćęłńóśżź", "acelnoszz")
    text = re.sub("[^A-Za-z0-9 ]+", "", str(text).lower().translate(table))
    return re.sub(r"(.)\1{2,}", r"\1", text)

def build(vocab_size):
    return transformer(vocab_size=vocab_size + 2, **CONFIG)

def train(csv_path, destination, epochs=10):
    tf.keras.utils.set_random_seed(42)
    frame = pd.read_csv(csv_path).dropna(subset=["question", "answer"])
    pairs = [(clean(q), clean(a)) for q, a in zip(frame.question, frame.answer)]
    pairs = list(dict.fromkeys((q, a) for q, a in pairs if q and a))
    if len(pairs) < 10:
        raise ValueError("Provide at least 10 distinct nonempty question-answer pairs.")
    rng = np.random.default_rng(42)
    rng.shuffle(pairs)
    split = int(len(pairs) * 0.9)
    fit, validation = pairs[:split], pairs[split:]
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        [text for pair in fit for text in pair], target_vocab_size=8192)
    start, end = tokenizer.vocab_size, tokenizer.vocab_size + 1

    def encode(rows):
        encoded = []
        for q, a in rows:
            q, a = [start] + tokenizer.encode(q) + [end], [start] + tokenizer.encode(a) + [end]
            if max(len(q), len(a)) <= MAX_LENGTH:
                encoded.append((q, a))
        if not encoded:
            raise ValueError("No pairs fit the maximum token length in a partition.")
        q = tf.keras.utils.pad_sequences([p[0] for p in encoded], maxlen=MAX_LENGTH, padding="post")
        a = tf.keras.utils.pad_sequences([p[1] for p in encoded], maxlen=MAX_LENGTH, padding="post")
        return [q, a[:, :-1]], a[:, 1:]
    x, y = encode(fit)
    xv, yv = encode(validation)
    model = build(tokenizer.vocab_size)
    model.compile(optimizer="adam", loss=loss_function)
    history = model.fit(x, y, validation_data=(xv, yv), epochs=epochs, batch_size=32,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer.save_to_file(str(destination / "tokenizer"))
    model.save_weights(str(destination / "model.weights.h5"))
    (destination / "history.json").write_text(json.dumps(history.history, indent=2))
    (destination / "config.json").write_text(json.dumps(CONFIG, indent=2))

def load(directory):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(str(directory / "tokenizer"))
    model = transformer(vocab_size=tokenizer.vocab_size + 2, **config)
    model.load_weights(str(directory / "model.weights.h5"))
    return model, tokenizer

def respond(text, model, tokenizer):
    start, end = tokenizer.vocab_size, tokenizer.vocab_size + 1
    tokens = tokenizer.encode(clean(text))
    if not tokens:
        return ""
    if len(tokens) > MAX_LENGTH - 2:
        raise ValueError("Please use a shorter message (at most 98 subword tokens).")
    inputs = tf.constant([[start] + tokens + [end]])
    output = [start]
    for _ in range(MAX_LENGTH - 1):
        logits = model([inputs, tf.constant([output])], training=False)[0, -1].numpy()
        logits[0] = -np.inf
        logits[start] = -np.inf
        token = int(np.argmax(logits))
        if token == end:
            break
        output.append(token)
    return tokenizer.decode(output[1:])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    train(args.data, args.output, args.epochs)
