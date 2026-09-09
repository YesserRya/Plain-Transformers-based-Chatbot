"""Exercise training, paired-asset reload and decoding with a tiny synthetic corpus."""
import sys
from pathlib import Path
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import tensorflow as tf
import train
from transformer import create_look_ahead_mask

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    pd.DataFrame({"question": [f"question number {i}" for i in range(30)],
                  "answer": [f"answer number {i}" for i in range(30)]}).to_csv(root / "pairs.csv", index=False)
    train.CONFIG.update(num_layers=1, units=32, d_model=16, num_heads=2)
    train.train(root / "pairs.csv", root / "model", epochs=1)
    model, tokenizer = train.load(root / "model")
    assert isinstance(train.respond("question number 1", model, tokenizer), str)
    mask = create_look_ahead_mask(tf.constant([[1, 2, 0]])).numpy()
    assert mask[0, 0, 0, 1] == 1
    assert mask[0, 0, 1, 0] == 0
    assert mask[0, 0, 2, 2] == 1
print("Synthetic train/load/decode and attention-mask checks passed.")
