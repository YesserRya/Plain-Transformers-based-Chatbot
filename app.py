from pathlib import Path
import streamlit as st

st.title("Transformer Conversation Demo")
st.caption("An educational encoder-decoder model trained on paired conversations.")
directory = st.sidebar.text_input("Model directory", "artifacts")
required = ["config.json", "tokenizer.subwords", "model.weights.h5"]
if not all((Path(directory) / name).is_file() for name in required):
    st.info("Train a model first, then select its output directory.")
    st.code("python train.py --data data/Conversation.csv --output artifacts --epochs 10")
    st.stop()

@st.cache_resource
def assets(folder):
    from train import load
    return load(folder)

try:
    model, tokenizer = assets(directory)
except Exception as exc:
    st.error(f"Could not load model assets: {exc}")
    st.stop()
with st.form("chat"):
    message = st.text_input("Your message")
    submitted = st.form_submit_button("Send")
if submitted and message.strip():
    from train import respond
    try:
        with st.spinner("Generating response"):
            reply = respond(message, model, tokenizer)
        st.write(reply or "The model returned an empty response.")
    except ValueError as exc:
        st.warning(str(exc))
