# app.py
import streamlit as st
import torch
import pickle
from model import LSTMPoetryModel

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to the model and vocabulary files (adjust these paths as needed)
model_path = "LSTM_Poetry_Model/poetry_model.pth"
vocab_path = "LSTM_Poetry_Model/vocab.pkl"

# Load the vocabulary
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)
st.write("Vocabulary loaded successfully.")

# Create reverse mapping for generating text
idx_to_word = {idx: word for word, idx in vocab.items()}

vocab_size = len(vocab)

# Instantiate and load the model
model = LSTMPoetryModel(vocab_size).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
st.write("Model loaded successfully.")

# Define sequence length (should match your training configuration)
seq_length = 20

# Function to generate poetry
def generate_poetry(seed_text, next_words=20):
    model.eval()
    words = seed_text.lower().split()
    for _ in range(next_words):
        # Convert the last `seq_length` words to indices (pad if necessary)
        seq = [vocab.get(word, 0) for word in words[-seq_length:]]
        if len(seq) < seq_length:
            seq = [0] * (seq_length - len(seq)) + seq
        seq_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(seq_tensor)
            predicted_idx = torch.argmax(output, dim=1).item()
            predicted_word = idx_to_word.get(predicted_idx, '<UNK>')
        words.append(predicted_word)
    return ' '.join(words)

# Streamlit UI
st.title("Roman Urdu Poetry Generator")

seed_text = st.text_input("Enter seed text:", "dil ki baat")
next_words = st.slider("Number of words to generate:", min_value=5, max_value=50, value=20)

if st.button("Generate Poetry"):
    with st.spinner("Generating poetry..."):
        generated_poetry = generate_poetry(seed_text, next_words)
    st.markdown("### Generated Poetry")
    st.write(generated_poetry)
