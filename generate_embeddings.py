import os
import numpy as np
import pandas as pd
import re
import torch
from sentence_transformers import SentenceTransformer

# Path to the training data
data_file = os.path.join('.', 'data', 'train.csv')
if not os.path.exists(data_file):
    raise FileNotFoundError(f"Data file not found: {data_file}. Please ensure it exists in the './data/' directory.")

# Load the data
df = pd.read_csv(data_file)

# Define a basic text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# Clean the comments
df['clean_comment'] = df['comment_text'].apply(clean_text)

# Set device (GPU if available, otherwise CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Initialize the Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Encode the cleaned comments into embeddings
embeddings = model.encode(df['clean_comment'].tolist(), show_progress_bar=True)

# Save the embeddings to a NumPy file
np.save('embeddings.npy', embeddings)
print("Embeddings saved to 'embeddings.npy'")