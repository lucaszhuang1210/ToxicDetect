import os
import numpy as np
import pandas as pd
import re
import torch
from sentence_transformers import SentenceTransformer

# Define paths for training and test data
train_file = os.path.join('.', 'data', 'train.csv')
test_file = os.path.join('.', 'data', 'test.csv')

# Check if files exist
if not os.path.exists(train_file):
    raise FileNotFoundError(f"Training file not found: {train_file}.")
if not os.path.exists(test_file):
    raise FileNotFoundError(f"Test file not found: {test_file}.")

# Load the data
df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)

# Define a basic text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# Clean the comments in both datasets
df_train['clean_comment'] = df_train['comment_text'].apply(clean_text)
df_test['clean_comment'] = df_test['comment_text'].apply(clean_text)

# Set device (GPU if available, otherwise CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Initialize the Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Generate embeddings for training and test data
embeddings_train = model.encode(df_train['clean_comment'].tolist(), show_progress_bar=True)
embeddings_test = model.encode(df_test['clean_comment'].tolist(), show_progress_bar=True)

# Save the embeddings to separate NumPy files
np.save('embeddings_train.npy', embeddings_train)
np.save('embeddings_test.npy', embeddings_test)
print("Embeddings saved to 'embeddings_train.npy' and 'embeddings_test.npy'")