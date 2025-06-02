import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, \
    Trainer
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score


# Device detection: Use Apple Silicon MPS if available, otherwise use CUDA or CPU.
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon MPS device for training.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device for training.")
else:
    device = torch.device("cpu")
    print("Using CPU for training.")

# Define the label columns
label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Load the CSV using the datasets library
dataset = load_dataset("csv", data_files = {"train": "../data/train.csv"})["train"]

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")


# Tokenization function: convert text to inputs required by RoBERTa.
def tokenize_function(example):
    return tokenizer(example["comment_text"], truncation = True, padding = "max_length",
                     max_length = 128)


# Tokenize the dataset
dataset = dataset.map(tokenize_function, batched = True)


# Format labels: Combine the six binary columns into a single list of floats stored in "labels".
def format_labels(example):
    example["labels"] = [float(example[col]) for col in label_columns]
    return example


dataset = dataset.map(format_labels)


# Define a function to compute evaluation metrics.
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Convert logits to probabilities using sigmoid and threshold at 0.5.
    preds = torch.sigmoid(torch.tensor(logits))
    preds = (preds >= 0.5).int().numpy()
    labels = labels.astype(int)
    f1 = f1_score(labels, preds, average = "micro")
    accuracy = accuracy_score(labels, preds)
    return {"f1": f1, "accuracy": accuracy}


# Set up 5-fold cross validation using scikit-learn's KFold.
kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
indices = np.arange(len(dataset))
fold = 0
all_metrics = []

for train_index, val_index in kf.split(indices):
    fold += 1
    print(f"\n======== Fold {fold} ========")

    # Create train and validation subsets using the datasets 'select' method.
    train_dataset = dataset.select(train_index.tolist())
    val_dataset = dataset.select(val_index.tolist())

    # Load a fresh RoBERTa model with a classification head for 6 labels.
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base",
                                                               num_labels = len(label_columns))
    # Configure for multi-label classification (uses BCEWithLogitsLoss).
    model.config.problem_type = "multi_label_classification"

    # Move the model to the detected device (e.g., MPS for Apple Silicon).
    model.to(device)

    # Define training arguments.
    training_args = TrainingArguments(
        output_dir = f"./results_fold_{fold}",
        num_train_epochs = 2,  # Adjust as needed.
        per_device_train_batch_size = 8,
        per_device_eval_batch_size = 8,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        logging_steps = 100,
        logging_dir = f"./logs_fold_{fold}",
        load_best_model_at_end = True,
        metric_for_best_model = "eval_loss",
        seed = 42,
    )

    # Initialize the Trainer.
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        compute_metrics = compute_metrics,
    )

    # Train and evaluate on the current fold.
    trainer.train()
    eval_result = trainer.evaluate()
    print(f"Fold {fold} evaluation results:", eval_result)
    all_metrics.append(eval_result)

# Print cross validation results across all folds.
print("\nCross Validation Metrics:")
for i, metrics in enumerate(all_metrics, start = 1):
    print(f"Fold {i}: {metrics}")
