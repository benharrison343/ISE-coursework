########## 1. Import required libraries ##########

import pandas as pd
import numpy as np
import re
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########## 2. Define text preprocessing methods ##########

def remove_html(text):
    """Remove HTML tags using regex."""
    return re.sub(r'<.*?>', '', text)

def remove_stopwords(text):
    """Remove stopwords from text."""
    stop_words = set(stopwords.words('english'))
    return " ".join([word for word in text.split() if word not in stop_words])

def clean_text(text):
    """Basic text cleaning."""
    text = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", text)
    return text.strip().lower()

########## 3. Load Data ##########

# Choose the project (options: 'pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe')
project = 'caffe'
path = f'datasets/{project}.csv'

df = pd.read_csv(path)
df = df.sample(frac=1, random_state=999)  # Shuffle data

# Merge title and body
df['text'] = df.apply(lambda row: row['Title'] + ' ' + row['Body'] if pd.notna(row['Body']) else row['Title'], axis=1)
df = df[['text', 'class']]  # Keep only text and label
df['text'] = df['text'].apply(lambda x: remove_html(x))
df['text'] = df['text'].apply(lambda x: remove_stopwords(x))
df['text'] = df['text'].apply(lambda x: clean_text(x))

########## 4. Extract BERT Embeddings ##########

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

def get_bert_embeddings(text):
    """Convert text into BERT embeddings using the [CLS] token."""
    tokens = tokenizer(text, padding='max_length', max_length=256, truncation=True, return_tensors="pt")
    tokens = {key: val.to(device) for key, val in tokens.items()}
    
    with torch.no_grad():
        output = model(**tokens)
    
    return output.last_hidden_state[:, 0, :].cpu().numpy().flatten()  # Extract [CLS] token representation

# Compute embeddings for all texts
df['embeddings'] = df['text'].apply(get_bert_embeddings)

########## 5. Train-Test Split ##########

X = np.vstack(df['embeddings'].values)  # Convert list of embeddings to numpy array
y = df['class'].values

# ========== Key Configurations ==========

REPEAT = 10  # Number of repeated experiments
out_csv_name = f'../{project}_BERT_LogReg_results.csv'

# Lists to store metrics across repeated runs
accuracies  = []
precisions  = []
recalls     = []
f1_scores   = []

for repeated_time in range(REPEAT):
    # --- 5.1 Train-Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=repeated_time)

    ########## 6. Train Logistic Regression ##########
    
    # Hyperparameter tuning using GridSearchCV
    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}  # Regularization parameter
    
    clf = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='roc_auc')
    clf.fit(X_train, y_train)

    # Best model
    best_clf = clf.best_estimator_
    best_clf.fit(X_train, y_train)

    ########## 7. Evaluate Model ##########

    y_pred = best_clf.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    # Precision
    precision = precision_score(y_test, y_pred)
    precisions.append(precision)

    # Recall
    recall = recall_score(y_test, y_pred)
    recalls.append(recall)

    # F1 Score
    f1 = f1_score(y_test, y_pred)
    f1_scores.append(f1)

# --- 8. Aggregate results ---
final_accuracy  = np.mean(accuracies)
final_precision = np.mean(precisions)
final_recall    = np.mean(recalls)
final_f1        = np.mean(f1_scores)

print("=== BERT + Logistic Regression Results ===")
print(f"Number of repeats:     {REPEAT}")
print(f"Average Accuracy:      {final_accuracy:.4f}")
print(f"Average Precision:     {final_precision:.4f}")
print(f"Average Recall:        {final_recall:.4f}")
print(f"Average F1 Score:      {final_f1:.4f}")

# Save final results to CSV (append mode)
try:
    # Attempt to check if the file already has a header
    existing_data = pd.read_csv(out_csv_name, nrows=1)
    header_needed = False
except:
    header_needed = True

df_log = pd.DataFrame(
    {
        'repeated_times': [REPEAT],
        'Accuracy': [final_accuracy],
        'Precision': [final_precision],
        'Recall': [final_recall],
        'F1': [final_f1],
    }
)

df_log.to_csv(out_csv_name, mode='a', header=header_needed, index=False)

print(f"\nResults have been saved to: {out_csv_name}")
