import logging
import os
import sys

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset, load_from_disk
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertModel

code_filename = 'fine_tune_distilbert_ag_news'
result_dir = './result/3_basic/'
os.makedirs(result_dir, exist_ok=True)
log_file = os.path.join(result_dir, f"{code_filename}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.info(f"Code Filename: {code_filename}")
logging.info(f"Log File Location: {log_file}")

dataset_path = './data/ag_news/'

if os.path.exists(dataset_path):
    ds = load_from_disk(dataset_path)
else:
    ds = load_dataset("fancyzhx/ag_news")
    ds.save_to_disk(dataset_path)

def get_directory_size_mb(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for fpath in filenames:
            fp = os.path.join(dirpath, fpath)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)

dataset_size_mb = get_directory_size_mb(dataset_path)
logging.info(f" - Dataset size: {dataset_size_mb:.2f} MB\n")

tokenizer_path = './tokenizer/distilbert-base-uncased/'

if os.path.exists(tokenizer_path):
    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
else:
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    tokenizer.save_pretrained(tokenizer_path)

tokenizer_size_mb = get_directory_size_mb(tokenizer_path)
logging.info(f" - Tokenizer size: {tokenizer_size_mb:.2f} MB\n")

def collate_fn(batch):
    max_len = 400  # Adjust if necessary
    texts, labels = [], []
    for row in batch:
        labels.append(row['label'])
        texts.append(row['text'])

    encoding = tokenizer(texts, padding=True, truncation=False, max_length=max_len, return_tensors='pt')
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    labels = torch.LongTensor(labels)

    return input_ids, attention_mask, labels

train_loader = DataLoader(
    ds['train'], batch_size=64, shuffle=True, collate_fn=collate_fn
)
test_loader = DataLoader(
    ds['test'], batch_size=64, shuffle=False, collate_fn=collate_fn
)

class TextClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(TextClassifier, self).__init__()
        self.encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]  # [CLS] token output
        logits = self.classifier(cls_output)
        return logits

model = TextClassifier(num_classes=4)

for param in model.encoder.parameters():
    param.requires_grad = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"\nUsing device: {device}")
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.classifier.parameters(), lr=0.001)

n_epochs = 10

def accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in dataloader:
            input_ids, attention_mask, labels = data
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

train_losses = []

# Initialize variables
train_acc = 0.0
test_acc = 0.0

for epoch in range(n_epochs):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}", leave=False)
    for batch in progress_bar:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    logging.info(f"Epoch {epoch + 1}/{n_epochs} | Train Loss: {avg_loss:.4f}")

    # Calculate and log accuracy
    train_acc = accuracy(model, train_loader)
    test_acc = accuracy(model, test_loader)
    logging.info(f"=========> Train Accuracy: {train_acc:.3f} | Test Accuracy: {test_acc:.3f}\n")

# Final accuracy output
logging.info(f"=========> Final Train Accuracy: {train_acc:.3f} | Final Test Accuracy: {test_acc:.3f}\n")

# Plot loss graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_epochs + 1), train_losses, marker='o', label='Train Loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Save plot
plot_file = os.path.join(result_dir, f"{code_filename}.png")
plt.savefig(plot_file)
plt.close()
logging.info(f"Loss plot saved to {plot_file}")

# Save model (optional)
model_save_path = os.path.join(result_dir, f"{code_filename}_model.pth")
torch.save(model.state_dict(), model_save_path)
logging.info(f"Model saved to {model_save_path}.")

logging.info("Training completed.")
