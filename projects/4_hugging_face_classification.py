import random
import evaluate
import numpy as np
import logging
import sys
import os
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
    EarlyStoppingCallback
)

# 로그 설정
directory = "./result/4_basic"
filename = os.path.splitext(os.path.basename(__file__))[0]
log_path = f"{directory}/{filename}.txt"

if not os.path.exists(directory):
    os.makedirs(directory)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Dataset 준비
logger.info("Loading dataset...")
ag_news = load_dataset("ag_news")

# Tokenizer 준비
logger.info("Preparing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(data):
    return tokenizer(data["text"], truncation=True)

logger.info("Tokenizing dataset...")
ag_news_tokenized = ag_news.map(preprocess_function, batched=True)

# Train, Validation Split
tag_news_split = ag_news_tokenized['train'].train_test_split(test_size=0.2)
ag_news_train, ag_news_val = tag_news_split['train'], tag_news_split['test']
ag_news_test = ag_news_tokenized['test']

# Model 구현
logger.info("Loading model...")
id2label = {0: "WORLD", 1: "SPORTS", 2: "BUSINESS", 3: "SCIENCE"}
label2id = {v: k for k, v in id2label.items()}

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=4,
    id2label=id2label,
    label2id=label2id
)

# 학습 인자 설정
logger.info("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="hf_transformer_ag_news",  # 모델, log 등을 저장할 directory
    num_train_epochs=10,  # epoch 수
    per_device_train_batch_size=32,  # training data의 batch size
    per_device_eval_batch_size=32,  # validation data의 batch size
    logging_strategy="epoch",  # Epoch가 끝날 때마다 training loss 등을 log하라는 의미
    do_train=True,  # 학습을 진행하겠다는 의미
    do_eval=True,  # 학습 중간에 validation data에 대한 평가를 수행하겠다는 의미
    eval_strategy="epoch",  # 매 epoch가 끝날 때마다 validation data에 대한 평가를 수행한다는 의미
    save_strategy="epoch",  # 매 epoch가 끝날 때마다 모델을 저장하겠다는 의미
    learning_rate=1e-4,  # optimizer에 사용할 learning rate
    load_best_model_at_end=True,  # 학습이 끝난 후, validation data에 대한 성능이 가장 좋은 모델을 채택하겠다는 의미
    logging_dir="./logs",  # TensorBoard log directory
    report_to=["tensorboard"],
)

# 평가 함수
eval_metric = evaluate.load("accuracy")

def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=1)
    return eval_metric.compute(predictions=predictions, references=labels)

# Trainer 설정
logger.info("Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ag_news_train,
    eval_dataset=ag_news_val,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# 학습
logger.info("Starting training...")
train_result = trainer.train()
trainer.save_model()

# Training loss 로그 저장
logger.info("Saving training log...")
with open(log_path, "a") as log_file:
    log_file.write(f"\n{train_result}")

# Test data 평가
logger.info("Evaluating on test data...")
test_result = trainer.evaluate(ag_news_test)
logger.info(f"Test Accuracy: {test_result['eval_accuracy']:.4f}")

# 결과 그래프 저장
logger.info("Saving result graph...")
train_loss = train_result.training_losses
plt.plot(train_loss, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.savefig(f"{filename}.png")

# 예시 예측
logger.info("Predicting sample text...")
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)
sample_text = "UK charges 8 in terror plot linked to alert in US LONDON, AUGUST 17: Britain charged eight terror suspects on Tuesday with conspiracy to commit murder and said one had plans that could be used in striking US buildings that were the focus of security scares this month."
prediction = classifier(sample_text)
logger.info(f"Prediction: {prediction}")