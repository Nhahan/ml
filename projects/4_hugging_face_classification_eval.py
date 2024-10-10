import logging
import os
import sys

import evaluate
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# 로그 설정
directory = "./result/4_basic"
filename = os.path.splitext(os.path.basename(__file__))[0]
log_path = f"{directory}/{filename}_evaluation.txt"

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

# 평가 함수
logger.info("Setting up evaluation metric...")
eval_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return eval_metric.compute(predictions=predictions, references=labels)


# 체크포인트 목록
checkpoints = ["checkpoint-3000", "checkpoint-6000", "checkpoint-9000", "checkpoint-12000"]
checkpoint_paths = [os.path.join("hf_transformer_ag_news", cp) for cp in checkpoints]

# 체크포인트별 지표 저장을 위한 리스트
eval_accuracies = []
eval_losses = []

# 각 체크포인트에서 모델 로드 및 평가
for cp_path in checkpoint_paths:
    logger.info(f"Evaluating model at {cp_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(cp_path)

    # Trainer 설정 (평가만 수행)
    training_args = TrainingArguments(
        output_dir=directory,
        per_device_eval_batch_size=32,
        do_train=False,
        do_eval=True,
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=ag_news_val,  # 또는 ag_news_test로 변경 가능
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # 평가 수행
    eval_result = trainer.evaluate()
    eval_accuracies.append(eval_result['eval_accuracy'])
    eval_losses.append(eval_result['eval_loss'])

    logger.info(
        f"Checkpoint {cp_path}: Accuracy = {eval_result['eval_accuracy']:.4f}, Loss = {eval_result['eval_loss']:.4f}")

# 결과 그래프 그리기
logger.info("Plotting evaluation metrics...")
epochs = [3, 6, 9, 12]  # 각 체크포인트에 해당하는 에포크 수 (예시로 3000 스텝당 1 에포크로 가정)
plt.figure(figsize=(10, 5))

# 정확도 그래프
plt.subplot(1, 2, 1)
plt.plot(epochs, eval_accuracies, marker='o')
plt.title('Validation Accuracy per Checkpoint')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(epochs)

# 손실 그래프
plt.subplot(1, 2, 2)
plt.plot(epochs, eval_losses, marker='o', color='red')
plt.title('Validation Loss per Checkpoint')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(epochs)

plt.tight_layout()
plt.savefig(f"{directory}/{filename}_metrics.png")
plt.show()
