import logging
import os
from datetime import datetime

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import torch.quantization
from datasets import load_dataset, concatenate_datasets, load_from_disk
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
)

# 현재 파일명과 타임스탬프 생성
current_file_name = os.path.basename(__file__).replace('.py', '')
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 로그 및 결과 저장 디렉토리 설정
log_dir = f"./result/0_"
os.makedirs(log_dir, exist_ok=True)

# 로그 설정
logging.basicConfig(
    filename=os.path.join(log_dir, f'training_log_{current_file_name}_{timestamp}.txt'),
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# GPU 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
    logging.info(f"Using device: {device} (GPU: {gpu_name}, Memory: {gpu_mem} GB)")
else:
    logging.info(f"Using device: {device}")

# 토크나이저 및 모델 로드
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.to(device)

# 모델 설정 조정
model.config.max_length = 128
model.config.no_repeat_ngram_size = 3


# 컬럼 이름 확인 및 전처리
def preprocess_dataset(dataset):
    possible_columns = [
        ('en', 'ko'),
        ('eng', 'kor'),
        ('english', 'korean'),
    ]
    for src_col, tgt_col in possible_columns:
        if src_col in dataset.column_names and tgt_col in dataset.column_names:
            return dataset.rename_columns({src_col: 'en', tgt_col: 'ko'})
    raise ValueError("No matching column names found for source and target languages.")


# 데이터셋 로드
korean_parallel_corpora = load_dataset("Moo/korean-parallel-corpora")
aihub_koen_dataset = load_dataset("traintogpb/aihub-koen-translation-integrated-large-10m")


# 랜덤하게 데이터셋을 90% 학습용, 10% 검증용으로 분할
def random_split(dataset, train_split=0.9):
    return dataset.train_test_split(test_size=1 - train_split)


# 데이터셋에 train/validation이 존재하는지 확인 후 처리
def split_or_use_existing(dataset, split_ratio=0.9):
    if "train" in dataset and "validation" in dataset:
        # 이미 train과 validation이 있는 경우 그대로 사용
        return dataset["train"], dataset["validation"]
    else:
        # 없는 경우 랜덤 분할
        split_dataset = random_split(dataset["train"], train_split=split_ratio)
        return split_dataset["train"], split_dataset["test"]


# 각 데이터셋에 대해 분할 또는 기존 데이터 사용
korean_parallel_corpora_train, korean_parallel_corpora_validation = split_or_use_existing(korean_parallel_corpora)
aihub_koen_dataset_train, aihub_koen_dataset_validation = split_or_use_existing(aihub_koen_dataset)

# 컬럼 이름 맞추기 및 병합
train_datasets = [
    preprocess_dataset(korean_parallel_corpora_train),
    preprocess_dataset(aihub_koen_dataset_train),
]

eval_datasets = [
    preprocess_dataset(korean_parallel_corpora_validation),
    preprocess_dataset(aihub_koen_dataset_validation),
]

train_dataset = concatenate_datasets(train_datasets)
eval_dataset = concatenate_datasets(eval_datasets)

# 데이터셋 크기 제한 (예: 1,000,000 예제로 제한)
max_train_samples = 1_000_000
max_eval_samples = 100_000

train_dataset = train_dataset.shuffle(seed=42).select(range(min(len(train_dataset), max_train_samples)))
eval_dataset = eval_dataset.shuffle(seed=42).select(range(min(len(eval_dataset), max_eval_samples)))


# None 값 또는 빈 문자열이 포함된 데이터 제거 함수
def remove_empty_examples(dataset, input_col, target_col):
    return dataset.filter(lambda x: x[input_col] and x[target_col] and x[input_col].strip() and x[target_col].strip())


# 훈련 및 검증 데이터셋에서 빈 값이 있는 예제 제거
train_dataset = remove_empty_examples(train_dataset, 'en', 'ko')
eval_dataset = remove_empty_examples(eval_dataset, 'en', 'ko')


# 데이터 전처리 함수
def preprocess_function(examples):
    inputs = examples['en']
    targets = examples['ko']

    # 입력과 레이블을 토큰화하고 인코딩
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# 데이터셋 저장 경로 설정
processed_train_dataset_path = os.path.join(log_dir, "tokenized_train_dataset")
processed_eval_dataset_path = os.path.join(log_dir, "tokenized_eval_dataset")

# 전처리된 데이터셋이 이미 저장되어 있는지 확인하고, 있으면 불러오기
if os.path.exists(processed_train_dataset_path) and os.path.exists(processed_eval_dataset_path):
    logging.info("Loading preprocessed datasets from disk...")
    tokenized_train_dataset = load_from_disk(processed_train_dataset_path)
    tokenized_eval_dataset = load_from_disk(processed_eval_dataset_path)
else:
    logging.info("Preprocessing datasets...")
    tokenized_train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=4,
        desc="Tokenizing training data",
    )
    tokenized_eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        num_proc=4,
        desc="Tokenizing evaluation data",
    )

    tokenized_train_dataset.save_to_disk(processed_train_dataset_path)
    tokenized_eval_dataset.save_to_disk(processed_eval_dataset_path)

# 평가 메트릭 설정
bleu_metric = evaluate.load("sacrebleu")
meteor_metric = evaluate.load("meteor")


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_labels_bleu = [[label] for label in decoded_labels]
    bleu_result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels_bleu)
    meteor_result = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)

    return {
        "bleu": bleu_result["score"],
        "meteor": meteor_result["meteor"]
    }


# 데이터 콜레이터 설정
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding='max_length',
    max_length=128,
    pad_to_multiple_of=None,
    return_tensors="pt"
)

# 조기 종료 콜백 설정
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=4)

# 체크포인트 경로 확인 및 설정
checkpoint_dir = None
if os.path.exists(os.path.join(log_dir, "checkpoint")):
    checkpoint_dir = os.path.join(log_dir, "checkpoint")

# 훈련 인자 설정
training_args = Seq2SeqTrainingArguments(
    output_dir=log_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_steps=500,
    logging_strategy="steps",
    logging_steps=500,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=30,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=5,
    predict_with_generate=True,
    generation_max_length=128,
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    lr_scheduler_type="cosine",
)


# Custom Seq2SeqTrainer 클래스 정의 (로그 기록용)
class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def log(self, logs):
        super().log(logs)  # 기본 로그 처리
        # 로그 파일에 기록
        with open(os.path.join(log_dir, f'training_log_{current_file_name}_{timestamp}.txt'), 'a') as f:
            f.write(f"{logs}\n")


# 트레이너 초기화
trainer = CustomSeq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback],
)

# 모델 훈련 (체크포인트 있을 시, 체크포인트에서 학습 재개)
trainer.train(resume_from_checkpoint=checkpoint_dir)

# 모델 양자화(Quantization) 경량화
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
quantized_model.save_pretrained(os.path.join(log_dir, "quantized_model"))
tokenizer.save_pretrained(os.path.join(log_dir, "quantized_model"))


# 학습 및 평가 손실 그래프 그리기
def plot_loss(log_history, save_path):
    train_loss = [log["loss"] for log in log_history if "loss" in log]
    eval_loss = [log["eval_loss"] for log in log_history if "eval_loss" in log]
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs[:len(eval_loss)], eval_loss, label='Evaluation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training & Evaluation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


# 손실 그래프 저장
plot_loss(trainer.state.log_history, os.path.join(log_dir, "loss_plot.png"))
