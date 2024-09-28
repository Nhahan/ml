import logging
import os
from datetime import datetime

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    BertTokenizerFast,
    EncoderDecoderModel,
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

# 토크나이저 로드
tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")

# pad_token 설정 확인 및 설정
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 모델 로드 및 설정
model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    "bert-base-multilingual-cased",
    "bert-base-multilingual-cased"
)

# 모델 설정 조정
model.config.pad_token_id = tokenizer.pad_token_id
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.max_length = 128
model.config.no_repeat_ngram_size = 3

# 모델 크기 감소를 위한 설정
model.config.hidden_size = 256
model.config.num_hidden_layers = 4
model.config.num_attention_heads = 4
model.config.encoder_ffn_dim = 512
model.config.decoder_ffn_dim = 512

model.to(device)

# 데이터셋 로드
korean_parallel_corpora = load_dataset("Moo/korean-parallel-corpora")
ted_talks_dataset = load_dataset("msarmi9/korean-english-multitarget-ted-talks-task")

# 데이터셋 병합
train_dataset = concatenate_datasets([
    korean_parallel_corpora["train"],
    ted_talks_dataset["train"],
])
eval_dataset = concatenate_datasets([
    korean_parallel_corpora["test"],
    ted_talks_dataset["validation"],
])


# 데이터 전처리 함수 정의
def preprocess_function(examples):
    if 'ko' in examples and 'en' in examples:
        inputs = examples['ko']
        targets = examples['en']
    elif 'korean' in examples and 'english' in examples:
        inputs = examples['korean']
        targets = examples['english']
    else:
        raise ValueError("Unexpected column names in dataset!")

    # inputs와 targets이 리스트 형식인지 확인하고, 그렇지 않으면 리스트로 변환
    if isinstance(inputs, str):
        inputs = [inputs]
    if isinstance(targets, str):
        targets = [targets]

    # 빈 값이나 None이 있는 경우 필터링 (None 또는 빈 문자열을 " "로 대체)
    inputs = [input_text if input_text else " " for input_text in inputs]
    targets = [target_text if target_text else " " for target_text in targets]

    # 입력과 레이블을 토큰화하고 인코딩
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# 데이터셋 전처리
tokenized_train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
)
tokenized_eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=eval_dataset.column_names,
)


# 평가 메트릭 설정
bleu_metric = evaluate.load("sacrebleu")
meteor_metric = evaluate.load("meteor")
rouge_metric = evaluate.load("rouge")


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_labels_bleu = [[label] for label in decoded_labels]
    bleu_result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels_bleu)
    meteor_result = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)
    rouge_result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)

    return {
        "bleu": bleu_result["score"],
        "meteor": meteor_result["meteor"],
        "rouge": rouge_result["rougeL"].mid.fmeasure
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
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)

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
    logging_steps=50,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    learning_rate=3e-4,
    save_total_limit=5,
    predict_with_generate=True,
    generation_max_length=128,
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# 트레이너 초기화
trainer = Seq2SeqTrainer(
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

# 훈련 로그 저장
with open(os.path.join(log_dir, f'training_log_{current_file_name}_{timestamp}.txt'), 'a') as f:
    for log in trainer.state.log_history:
        f.write(f"{log}\n")

# 모델 저장
trainer.save_model(os.path.join(log_dir, "trained_model"))
tokenizer.save_pretrained(os.path.join(log_dir, "trained_model"))


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


# 그래프 저장
plot_loss(trainer.state.log_history, os.path.join(log_dir, f"loss_plot_{current_file_name}_{timestamp}.png"))
