# 필요한 라이브러리 임포트
import torch
from transformers import (
    BertTokenizerFast,
    EncoderDecoderModel,
    Seq2SeqTrainer,                # 수정된 부분
    Seq2SeqTrainingArguments,      # 수정된 부분
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset, concatenate_datasets
import evaluate                    # 수정된 부분
import logging
import numpy as np

# 로그 설정
logging.basicConfig(
    filename='training_log.txt',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# GPU 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# 토크나이저 로드
tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")

# 모델 로드 및 설정
model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    "bert-base-multilingual-cased",
    "bert-base-multilingual-cased"
)

# 모델 설정 조정
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
opus_books = load_dataset("opus_books", "ko-en")
open_subtitles = load_dataset("open_subtitles", "ko-en")

# 데이터셋 병합
train_dataset = concatenate_datasets([opus_books["train"], open_subtitles["train"]])
eval_dataset = concatenate_datasets([opus_books["validation"], open_subtitles["validation"]])

# 데이터 전처리 함수 정의
def preprocess_function(examples):
    inputs = examples["translation"]["ko"] + examples["translation"]["en"]
    targets = examples["translation"]["en"] + examples["translation"]["ko"]

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
metric = evaluate.load("sacrebleu")  # 수정된 부분

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_labels = [[label] for label in decoded_labels]
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

# 데이터 콜레이터 설정
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding='max_length', max_length=128)

# 조기 종료 콜백 설정
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)

# 훈련 인자 설정
training_args = Seq2SeqTrainingArguments(    # 수정된 부분
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=30,
    learning_rate=3e-4,
    save_total_limit=3,
    predict_with_generate=True,              # 수정된 부분
    generation_max_length=128,               # 수정된 부분
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# 트레이너 초기화
trainer = Seq2SeqTrainer(                    # 수정된 부분
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback],
)

# 모델 훈련
trainer.train()

# 훈련 로그 저장
with open('training_log.txt', 'a') as f:
    for log in trainer.state.log_history:
        f.write(f"{log}\n")

# 모델 저장
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")
