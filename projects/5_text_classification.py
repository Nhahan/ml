from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    raise ValueError("HF_TOKEN 환경 변수가 설정되지 않았습니다.")

# Hugging Face 로그인
login(token=HF_TOKEN, add_to_git_credential=True)

# 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto")


# Zero-Shot Classification 함수 정의
def zero_shot_classification(text, task_description, labels):
    text_ids = tokenizer(task_description + text, return_tensors="pt").to("cuda")
    probs = []
    for label in labels:
        label_ids = tokenizer(label, return_tensors="pt").to("cuda")
        n_label_tokens = label_ids['input_ids'].shape[-1] - 1
        input_ids = {
            'input_ids': torch.cat([text_ids['input_ids'], label_ids['input_ids'][:, 1:]], axis=-1),
            'attention_mask': torch.cat([text_ids['attention_mask'], label_ids['attention_mask'][:, 1:]], axis=-1)
        }

        logits = model(**input_ids).logits
        prob = 0
        n_total = input_ids['input_ids'].shape[-1]
        for i in range(n_label_tokens, 0, -1):
            token = label_ids['input_ids'][0, i].item()
            prob += logits[0, n_total - i, token].item()
        probs.append(prob)

        del input_ids
        del logits
        torch.cuda.empty_cache()

    return probs


# ag_news 데이터셋 로드 및 레이블 추출
ag_news = load_dataset("fancyzhx/ag_news")
label_names = ag_news['train'].features['label'].names
print("레이블 이름:", label_names)  # ['World', 'Sports', 'Business', 'Sci/Tech']

# 결과 저장 디렉토리 및 파일명 설정
os.makedirs('./result/5_basic', exist_ok=True)
current_file_name = '5_text_classification'  # 현재 파일명이 '5_text_classification.py'로 가정
log_file_path = f'./result/5_basic/{current_file_name}.txt'

# 정확도 계산
n_corrects = 0
total = 50

with open(log_file_path, 'w', encoding='utf-8') as log_file:
    for i in tqdm(range(50), desc="예측 중"):
        text = ag_news['test'][i]['text']
        label = ag_news['test'][i]['label']
        label_text = label_names[label]

        task_description = "이 문장은 어떤 카테고리에 속합니까? "
        labels = label_names

        probs = zero_shot_classification(text, task_description, labels)
        predicted_label = labels[probs.index(max(probs))]

        is_correct = (predicted_label == label_text)
        if is_correct:
            n_corrects += 1

        # 로그 기록
        log_file.write(f"Sample {i + 1}:\n")
        log_file.write(f"Text: {text}\n")
        log_file.write(f"True Label: {label_text}\n")
        log_file.write(f"Predicted Label: {predicted_label}\n")
        log_file.write(f"Probabilities: {probs}\n")
        log_file.write(f"Correct: {is_correct}\n\n")

accuracy = n_corrects / total * 100
print(f"정확도: {accuracy:.2f}%")

# 정확도 로그에 기록
with open(log_file_path, 'a', encoding='utf-8') as log_file:
    log_file.write(f"총 정확도: {accuracy:.2f}%\n")

# 정확도 그래프 생성 및 저장
plt.figure(figsize=(6, 4))
categories = ['Correct', 'Incorrect']
counts = [n_corrects, total - n_corrects]
colors = ['green', 'red']

plt.bar(categories, counts, color=colors)
plt.title('Zero-Shot Classification 정확도')
plt.ylabel('샘플 수')

plt.savefig(f'./result/5_basic/{current_file_name}.png')
plt.show()
