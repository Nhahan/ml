import torch
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer

# 양자화된 모델과 토크나이저 로드 (체크포인트 경로는 모델이 저장된 디렉토리 경로로 설정)
quantized_model_dir = "./result/0_/checkpoint-301051"
tokenizer = T5Tokenizer.from_pretrained(quantized_model_dir)
model = T5ForConditionalGeneration.from_pretrained(quantized_model_dir)

# 모델을 CPU로 이동 (양자화된 모델은 CPU에서 효율적으로 작동)
device = torch.device("cpu")
model.to(device)

# 모델의 용량을 출력
model_size = sum(os.path.getsize(os.path.join(quantized_model_dir, f)) for f in os.listdir(quantized_model_dir))
print(f"모델의 용량: {model_size / (1024 * 1024):.2f} MB")


# 번역 함수
def translate_text(sentence, src_lang='ko', tgt_lang='en'):
    if src_lang == 'ko' and tgt_lang == 'en':
        # 한-영 번역
        prefix = "translate Korean to English: "
    elif src_lang == 'en' and tgt_lang == 'ko':
        # 영-한 번역
        prefix = "translate English to Korean: "
    else:
        raise ValueError("지원하지 않는 번역 방향입니다.")

    # 번역할 문장에 prefix 추가
    sentence = prefix + sentence

    # 모델 입력을 위한 전처리 (tokenization)
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)

    # 모델을 사용하여 번역 생성 (CPU에서 실행)
    with torch.no_grad():
        generated_ids = model.generate(input_ids=inputs["input_ids"], max_length=128, num_beams=4, early_stopping=True)

    # 생성된 번역문을 디코딩
    translated_sentence = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return translated_sentence


# 번역할 한국어 및 영어 문장
korean_sentence = "안녕하세요. 반갑습니다."
english_sentence = "Hello. Nice to meet you."

# 한-영 번역 시도
translated_en = translate_text(korean_sentence, src_lang='ko', tgt_lang='en')
print(f"한국어 문장: {korean_sentence}")
print(f"번역된 영어 문장: {translated_en}")

# 영-한 번역 시도
translated_ko = translate_text(english_sentence, src_lang='en', tgt_lang='ko')
print(f"영어 문장: {english_sentence}")
print(f"번역된 한국어 문장: {translated_ko}")
