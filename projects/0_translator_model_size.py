import os
from transformers import T5ForConditionalGeneration

# 모델 경로 설정
model_name = "t5-small"

# 모델 다운로드 및 로드
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 모델 저장 폴더 경로 확인
model_dir = model.save_pretrained("./t5_small_model")


# 모델 디렉토리 내 파일 크기 계산 함수
def get_model_size(model_dir):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(model_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


# 모델 크기 출력 (MB 단위)
model_size_mb = get_model_size("./t5_small_model") / (1024 ** 2)
print(f"Model size: {model_size_mb:.2f} MB")
