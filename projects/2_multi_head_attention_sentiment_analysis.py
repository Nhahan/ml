import logging
import os
from math import sqrt
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer

current_filename = os.path.basename(__file__).replace('.py', '')
log_dir = "./result/2_basic/"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, f"{current_filename}.txt")
logging.basicConfig(
    filename=log_file_path,  # 로그 파일 이름 설정
    level=logging.INFO,  # 로그 레벨을 INFO로 설정 (INFO 이상의 레벨만 기록)
    format='%(asctime)s %(levelname)s %(message)s'  # 로그 메시지의 형식 설정
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

if __name__ == "__main__":
    try:
        # GPU 사용 가능 여부에 따라 장치(device) 설정 ('cuda' 또는 'cpu')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # IMDB 영화 리뷰 데이터셋 로드
        ds = load_dataset("imdb")

        # 사전 학습된 BERT 토크나이저 로드
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


        # 데이터셋을 토큰화하는 함수 정의
        def tokenize_function(examples):
            # 텍스트를 토큰화하고 패딩 및 트렁케이션 적용
            return tokenizer(
                examples['text'],  # 입력 텍스트
                padding='max_length',  # 최대 길이까지 패딩
                truncation=True,  # 최대 길이를 초과하는 부분은 잘라냄
                max_length=128  # 최대 시퀀스 길이 설정
            )


        # 데이터셋 전체에 토큰화 함수 적용 (배치 단위로 처리하여 속도 향상)
        tokenized_datasets = ds.map(tokenize_function, batched=True)

        # 데이터셋을 PyTorch 텐서 형식으로 설정하고 필요한 열만 포함
        tokenized_datasets.set_format(type='torch', columns=['input_ids', 'label'])

        # 학습용 DataLoader 생성
        train_loader = DataLoader(
            tokenized_datasets['train'],  # 토큰화된 학습 데이터셋
            batch_size=64,  # 배치 크기 설정
            shuffle=True,  # 매 에포크마다 데이터 섞기
            num_workers=4  # 데이터 로딩을 위한 서브프로세스 수
        )

        # 테스트용 DataLoader 생성
        test_loader = DataLoader(
            tokenized_datasets['test'],  # 토큰화된 테스트 데이터셋
            batch_size=64,  # 배치 크기 설정
            shuffle=False,  # 테스트 데이터는 섞지 않음
            num_workers=4  # 데이터 로딩을 위한 서브프로세스 수
        )


        # 위치 인코딩(Position Encoding) 정의
        def get_angles(pos, i, d_model):
            # 각 위치와 차원에 대한 각도 계산을 위한 함수
            angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
            return pos * angle_rates


        def positional_encoding(position, d_model):
            # 위치 인코딩 행렬 생성
            angle_rads = get_angles(np.arange(position)[:, None], np.arange(d_model)[None, :], d_model)
            # 짝수 인덱스에는 사인 함수를 적용
            angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
            # 홀수 인덱스에는 코사인 함수를 적용
            angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
            # 배치 차원을 추가하여 PyTorch 텐서로 변환
            pos_encoding = angle_rads[None, ...]
            return torch.FloatTensor(pos_encoding)


        # 최대 시퀀스 길이 설정 (토큰화 시의 max_length와 동일)
        max_len = 128


        # 멀티헤드 셀프 어텐션 클래스 정의
        class SelfAttention(nn.Module):
            def __init__(self, input_dim, d_model, n_heads):
                super().__init__()

                # d_model이 헤드 수로 나누어 떨어지는지 확인
                assert d_model % n_heads == 0, "d_model은 n_heads로 나누어 떨어져야 합니다."

                self.input_dim = input_dim
                self.d_model = d_model
                self.n_heads = n_heads
                self.d_head = d_model // n_heads  # 각 헤드의 차원

                # 쿼리, 키, 밸류를 계산하기 위한 선형 레이어 정의
                self.wq = nn.Linear(input_dim, d_model)
                self.wk = nn.Linear(input_dim, d_model)
                self.wv = nn.Linear(input_dim, d_model)
                self.dense = nn.Linear(d_model, d_model)

                # 어텐션 가중치를 계산하기 위한 소프트맥스 함수
                self.softmax = nn.Softmax(dim=-1)

            def forward(self, x, mask):
                B, S, _ = x.shape  # 배치 크기(B), 시퀀스 길이(S)
                H = self.n_heads  # 헤드 수(H)
                D = self.d_model  # 모델 차원(D)
                D_head = self.d_head  # 각 헤드의 차원(D_head)

                # 쿼리, 키, 밸류 계산
                q = self.wq(x)  # (B, S, D)
                k = self.wk(x)  # (B, S, D)
                v = self.wv(x)  # (B, S, D)

                # (B, S, H, D_head) 형태로 변환
                q = q.view(B, S, H, D_head)
                k = k.view(B, S, H, D_head)
                v = v.view(B, S, H, D_head)

                # (B, H, S, D_head) 형태로 변환하여 헤드 차원을 앞으로 이동
                q = q.permute(0, 2, 1, 3)
                k = k.permute(0, 2, 1, 3)
                v = v.permute(0, 2, 1, 3)

                # 스케일된 닷 프로덕트 어텐션 계산
                score = torch.matmul(q, k.transpose(-2, -1))  # (B, H, S, S)
                score = score / sqrt(D_head)

                if mask is not None:
                    # 마스크를 어텐션 스코어에 적용
                    mask = mask[:, None, None, :]  # (B, 1, 1, S)
                    score = score.masked_fill(mask == True, -1e9)

                # 어텐션 가중치 계산
                attention_weights = self.softmax(score)  # (B, H, S, S)

                # 컨텍스트 벡터 계산
                context = torch.matmul(attention_weights, v)  # (B, H, S, D_head)

                # 원래 형태로 변환하고 이어붙이기
                context = context.permute(0, 2, 1, 3).contiguous()  # (B, S, H, D_head)
                context = context.view(B, S, D)  # (B, S, D)

                # 최종 선형 레이어 통과
                result = self.dense(context)  # (B, S, D)

                return result


        # Transformer 레이어 구현
        class TransformerLayer(nn.Module):
            def __init__(self, input_dim, d_model, dff, n_heads, dropout_rate=0.1):
                super().__init__()

                self.input_dim = input_dim
                self.d_model = d_model
                self.dff = dff
                self.n_heads = n_heads

                # 셀프 어텐션 레이어
                self.sa = SelfAttention(input_dim, d_model, n_heads)
                # 피드포워드 네트워크
                self.ffn = nn.Sequential(
                    nn.Linear(d_model, dff),
                    nn.ReLU(),
                    nn.Linear(dff, d_model)
                )

                # 드롭아웃 및 레이어 정규화
                self.dropout1 = nn.Dropout(dropout_rate)
                self.dropout2 = nn.Dropout(dropout_rate)
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)

            def forward(self, x, mask):
                # 셀프 어텐션 적용 및 잔차 연결, 레이어 정규화
                sa_out = self.sa(x, mask)
                sa_out = self.dropout1(sa_out)
                x1 = self.norm1(x + sa_out)

                # 피드포워드 네트워크 적용 및 잔차 연결, 레이어 정규화
                ffn_out = self.ffn(x1)
                ffn_out = self.dropout2(ffn_out)
                x2 = self.norm2(x1 + ffn_out)

                return x2


        # 전체 모델 구현
        class TextClassifier(nn.Module):
            def __init__(self, vocab_size, d_model, n_layers, dff, n_heads, dropout_rate=0.1):
                super().__init__()

                self.vocab_size = vocab_size
                self.d_model = d_model
                self.n_layers = n_layers
                self.dff = dff
                self.n_heads = n_heads

                # 임베딩 레이어
                self.embedding = nn.Embedding(vocab_size, d_model)
                # 위치 인코딩 설정 (학습되지 않는 파라미터)
                self.pos_encoding = nn.parameter.Parameter(positional_encoding(max_len, d_model), requires_grad=False)
                self.dropout = nn.Dropout(dropout_rate)
                # Transformer 레이어들을 모듈 리스트로 생성
                self.layers = nn.ModuleList(
                    [TransformerLayer(d_model, d_model, dff, n_heads, dropout_rate) for _ in range(n_layers)]
                )
                # 최종 분류를 위한 선형 레이어
                self.classification = nn.Linear(d_model, 1)

            def forward(self, x):
                # 패딩 토큰에 대한 마스크 생성
                mask = (x == tokenizer.pad_token_id)
                seq_len = x.shape[1]

                # 임베딩 및 위치 인코딩 적용
                x = self.embedding(x)
                x = x * sqrt(self.d_model)
                x = x + self.pos_encoding[:, :seq_len]
                x = self.dropout(x)

                # 각 Transformer 레이어 통과
                for layer in self.layers:
                    x = layer(x, mask)

                # 첫 번째 토큰의 출력만 사용 (분류를 위해)
                x = x[:, 0]
                x = self.classification(x)

                return x


        # 모델 초기화
        d_model = 32  # 임베딩 및 히든 레이어의 차원
        n_layers = 5  # Transformer 레이어의 수
        dff = 64  # 피드포워드 네트워크의 차원
        n_heads = 4  # 어텐션 헤드의 수

        model = TextClassifier(len(tokenizer), d_model, n_layers, dff, n_heads)
        model = model.to(device)  # 모델을 장치로 이동

        # 학습 준비
        from torch.optim import Adam  # 옵티마이저 임포트

        lr = 0.001  # 학습률 설정
        loss_fn = nn.BCEWithLogitsLoss()  # 손실 함수 정의 (이진 분류용)
        optimizer = Adam(model.parameters(), lr=lr)  # 옵티마이저 초기화


        # 정확도 계산 함수 정의
        def accuracy(model, dataloader):
            cnt = 0  # 총 샘플 수
            acc = 0  # 정확한 예측 수

            for data in dataloader:
                inputs = data['input_ids'].to(device)
                labels = data['label'].to(device)

                preds = model(inputs)
                preds = (preds > 0).long()[..., 0]  # 로짓을 이진 예측으로 변환

                cnt += labels.shape[0]
                acc += (labels == preds).sum().item()

            return acc / cnt  # 정확도 계산


        # 학습 시작
        n_epochs = 50  # 에포크 수 설정

        for epoch in range(n_epochs):
            total_loss = 0.
            model.train()  # 모델을 학습 모드로 설정
            for data in train_loader:
                optimizer.zero_grad()  # 그래디언트 초기화
                inputs = data['input_ids'].to(device)
                labels = data['label'].float().to(device)

                preds = model(inputs)[..., 0]
                loss = loss_fn(preds, labels)  # 손실 계산
                loss.backward()  # 역전파
                optimizer.step()  # 파라미터 업데이트

                total_loss += loss.item()  # 손실 누적

            logging.info(f"Epoch {epoch + 1:3d} | Train Loss: {total_loss}")  # 에포크별 손실 로그

            with torch.no_grad():
                model.eval()  # 모델을 평가 모드로 설정
                train_acc = accuracy(model, train_loader)  # 학습 데이터 정확도
                test_acc = accuracy(model, test_loader)  # 테스트 데이터 정확도
                logging.info(f"=========> Train acc: {train_acc:.3f} | Test acc: {test_acc:.3f}")  # 정확도 로그

        logging.info("학습이 완료되었습니다.")

    except Exception as e:
        logging.exception("예외 발생")
        raise
