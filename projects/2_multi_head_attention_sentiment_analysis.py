import logging
import os
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer

# 현재 파일 이름을 기반으로 로그 파일 경로 설정
current_filename = os.path.basename(__file__).replace('.py', '')
log_dir = "./result/2_basic/"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, f"{current_filename}.txt")

# 로그 설정
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

# 콘솔에도 로그 출력 추가
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# 코드의 실행 진입점을 정의합니다.
if __name__ == "__main__":
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 로컬에 저장된 데이터셋 불러오기
        ds = load_dataset("imdb")

        # 토크나이저 로드
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # 데이터셋 미리 토큰화
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=128  # max_len을 줄임
            )

        tokenized_datasets = ds.map(tokenize_function, batched=True)

        # 데이터셋의 포맷을 PyTorch 텐서로 설정
        tokenized_datasets.set_format(type='torch', columns=['input_ids', 'label'])

        # DataLoader 생성
        train_loader = DataLoader(
            tokenized_datasets['train'], batch_size=64, shuffle=True, num_workers=4
        )
        test_loader = DataLoader(
            tokenized_datasets['test'], batch_size=64, shuffle=False, num_workers=4
        )

        # Positional encoding
        def get_angles(pos, i, d_model):
            angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
            return pos * angle_rates

        def positional_encoding(position, d_model):
            angle_rads = get_angles(np.arange(position)[:, None], np.arange(d_model)[None, :], d_model)
            angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
            angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
            pos_encoding = angle_rads[None, ...]
            return torch.FloatTensor(pos_encoding)

        max_len = 128  # 토큰화에서 사용한 max_length와 동일하게 설정

        # Multi-head Attention
        class SelfAttention(nn.Module):
            def __init__(self, input_dim, d_model, n_heads):
                super().__init__()

                assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

                self.input_dim = input_dim
                self.d_model = d_model
                self.n_heads = n_heads
                self.d_head = d_model // n_heads

                self.wq = nn.Linear(input_dim, d_model)
                self.wk = nn.Linear(input_dim, d_model)
                self.wv = nn.Linear(input_dim, d_model)
                self.dense = nn.Linear(d_model, d_model)

                self.softmax = nn.Softmax(dim=-1)

            def forward(self, x, mask):
                B, S, _ = x.shape
                H = self.n_heads
                D = self.d_model
                D_head = self.d_head

                # Linear projections
                q = self.wq(x)  # (B, S, D)
                k = self.wk(x)  # (B, S, D)
                v = self.wv(x)  # (B, S, D)

                # Reshape to (B, S, H, D_head)
                q = q.view(B, S, H, D_head)
                k = k.view(B, S, H, D_head)
                v = v.view(B, S, H, D_head)

                # Transpose to (B, H, S, D_head)
                q = q.permute(0, 2, 1, 3)  # (B, H, S, D_head)
                k = k.permute(0, 2, 1, 3)  # (B, H, S, D_head)
                v = v.permute(0, 2, 1, 3)  # (B, H, S, D_head)

                # Compute scaled dot-product attention
                score = torch.matmul(q, k.transpose(-2, -1))  # (B, H, S, S)
                score = score / sqrt(D_head)

                if mask is not None:
                    mask = mask[:, None, None, :]  # (B, 1, 1, S)
                    score = score.masked_fill(mask == True, -1e9)

                # Apply softmax
                attention_weights = self.softmax(score)  # (B, H, S, S)

                # Compute attention output
                context = torch.matmul(attention_weights, v)  # (B, H, S, D_head)

                # Transpose and reshape back to (B, S, D)
                context = context.permute(0, 2, 1, 3).contiguous()  # (B, S, H, D_head)
                context = context.view(B, S, D)  # (B, S, D)

                # Final linear layer
                result = self.dense(context)  # (B, S, D)

                return result

        # Transformer Layer 구현
        class TransformerLayer(nn.Module):
            def __init__(self, input_dim, d_model, dff, n_heads, dropout_rate=0.1):
                super().__init__()

                self.input_dim = input_dim
                self.d_model = d_model
                self.dff = dff
                self.n_heads = n_heads

                self.sa = SelfAttention(input_dim, d_model, n_heads)
                self.ffn = nn.Sequential(
                    nn.Linear(d_model, dff),
                    nn.ReLU(),
                    nn.Linear(dff, d_model)
                )

                self.dropout1 = nn.Dropout(dropout_rate)
                self.dropout2 = nn.Dropout(dropout_rate)
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)

            def forward(self, x, mask):
                # Multi-head Attention with residual connection and layer normalization
                sa_out = self.sa(x, mask)
                sa_out = self.dropout1(sa_out)
                x1 = self.norm1(x + sa_out)

                # Feed-forward network with residual connection and layer normalization
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

                self.embedding = nn.Embedding(vocab_size, d_model)
                self.pos_encoding = nn.parameter.Parameter(positional_encoding(max_len, d_model), requires_grad=False)
                self.dropout = nn.Dropout(dropout_rate)
                self.layers = nn.ModuleList(
                    [TransformerLayer(d_model, d_model, dff, n_heads, dropout_rate) for _ in range(n_layers)])
                self.classification = nn.Linear(d_model, 1)

            def forward(self, x):
                mask = (x == tokenizer.pad_token_id)
                seq_len = x.shape[1]

                x = self.embedding(x)
                x = x * sqrt(self.d_model)
                x = x + self.pos_encoding[:, :seq_len]
                x = self.dropout(x)

                for layer in self.layers:
                    x = layer(x, mask)

                x = x[:, 0]
                x = self.classification(x)

                return x

        # 모델 초기화
        d_model = 32
        n_layers = 5
        dff = 64
        n_heads = 4

        model = TextClassifier(len(tokenizer), d_model, n_layers, dff, n_heads)
        model = model.to(device)

        # 학습 준비
        from torch.optim import Adam

        lr = 0.001
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = Adam(model.parameters(), lr=lr)

        def accuracy(model, dataloader):
            cnt = 0
            acc = 0

            for data in dataloader:
                inputs = data['input_ids'].to(device)
                labels = data['label'].to(device)

                preds = model(inputs)
                preds = (preds > 0).long()[..., 0]

                cnt += labels.shape[0]
                acc += (labels == preds).sum().item()

            return acc / cnt

        # 학습
        n_epochs = 50

        for epoch in range(n_epochs):
            total_loss = 0.
            model.train()
            for data in train_loader:
                optimizer.zero_grad()
                inputs = data['input_ids'].to(device)
                labels = data['label'].float().to(device)

                preds = model(inputs)[..., 0]
                loss = loss_fn(preds, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            logging.info(f"Epoch {epoch + 1:3d} | Train Loss: {total_loss}")

            with torch.no_grad():
                model.eval()
                train_acc = accuracy(model, train_loader)
                test_acc = accuracy(model, test_loader)
                logging.info(f"=========> Train acc: {train_acc:.3f} | Test acc: {test_acc:.3f}")

        logging.info("학습이 완료되었습니다.")

    except Exception as e:
        logging.exception("예외 발생")
        raise
