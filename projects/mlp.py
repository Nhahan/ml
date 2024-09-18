import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np


# 시드 고정
seed = 7777

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# XOR 데이터 생성
# 입력 데이터 (4 x 2 행렬)
x = torch.tensor([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
])


# 출력 데이터 (4차원 벡터)
y = torch.tensor([0, 1, 1, 0], dtype=torch.float32)


# MLP 모델 정의
class MLP(nn.Module):
    # input_size: 입력 데이터의 특성 수 / hidden_size: 은닉층의 뉴런 수
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()

        # 첫 번째 선형 레이어
        self.layer1 = nn.Linear(input_size, hidden_size)
        # ReLU 활성화 함수
        self.relu = nn.ReLU()
        # 두 번째 선형 레이어
        self.layer2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # 순전파 정의
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out


# 모델 생성 (인스턴스화)
model = MLP(input_size=2, hidden_size=10)


# 손실함수와 옵티마이저 정의
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


# 모델 학습 정의
def train(model, criterion, optimizer, x, y, num_epochs):
    for epoch in range(num_epochs):
        # 그래디언트 초기화
        optimizer.zero_grad()

        # 순전파
        outputs = model(x)
        outputs = outputs.view(-1)  # 출력 형태 조정

        # 손실 계산
        loss = criterion(outputs, y)

        # 역전파 및 가중치 업데이트
        loss.backward()
        optimizer.step()

        # 학습 과정 출력
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# 모델 학습
num_epochs = 1000
train(model, criterion, optimizer, x, y, num_epochs)


# 학습된 모델의 성능 테스트
with torch.no_grad():
    outputs = model(x)
    predicted = outputs.view(-1)
    print('Predicted outputs:')
    print(predicted)
    print('Actual outputs:')
    print(y)
