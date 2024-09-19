import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

# 1. 데이터셋 준비하기
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

data_dir = './data'

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

full_train_dataset = torchvision.datasets.MNIST(root=data_dir,
                                                train=True,
                                                download=not os.path.exists(os.path.join(data_dir, 'MNIST', 'processed', 'training.pt')),
                                                transform=transform)

test_dataset = torchvision.datasets.MNIST(root=data_dir,
                                          train=False,
                                          download=not os.path.exists(os.path.join(data_dir, 'MNIST', 'processed', 'test.pt')),
                                          transform=transform)

batch_size = 256

# 훈련 데이터셋을 훈련과 검증으로 분할
train_size = int(len(full_train_dataset) * 0.8)
val_size = len(full_train_dataset) - train_size

train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size,
                                         shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# 2. 모델 정의하기
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 512)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MNISTModel().to(device)

# 3. 손실 함수와 옵티마이저 정의하기
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 4. 로그 저장을 위한 설정
# 현재 파일 이름 가져오기
if '__file__' in globals():
    script_name = os.path.basename(__file__)
    script_name_without_ext = os.path.splitext(script_name)[0]
else:
    script_name_without_ext = 'notebook'

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

# result 디렉토리 생성
result_dir = './result/1_basic'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 그래프와 로그 파일명 생성
graph_filename = f"{script_name_without_ext}_{timestamp}.png"
log_filename = f"{script_name_without_ext}_{timestamp}.txt"

graph_path = os.path.join(result_dir, graph_filename)
log_path = os.path.join(result_dir, log_filename)

# 로거 설정
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 로그 포맷 설정
formatter = logging.Formatter('%(message)s')

# 파일 핸들러 생성 및 설정
file_handler = logging.FileHandler(log_path)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 콘솔 핸들러 생성 및 설정
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 5. 정확도 측정 함수
def accuracy(model, dataloader):
    cnt = 0
    acc = 0

    model.eval()
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            preds = model(inputs)
            preds = torch.argmax(preds, dim=-1)

            cnt += labels.shape[0]
            acc += (labels == preds).sum().item()

    return acc / cnt

# 6. 학습 함수
def train(model, criterion, optimizer, train_loader, val_loader, test_loader, n_epochs):
    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        train_acc = accuracy(model, train_loader)
        val_acc = accuracy(model, val_loader)
        test_acc = accuracy(model, test_loader)

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)

        log_message = (f"Epoch [{epoch+1}/{n_epochs}], "
                       f"Loss: {epoch_loss:.4f}, "
                       f"Train Acc: {train_acc*100:.2f}%, "
                       f"Val Acc: {val_acc*100:.2f}%, "
                       f"Test Acc: {test_acc*100:.2f}%")
        logger.info(log_message)

    return train_acc_list, val_acc_list, test_acc_list

# 7. 모델 학습하기
n_epochs = 100
train_acc_list, val_acc_list, test_acc_list = train(model, criterion, optimizer, train_loader, val_loader, test_loader, n_epochs)

# 8. 정확도 그래프 그리기
def plot_acc(train_accs, val_accs, test_accs, label1='train', label2='validation', label3='test', save_path=None):
    x = np.arange(len(train_accs))

    plt.figure(figsize=(10, 6))
    plt.plot(x, train_accs, label=label1)
    plt.plot(x, val_accs, label=label2)
    plt.plot(x, test_accs, label=label3)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train, Validation, and Test Accuracy over Epochs')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"그래프가 '{save_path}'에 저장되었습니다.")
        plt.close()
    else:
        plt.show()

plot_acc(train_acc_list, val_acc_list, test_acc_list, save_path=graph_path)

# 9. 최종 검증 및 테스트 정확도 출력
final_val_acc = accuracy(model, val_loader)
final_test_acc = accuracy(model, test_loader)
logger.info(f'Final Validation Accuracy: {final_val_acc*100:.2f}%')
logger.info(f'Final Test Accuracy: {final_test_acc*100:.2f}%')
