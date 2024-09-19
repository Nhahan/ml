import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import logging

# 1. 데이터셋 준비하기
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

data_dir = './data'

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

train_dataset = torchvision.datasets.CIFAR10(root=data_dir,
                                             train=True,
                                             download=not os.path.exists(os.path.join(data_dir, 'cifar-10-batches-py')),
                                             transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root=data_dir,
                                            train=False,
                                            download=not os.path.exists(os.path.join(data_dir, 'cifar-10-batches-py')),
                                            transform=transform)

batch_size = 256

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# 입력 데이터의 형태 확인
data_iter = iter(train_loader)
images, labels = next(data_iter)  # 여기서 수정합니다.
print(f"이미지 크기: {images.shape}")  # [batch_size, channels, height, width]


# 2. 모델 정의하기
class CIFAR10Model(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()
        self.layer1 = nn.Linear(3 * 32 * 32, 1024)
        self.leaky_relu = nn.LeakyReLU()
        self.layer2 = nn.Linear(1024, 512)
        self.layer3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        out = self.layer1(x)
        out = self.leaky_relu(out)
        out = self.layer2(out)
        out = self.leaky_relu(out)
        out = self.layer3(out)
        return out


# Sigmoid 활성화 함수를 사용하는 모델
class CIFAR10ModelSigmoid(nn.Module):
    def __init__(self):
        super(CIFAR10ModelSigmoid, self).__init__()
        self.layer1 = nn.Linear(3 * 32 * 32, 1024)
        self.sigmoid = nn.Sigmoid()
        self.layer2 = nn.Linear(1024, 512)
        self.layer3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        out = self.layer1(x)
        out = self.sigmoid(out)
        out = self.layer2(out)
        out = self.sigmoid(out)
        out = self.layer3(out)
        return out


# Dropout이 적용된 모델
class CIFAR10ModelDropout(nn.Module):
    def __init__(self):
        super(CIFAR10ModelDropout, self).__init__()
        self.layer1 = nn.Linear(3 * 32 * 32, 1024)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)
        self.layer2 = nn.Linear(1024, 512)
        self.layer3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        out = self.layer1(x)
        out = self.leaky_relu(out)
        out = self.dropout(out)
        out = self.layer2(out)
        out = self.leaky_relu(out)
        out = self.dropout(out)
        out = self.layer3(out)
        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 3. 옵티마이저 설정
criterion = nn.CrossEntropyLoss()

# SGD 모델
model_sgd = CIFAR10Model().to(device)
optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.001)

# Adam 모델
model_adam = CIFAR10Model().to(device)
optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001)

# LeakyReLU 모델
model_leakyrelu = CIFAR10Model().to(device)
optimizer_leakyrelu = optim.Adam(model_leakyrelu.parameters(), lr=0.001)

# Sigmoid 모델
model_sigmoid = CIFAR10ModelSigmoid().to(device)
optimizer_sigmoid = optim.Adam(model_sigmoid.parameters(), lr=0.001)

# Dropout 모델
model_dropout = CIFAR10ModelDropout().to(device)
optimizer_dropout = optim.Adam(model_dropout.parameters(), lr=0.001)

# 4. 로그 및 결과 저장 설정
if '__file__' in globals():
    script_name = os.path.basename(__file__)
    script_name_without_ext = os.path.splitext(script_name)[0]
else:
    script_name_without_ext = 'notebook'

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
result_dir = './result/1_advanced'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 로그 파일 설정
log_filename = f"{script_name_without_ext}_{timestamp}.txt"
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
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


# 6. 학습 함수
def train(model, criterion, optimizer, train_loader, n_epochs, logger, model_name):
    train_acc_list = []

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
        train_acc_list.append(train_acc)

        log_message = (
            f'{model_name} - Epoch [{epoch + 1}/{n_epochs}], '
            f'Loss: {epoch_loss:.4f}, '
            f'Train Acc: {train_acc * 100:.2f}%'
        )
        logger.info(log_message)

    return train_acc_list


def train_with_validation(model, criterion, optimizer, train_loader, test_loader, n_epochs, logger, model_name):
    train_acc_list = []
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

        with torch.no_grad():
            model.eval()
            train_acc = accuracy(model, train_loader)
            test_acc = accuracy(model, test_loader)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        log_message = (
            f'{model_name} - Epoch [{epoch + 1}/{n_epochs}], '
            f'Loss: {epoch_loss:.4f}, '
            f'Train Acc: {train_acc * 100:.2f}%, '
            f'Test Acc: {test_acc * 100:.2f}%'
        )
        logger.info(log_message)

    return train_acc_list, test_acc_list


# 7. 모델 학습하기
n_epochs = 50

# SGD 옵티마이저로 학습
logger.info("SGD Optimizer Training Started")
train_acc_sgd = train(model_sgd, criterion, optimizer_sgd, train_loader, n_epochs, logger, 'SGD')

# Adam 옵티마이저로 학습
logger.info("Adam Optimizer Training Started")
train_acc_adam = train(model_adam, criterion, optimizer_adam, train_loader, n_epochs, logger, 'Adam')

# 활성화 함수 비교
logger.info("LeakyReLU Model Training Started")
train_acc_leakyrelu = train(model_leakyrelu, criterion, optimizer_leakyrelu, train_loader, n_epochs, logger,
                            'LeakyReLU')

logger.info("Sigmoid Model Training Started")
train_acc_sigmoid = train(model_sigmoid, criterion, optimizer_sigmoid, train_loader, n_epochs, logger, 'Sigmoid')

# Dropout 모델 학습
logger.info("Dropout Model Training Started")
train_acc_dropout, test_acc_dropout = train_with_validation(
    model_dropout, criterion, optimizer_dropout, train_loader, test_loader, n_epochs, logger, 'Dropout'
)


# 8. 결과 그래프 저장
def plot_train_acc(acc_list1, acc_list2, label1, label2, save_path):
    x = np.arange(len(acc_list1))

    plt.figure(figsize=(10, 6))
    plt.plot(x, acc_list1, label=label1)
    plt.plot(x, acc_list2, label=label2)
    plt.xlabel('Epoch')
    plt.ylabel('Train Accuracy')
    plt.title('Train Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"그래프가 '{save_path}'에 저장되었습니다.")


def plot_train_test_acc(train_accs, test_accs, save_path):
    x = np.arange(len(train_accs))

    plt.figure(figsize=(10, 6))
    plt.plot(x, train_accs, label='Train Accuracy')
    plt.plot(x, test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Test Accuracy with Dropout')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"그래프가 '{save_path}'에 저장되었습니다.")


# 그래프 저장 경로 설정
plot1_filename = f"{script_name_without_ext}_{timestamp}_optimizer_comparison.png"
plot1_path = os.path.join(result_dir, plot1_filename)

plot2_filename = f"{script_name_without_ext}_{timestamp}_activation_comparison.png"
plot2_path = os.path.join(result_dir, plot2_filename)

plot3_filename = f"{script_name_without_ext}_{timestamp}_dropout.png"
plot3_path = os.path.join(result_dir, plot3_filename)

# 그래프 저장
plot_train_acc(train_acc_sgd, train_acc_adam, 'SGD', 'Adam', plot1_path)
plot_train_acc(train_acc_leakyrelu, train_acc_sigmoid, 'LeakyReLU', 'Sigmoid', plot2_path)
plot_train_test_acc(train_acc_dropout, test_acc_dropout, plot3_path)
