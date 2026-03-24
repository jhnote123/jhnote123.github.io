```python
## PyTorch Learn the Basics - 0. Quickstart
# https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
```

## 1. Working with data

PyTorch에서 데이터를 다루기 위한 두 가지: `torch.utils.data.DataLoader`와 `torch.utils.data.Dataset`
- `Dataset`은 샘플과 그에 해당하는 정답(label)을 저장하고, `DataLoader`는 `Dataset`을 감싸서 반복 가능한 객체(iterable)로 만든다.

PyTorch는 TorchText, TorchVision, TorchAudio와 같은 domain-specific 라이브러리를 제공, 이들 모두 datasets을 가지고 있음.

torchvision.datasets 모듈은 CIFAR, COCO 등 다양한 real-world vision data를 위한 `Dataset` 객체를 포함.


```python
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
```

모든 TorchVision `Dataset`은 samples과 labels을 각각 수정하기 위한 `transform` `target_transform`이라는 두 가지 인자를 포함.


```python
train_data = datasets.FashionMNIST(
    root="data",
    train=True, # download training data
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False, # download test data
    download=True,
    transform=ToTensor(),
)
```

    100%|██████████| 26.4M/26.4M [00:02<00:00, 10.9MB/s]
    100%|██████████| 29.5k/29.5k [00:00<00:00, 168kB/s]
    100%|██████████| 4.42M/4.42M [00:01<00:00, 3.15MB/s]
    100%|██████████| 5.15k/5.15k [00:00<00:00, 25.3MB/s]
    


```python

```

`Dataset`을 `DataLoader`의 인자로 전달 $\rightarrow$ 데이터셋을 iterable로 감싸며, 자동 배치 처리, sampling, shuffling, multiprocess data loading을 지원


```python
batch_size = 64

# create data loaders
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
```

    Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
    Shape of y: torch.Size([64]) torch.int64
    


```python

```

## 2. Creating Models

신경망(neural network)을 정의하기 위해 `nn.Module`을 상속받는 클래스를 생성.

신경망의 layers은 `__init__` 함수에서, 데이터를 입력받아 어떤 방식으로 층을 통과시킬지 `forward`함수에서 정의.


```python
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
```


```python
# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() # 텐서를 1차원으로 평탄화
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
```

    NeuralNetwork(
      (flatten): Flatten(start_dim=1, end_dim=-1)
      (linear_relu_stack): Sequential(
        (0): Linear(in_features=784, out_features=512, bias=True)
        (1): ReLU()
        (2): Linear(in_features=512, out_features=512, bias=True)
        (3): ReLU()
        (4): Linear(in_features=512, out_features=10, bias=True)
      )
    )
    

nn.Flatten(start_dim, end_dim)

- default: start_dim=1, end_dim=-1
- 위에서 input X의 shape은 [N, C, H, W]: torch.Size([64, 1, 28, 28])
- 두 번째 차원(start_dim=1)부터 마지막 차원(end_dim=-1)까지 평탄화
- 이 예시의 경우 [64, 1 x 28 x 28] = [64, 784]. 즉, 64장의 데이터를 각각 784개의 feature를 가진 벡터로 변환


```python

```

## 3. Optimizing the Model Parameters

모델을 학습시키려면 손실 함수(loss function)와 옵티마이저(optimizer)가 필요.

training loop에서 모델은 training dataset에 대해 예측을 수행하고, 그 결과로 얻게 되는 prediction error를 역전파하여 모델의 parameters를 조정.

학습 과정은 여러 번의 반복(즉, epochs)을 거쳐 진행.

각 에폭 동안 모델은 더 나은 예측을 위해 parameters 업데이트.


```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```


```python
from numpy._core.numeric import correlate
from tqdm.auto import tqdm

def train(dataloader, model, loss_fn, optimizer):
    model.train()

    for i, (X, y) in enumerate(tqdm(dataloader)):
        X, y = X.to(device), y.to(device)

        # compute loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 

        if i % 100 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f}")

def test(dataloader, model, loss_fn):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= len(dataloader)
    correct /= len(dataloader.dataset) 
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```


```python
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
```

    Epoch 1
    -------------------------------
    


      0%|          | 0/938 [00:00<?, ?it/s]


    loss: 1.026021
    loss: 1.060960
    loss: 0.852260
    loss: 1.006915
    loss: 0.905024
    loss: 0.921260
    loss: 0.970937
    loss: 0.925269
    loss: 0.948763
    loss: 0.901032
    Test Error: 
     Accuracy: 67.6%, Avg loss: 0.904582 
    
    Epoch 2
    -------------------------------
    


      0%|          | 0/938 [00:00<?, ?it/s]


    loss: 0.941326
    loss: 0.994598
    loss: 0.770832
    loss: 0.943694
    loss: 0.848546
    loss: 0.852915
    loss: 0.917956
    loss: 0.876938
    loss: 0.890671
    loss: 0.854691
    Test Error: 
     Accuracy: 69.0%, Avg loss: 0.853306 
    
    Epoch 3
    -------------------------------
    


      0%|          | 0/938 [00:00<?, ?it/s]


    loss: 0.874450
    loss: 0.945199
    loss: 0.710164
    loss: 0.896213
    loss: 0.808469
    loss: 0.801722
    loss: 0.877895
    loss: 0.843071
    loss: 0.847157
    loss: 0.818912
    Test Error: 
     Accuracy: 70.3%, Avg loss: 0.814336 
    
    Epoch 4
    -------------------------------
    


      0%|          | 0/938 [00:00<?, ?it/s]


    loss: 0.821284
    loss: 0.905445
    loss: 0.663028
    loss: 0.859200
    loss: 0.777941
    loss: 0.762276
    loss: 0.845442
    loss: 0.817984
    loss: 0.813387
    loss: 0.789932
    Test Error: 
     Accuracy: 71.5%, Avg loss: 0.783322 
    
    Epoch 5
    -------------------------------
    


      0%|          | 0/938 [00:00<?, ?it/s]


    loss: 0.777526
    loss: 0.871704
    loss: 0.625008
    loss: 0.829486
    loss: 0.753321
    loss: 0.731359
    loss: 0.817785
    loss: 0.798353
    loss: 0.786242
    loss: 0.765426
    Test Error: 
     Accuracy: 72.5%, Avg loss: 0.757602 
    
    


```python

```

## 4. Saving Models


```python
torch.save(model.state_dict(), "model.pth")
```


```python

```

## 5. Loading Models

모델의 구조를 다시 생성하고, 그 안에 state dictionary를 loading


```python
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))
```




    <All keys matched successfully>




```python

```
