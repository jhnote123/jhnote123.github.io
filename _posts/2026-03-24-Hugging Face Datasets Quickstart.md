# Hugging Face Datasets Quickstart


```python
# https://huggingface.co/docs/datasets/quickstart
```


```python
# 기본 Datasets 라이브러리 설치
# pip install datasets

# 오디오 데이터를 다룰 경우
# pip install datasets[audio]

# 이미지/비전 데이터를 다룰 경우
# pip install datasets[vision]
```


```python

```

## 1. Audio

audio datasets은 text datasets과 같은 방식으로 불러올 수 있음. 그러나 전처리 과정은 약간 다름. 토크나이저 대신 **feature extractor**가 필요

#### (1) 데이터셋 불러오기


```python
from datasets import load_dataset, Audio

dataset = load_dataset("PolyAI/minds14", "en-US", split="train")
```

    /usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(
    


    README.md: 0.00B [00:00, ?B/s]


    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
    WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
    


    en-US/train-00000-of-00001.parquet:   0%|          | 0.00/34.2M [00:00<?, ?B/s]



    Generating train split:   0%|          | 0/563 [00:00<?, ? examples/s]


#### (2) model과 feature extractor 불러오기


```python
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

# load pretrained Wav2Vec2 model and corresponding feature extractor
model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base")
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
```


    Loading weights:   0%|          | 0/211 [00:00<?, ?it/s]


    Wav2Vec2ForSequenceClassification LOAD REPORT from: facebook/wav2vec2-base
    Key                          | Status     | 
    -----------------------------+------------+-
    project_hid.weight           | UNEXPECTED | 
    quantizer.codevectors        | UNEXPECTED | 
    project_q.bias               | UNEXPECTED | 
    project_q.weight             | UNEXPECTED | 
    project_hid.bias             | UNEXPECTED | 
    quantizer.weight_proj.weight | UNEXPECTED | 
    quantizer.weight_proj.bias   | UNEXPECTED | 
    projector.bias               | MISSING    | 
    classifier.bias              | MISSING    | 
    classifier.weight            | MISSING    | 
    projector.weight             | MISSING    | 
    
    Notes:
    - UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
    - MISSING	:those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
    

#### (3) Audio Resampling

MInDS-14 dataset의 sampling rate는 8kHz, 그러나 Wav2Vec2 model은 16kHZ의 sampling rate로 사전학습된 상태.

$\rightarrow$ model의 sampling rate에 맞추기 위해 `cast_column()` 함수와 `Audio` 기능을 사용하여 `audio`column을 업샘플링해야 함.


```python
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000)) # 16kHZ로 업샘플링
dataset[0]["audio"]
```




    <datasets.features._torchcodec.AudioDecoder at 0x7f97370fc4d0>



#### (4) 전처리 함수 만들기 및 적용



```python
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=16000,
        padding=True,
        max_length=100000,
        truncation=True,
    ) # NLP에서 문장 길이를 맞췄던 것처럼, 오디오도 padding과 truncation으로 똑같은 길이의 텐서로 맞춤
    return inputs

dataset = dataset.map(preprocess_function, batched=True)
```


    Map:   0%|          | 0/563 [00:00<?, ? examples/s]


feature extractor로 audio array를 전처리하고, sequences을 자르거나 패딩하여 rectangular 형태의 tensors로 만드는 함수

실제 음성 신호인 배열 자체가 모델의 입력이 되기 때문에 feature extractor 안에서 이 오디오 배열을 호출해야 한다.

#### (5) 라벨 이름 변경 및 PyTorch 포맷팅


`Wav2Vec2ForSequenceClassification` 모델이 기대하는 입력 이름인 labels로 맞추기 위해, `rename_column()` 함수를 사용하여 intent_class 열의 이름을 labels로 변경


```python
dataset = dataset.rename_column("intent_class", "labels")
```

사용하는 프레임워크에 맞춰 dataset format을 설정.

`set_format()` 함수를 사용하여 dataset format을 `torch`로 설정하고 포맷할 열을 지정. 이 함수는 on-the-fly 방식으로 포맷을 적용.

파이토치 텐서로 변환한 후에는 데이터셋을 `torch.utils.data.DataLoader`에 넣는다.


```python
from torch.utils.data import DataLoader

dataset.set_format(type="torch", columns=["input_values", "labels"])
dataloader = DataLoader(dataset, batch_size=4)
```

## 2. Vision

Image datasets도 text datasets과 같은 방식으로 불러온다. 단, 토크나이저 대신 데이터셋을 전처리할 feature extractor가 필요하다.

컴퓨터 비전에서는 모델이 과적합되는 것을 방지하고 더 강건해지도록 이미지에 **data augmentation**을 적용하는 것이 일반적이다.

어떤 data augmentation library든 자유롭게 사용한 후, `Datasets`을 통해 그 augmentation을 적용할 수 있다.

#### (1) 데이터셋 불러오기 및 RGB 변환



```python
from datasets import load_dataset, Image

dataset = load_dataset("AI-Lab-Makerere/beans", split="train")
```


    README.md: 0.00B [00:00, ?B/s]



    data/train-00000-of-00001.parquet:   0%|          | 0.00/144M [00:00<?, ?B/s]



    data/validation-00000-of-00001.parquet:   0%|          | 0.00/18.5M [00:00<?, ?B/s]



    data/test-00000-of-00001.parquet:   0%|          | 0.00/17.7M [00:00<?, ?B/s]



    Generating train split:   0%|          | 0/1034 [00:00<?, ? examples/s]



    Generating validation split:   0%|          | 0/133 [00:00<?, ? examples/s]



    Generating test split:   0%|          | 0/128 [00:00<?, ? examples/s]


대부분의 이미지 모델은 RGB 이미지로 작동한다.

만약 데이터셋에 다른 형식의 이미지(예: 흑백 이미지)가 포함되어 있다면, `cast_column()` 함수를 사용하여 RGB로 설정할 수 있다.


```python
dataset = dataset.cast_column("image", Image(mode="RGB"))
```

#### (2) data augmentation 정의 및 적용

data augmentation library(Albumentations, imgaug, Kornia)를 사용하여 데이터 증강을 추가할 수 있다.


```python
# torchvision을 사용하여 이미지의 색상 속성을 무작위로 변경
from torchvision.transforms import Compose, ColorJitter, ToTensor

jitter = Compose(
    [ColorJitter(brightness=0.5, hue=0.5), ToTensor()]
)
```

아래는 데이터셋에 변환(transform)을 적용하고 모델의 입력값인 `pixel_values`를 생성하는 함수


```python
# 데이터셋의 각 이미지에 위에서 만든 'jitter' 변환을 적용하는 함수
def transforms(examples):
    examples["pixel_values"] = [jitter(image.convert("RGB")) for image in examples["image"]]
    return examples
```

`with_transform()` 함수를 사용하여 data augmentation 적용
- `with_transform()`은 on-the-fly 방식. 즉, 데이터를 꺼낼 때마다 그때그때 변환 적용


```python
# map() 대신 with_transform()을 사용하여 실시간으로 변환 적용
dataset = dataset.with_transform(transforms)
```

#### (3) PyTorch 포맷팅 및 DataLoader 묶기


```python
from torch.utils.data import DataLoader

# 개별 이미지 텐서들을 하나의 큰 배치(묶음)로 합쳐주는 역할
def collate_fn(examples):
    images = []
    labels = []
    for example in examples:
        images.append((example["pixel_values"]))
        labels.append(example["labels"])

    # # 리스트에 담긴 개별 텐서들을 torch.stack으로 쌓아서 하나의 텐서 블록으로
    pixel_values = torch.stack(images)
    labels = torch.tensor(labels)
    return {"pixel_values": pixel_values, "labels": labels}

dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=4)
```

## 3. NLP


```python
from datasets import load_dataset

dataset = load_dataset("nyu-mll/glue", "mrpc", split="train")
```


    README.md: 0.00B [00:00, ?B/s]



    mrpc/train-00000-of-00001.parquet:   0%|          | 0.00/649k [00:00<?, ?B/s]



    mrpc/validation-00000-of-00001.parquet:   0%|          | 0.00/75.7k [00:00<?, ?B/s]



    mrpc/test-00000-of-00001.parquet:   0%|          | 0.00/308k [00:00<?, ?B/s]



    Generating train split:   0%|          | 0/3668 [00:00<?, ? examples/s]



    Generating validation split:   0%|          | 0/408 [00:00<?, ? examples/s]



    Generating test split:   0%|          | 0/1725 [00:00<?, ? examples/s]



```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```


    config.json:   0%|          | 0.00/570 [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/440M [00:00<?, ?B/s]



    Loading weights:   0%|          | 0/199 [00:00<?, ?it/s]


    BertForSequenceClassification LOAD REPORT from: bert-base-uncased
    Key                                        | Status     | 
    -------------------------------------------+------------+-
    cls.predictions.transform.LayerNorm.bias   | UNEXPECTED | 
    cls.predictions.transform.dense.bias       | UNEXPECTED | 
    cls.seq_relationship.bias                  | UNEXPECTED | 
    cls.seq_relationship.weight                | UNEXPECTED | 
    cls.predictions.transform.dense.weight     | UNEXPECTED | 
    cls.predictions.bias                       | UNEXPECTED | 
    cls.predictions.transform.LayerNorm.weight | UNEXPECTED | 
    classifier.bias                            | MISSING    | 
    classifier.weight                          | MISSING    | 
    
    Notes:
    - UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
    - MISSING	:those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
    


    tokenizer_config.json:   0%|          | 0.00/48.0 [00:00<?, ?B/s]



    vocab.txt: 0.00B [00:00, ?B/s]



    tokenizer.json: 0.00B [00:00, ?B/s]



```python
def encode(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length")

dataset = dataset.map(encode, batched=True)
```

import torch

dataset = dataset.select_columns(["input_ids", "token_type_ids", "attention_mask", "label"])
dataset = dataset.with_format(type="torch")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
```
