---
title: "Hugging Face Vision Transformer (ViT)"
date: 2026-03-27 16:00:00 +0900
categories: [Hugging Face]
tags: [vision language model, vision transformer, vit, huggingface]
---

# Hugging Face Vision Transformer (ViT)


```python
# https://huggingface.co/docs/transformers/model_doc/vit
```

ViT는 computer vision tasks을 위해 adapted된 transformer이다.

image를 **더 작은 고정된 크기의 patches로 분할**되며, 이 patches은 NLP tasks의 단어들과 유사하게 **sequence of tokens로 취급**된다.


```python
# Pipeline

import torch
from transformers import pipeline

pipeline = pipeline(
    task="image-classification",
    model="google/vit-base-patch16-224",
    dtype=torch.float16,
    device=0
)
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
```

    /usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(
    


    config.json: 0.00B [00:00, ?B/s]


    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
    WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
    


    model.safetensors:   0%|          | 0.00/346M [00:00<?, ?B/s]



    Loading weights:   0%|          | 0/200 [00:00<?, ?it/s]



    preprocessor_config.json:   0%|          | 0.00/160 [00:00<?, ?B/s]


    Fast image processor class <class 'transformers.models.vit.image_processing_vit_fast.ViTImageProcessorFast'> is available for this model. Using slow image processor class. To use the fast image processor class set `use_fast=True`.
    




    [{'label': 'lynx, catamount', 'score': 0.43471530079841614},
     {'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor',
      'score': 0.03472110256552696},
     {'label': 'snow leopard, ounce, Panthera uncia',
      'score': 0.03236362338066101},
     {'label': 'Egyptian cat', 'score': 0.023956838995218277},
     {'label': 'tiger cat', 'score': 0.02285977452993393}]




```python
# AutoModel

import torch
import requests
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor

image_processor =AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224",
    use_fast=True,
)

model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = image_processor(image, return_tensors="pt").to(model.device)

with torch.no_grad():
  logits = model(**inputs).logits
predicted_class_id = logits.argmax(dim=-1).item()

class_labels = model.config.id2label
predicted_class_label = class_labels[predicted_class_id]
print(f"The predicted class label is: {predicted_class_label}")
```


    Loading weights:   0%|          | 0/200 [00:00<?, ?it/s]


    The predicted class label is: lynx, catamount
    

## 1. ViTConfig

### 1.1 class transformers.ViTConfig
Parameters
- (1) 패치 분할 및 입력 차원
- `image_size`(기본값: 224): 입력 이미지의 크기(해상도), $H \times W$
- `patch_size`(기본값: 16): 이미지를 분할할 개별 패치의 크기, $P \times P$
- `num_channels`(기본값: 3): 입력 이미지의 채널 수 (RGB의 경우 3)
- 트랜스포머에 입력으로 넣기 위해 1차원 시퀀스로 처리한다. 224 x 224 이미지에 16 x 16 패치 사용 시, 시퀀스 길이는 196이 된다. 여기에 `[CLS]` 토큰이 추가되어 최종 입력 토큰 수는 197개가 된다.
- (2) 트랜스포머 아키텍처
- `hidden_size`(기본값: 768): 은닉 표현(Hidden representations)의 차원
- `num_hidden_layers`(기본값: 12): 트랜스포머 블록(Layer)의 총 개수
- `num_attention_heads`(기본값: 12): 어텐션 헤드의 개수
- `intermediate_size`(기본값: 3072): 트랜스포머 블록 내부의 피드포워드 신경망(MLP)의 중간 확장 차원
- `qkv_bias`(기본값: True): Query, Key, Value를 생성하는 선형 투영(Linear Projection) 행렬에 편향(Bias) 항을 추가할지 여부
- (3) 활성화 함수 및 정규화
- `hidden_act`(기본값: "gelu"): MLP 레이어에서 사용되는 비선형 활성화 함수
- `layer_norm_eps`(기본값: 1e-12): Layer Normalization 연산 시, 분모가 0이 되는 것을 방지하기 위해 더해지는 값
- (4) 정규화 및 가중치 초기화
- `hidden_dropout_prob`(기본값: 0.0): 임베딩 계층, 인코더의 MLP 계층, 풀러(Pooler) 계층의 완전 연결망(Fully Connected Layers)에 적용되는 드롭아웃 확률
- `attention_probs_dropout_prob`(기본값: 0.0): 소프트맥스(Softmax)를 통과한 어텐션 가중치(Attention Probabilities) 행렬에 직접 적용되는 드롭아웃 비율입
- `initializer_range`(기본값: 0.02)
- (5) 출력 및 task-specific heads
- `pooler_output_size`(선택사항): 풀러(Pooler) 레이어의 출력 차원. `None`일 경우 `hidden_size`와 동일한 값으로 설정
- `pooler_act`(기본값: "tanh"): 풀러에서 사용되는 활성화 함수
- classification task 수행 시, 전체 시퀀스(197개 토큰) 중 `[CLS]` 토큰의 임베딩 벡터만 추출한다. 이후 추출된 벡터를 dense layer와 tanh activation function에 통과시킨 다음, 최종 classifier head에 전달한다.
- `encoder_stride`(기본값: 16): masked image modeling을 수행할 때, decoder head에서 해상도를 증가시키기 위해 사용


```python
from transformers import ViTConfig, ViTModel

# vit-base-patch16-224 style configuration
configuration = ViTConfig()

# Initializing a model (with random weights) from the vit-base-patch16-224 style configuration
model = ViTModel(configuration)

print(model.config)
```

    ViTConfig {
      "attention_probs_dropout_prob": 0.0,
      "encoder_stride": 16,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.0,
      "hidden_size": 768,
      "image_size": 224,
      "initializer_range": 0.02,
      "intermediate_size": 3072,
      "layer_norm_eps": 1e-12,
      "model_type": "vit",
      "num_attention_heads": 12,
      "num_channels": 3,
      "num_hidden_layers": 12,
      "patch_size": 16,
      "pooler_act": "tanh",
      "pooler_output_size": 768,
      "qkv_bias": true,
      "transformers_version": "5.0.0"
    }
    
    

## 2. ViTImageProcessor

### 2.1 class transformers.ViTImageProcessor
전처리할 이미지를 입력으로 사용한다.

픽셀 값이 0에서 255 사이인 하나의 이미지 또는 이미지 배치를 입력으로 받는다. 픽셀 값이 0에서 1사이인 이미지를 전달하는 경우 `do_rescale=False`로 설정

`return_tensors`가 `pt`로 설정된 경우 stacked tensors 반환, 그렇지 않으면 tensors의 리스트를 반환

## 3. ViTModel

모델 상단에 specific head가 부착되지 않은, raw hidden-states을 출력하는 순수 ViT 모델이다.

###3.1 class transformers.ViTModel
Parameters
- `config` (`ViTConfig`)
- `add_pooling_layer`(기본값: `True`): pooling layer 추가할지 여부
- `use_mask_token`(기본값: `False`): masked Image modeling을 할 경우, mask token을 사용할지 여부

####3.1.1 forward
Parameters
- `pixel_values`(`torch.Tensor`): `(batch_size, num_channels, image_size, image_size)`의 형태. `ViTImageProcessor`를 통해 얻을 수 있음.
- `bool_masked_pos`(`torch.BoolTensor`): `(batch_size, num_patches)` 형태로, 특정 패치의 마스킹 여부(1: 마스킹됨, 0: 정상)를 지정
- `interpolate_pos_encoding`(`bool`): 사전학습된 위치 임베딩을 보간할지 여부

Returns
- `last_hidden_state`(`torch.FloatTensor`): `(batch_size, sequence_length, hidden_size)`의 형태로 모델의 마지막 레이어에서 출력된 sequence of hidden-states
- `pooler_output`(`torch.FloatTensor`): 추가적인 처리를 거친(dense layer와 tanh act를 통과한), 첫 번째 토큰([CLS] token)의 마지막 레이어의 hidden-state
- `hidden_states`(`torch.FloatTensor`): optional. `output_hidden_states=True`를 파라미터로 넘기거나 `config.output_hidden_states=True`로 지정되어 있을 때 반환
- `attentions`: optional. attention sofmax를 통과한 이후의 attentions weights. `output_attentions=True`를 파라미터로 넘기거나 `config.output_attentions=True`로 지정되어 있을 때 반환.

## 4. ViTForMaskedImageModeling

masked image modeling을 위해 top에 decoder가 얹혀진 ViT 모델

### 4.1 class transformers.ViTForMaskedImageModeling
Parameters
- config (`ViTConfig`)

#### 4.1.1 forward
Parameters
- `pixel_values `
- `bool_masked_pos `
- `interpolate_pos_encodin`

Returns
- `loss` ((1, ) 형태의 `torch.FloatTensor`): `bool_masked_pos`가 제공될 때 반환되는 reconstruction loss. 모델이 재구성한 픽셀들이 원본 이미지의 실제 픽셀과 얼마나 차이가 나는지 계산한 오차.
- `reconstruction` (`(batch_size, num_channels, height, width)`형태의 `torch.FloatTensor`): reconstructed / completed images.
- `hidden_states`
- `attentions`


```python
from transformers import AutoImageProcessor, ViTForMaskedImageModeling
import torch
from PIL import Image
import httpx
from io import BytesIO

# Download sample image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
with httpx.stream("GET", url) as response:
    image = Image.open(BytesIO(response.read()))

# Load model, image preprocessor
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTForMaskedImageModeling.from_pretrained("google/vit-base-patch16-224-in21k")

# 이미치 패치 개수 계산 및 전처리
num_patches = (model.config.image_size // model.config.patch_size) ** 2
pixel_values = image_processor(images=image, return_tensors="pt").pixel_values

# create random boolean mask of shape (batch_size, num_patches)
bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()
"""
torch.randint(low=0, high=2)을 통해 0 또는 1을 무작위로 뽑는다.
size는 (1, num_patches)로 설정되며, .bool()을 통해 1은 True(가림), 0은 False(가리지 않음)로 변환
"""

outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)

# loss와 모델이 mask를 유추해서 만든 이미지 텐서 반환
loss, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
list(reconstructed_pixel_values.shape) # 원본 입력과 동일한 해상도로 복원되어 [1, 3, 224, 224]가 출력된다.
```

    /usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(
    


    preprocessor_config.json:   0%|          | 0.00/160 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/502 [00:00<?, ?B/s]


    Fast image processor class <class 'transformers.models.vit.image_processing_vit_fast.ViTImageProcessorFast'> is available for this model. Using slow image processor class. To use the fast image processor class set `use_fast=True`.
    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
    WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
    


    model.safetensors:   0%|          | 0.00/346M [00:00<?, ?B/s]



    Loading weights:   0%|          | 0/198 [00:00<?, ?it/s]


    ViTForMaskedImageModeling LOAD REPORT from: google/vit-base-patch16-224-in21k
    Key                       | Status     | 
    --------------------------+------------+-
    pooler.dense.bias         | UNEXPECTED | 
    pooler.dense.weight       | UNEXPECTED | 
    vit.embeddings.mask_token | MISSING    | 
    decoder.0.bias            | MISSING    | 
    decoder.0.weight          | MISSING    | 
    
    Notes:
    - UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
    - MISSING	:those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
    




    [1, 3, 224, 224]



## 5. ViTForImageClassification
ImageNet 등과 같은 이미지 분류를 위해, 최상단에 image classification head(`[CLS]` token의 최종 은닉 상태 위에 얹힌 linear layer)가 추가된 ViT Model transformer

모델의 순전파(forward) 과정에서 `interpolate_pos_encoding=True`로 설정하면, 모델이 학습했던 해상도보다 더 높은 해상도의 이미지로 ViT를 fine-tune하는 것이 가능. pre-trained position embeddings을 더 높은 해상도에 맞게 interpolate

### 5.1 class transformers.ViTForImageClassification
Parameters
- config (`ViTConfig`)

#### 5.1.1 forward
Parameters
- `pixel_values`
- `labels` (`(batch_size, )` 형태의 `torch.LongTensor`, optional): image classification/regression loss를 계산하기 위한 labels. 인덱스는 `[0, ..., config.num_labels - 1]` 범위 내에 있어야 한다. `config.num_labels == 1`이면 Mean-Square loss로 regression loss가 계산되고, `config.num_labels > 1`인 경우 Cross-Entropy로 classification loss가 계산된다.
- `interpolate_pos_encoding`

Returns
- `loss` ((1, ) 형태의 `torch.FloatTensor`): `bool_masked_pos`가 제공될 때 반환되는 reconstruction loss. 모델이 재구성한 픽셀들이 원본 이미지의 실제 픽셀과 얼마나 차이가 나는지 계산한 오차.
- `logits` (`(batch_size, config.num_labels)` 형태의 torch.FloatTensor): softMax 함수를 통과하기 전의 값
- `hidden_states`
- `attentions`


```python
from transformers import AutoImageProcessor, ViTForImageClassification
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```


    README.md:   0%|          | 0.00/96.0 [00:00<?, ?B/s]



    cats_image.jpeg:   0%|          | 0.00/173k [00:00<?, ?B/s]



    Generating test split:   0%|          | 0/1 [00:00<?, ? examples/s]



    preprocessor_config.json:   0%|          | 0.00/160 [00:00<?, ?B/s]



    config.json: 0.00B [00:00, ?B/s]


    Fast image processor class <class 'transformers.models.vit.image_processing_vit_fast.ViTImageProcessorFast'> is available for this model. Using slow image processor class. To use the fast image processor class set `use_fast=True`.
    


    model.safetensors:   0%|          | 0.00/346M [00:00<?, ?B/s]



    Loading weights:   0%|          | 0/200 [00:00<?, ?it/s]


    Egyptian cat
    


```python
# interpolate_pos_encoding=True

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

model_name = "google/vit-base-patch16-224"
image_processor = AutoImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# 더 높은 해상도(384x384)로 전처리 크기 강제 변경
inputs_high_res = image_processor(
    image,
    size={"height": 384, "width": 384}, # 해상도 224x224에 확대
    return_tensors="pt"
)

print(inputs_high_res['pixel_values'].shape) # [1, 3, 384, 384]
```

    Fast image processor class <class 'transformers.models.vit.image_processing_vit_fast.ViTImageProcessorFast'> is available for this model. Using slow image processor class. To use the fast image processor class set `use_fast=True`.
    


    Loading weights:   0%|          | 0/200 [00:00<?, ?it/s]


    torch.Size([1, 3, 384, 384])
    


```python
# interpolate 옵션을 켜고 모델 추론
with torch.no_grad():
    outputs = model(
        **inputs_high_res,
        interpolate_pos_encoding=True # interpolate_pos_encoding=True
    )

logits = outputs.logits
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

    remote control, remote
    
