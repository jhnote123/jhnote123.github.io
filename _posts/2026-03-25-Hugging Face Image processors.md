# Hugging Face Image processors


```python
# https://huggingface.co/docs/transformers/main/ko/image_processors
```

이미지 프로세서(Image processors)는 이미지를 픽셀 값, 즉 이미지의 색상과 크기를 나타내는 텐서로 변환한다. 이 픽셀 값들은 비전 모델의 입력값이 된다.

이미지 프로세서는 (새로 입력되는) 이미지가 사전학습된 모델(pretrained model)이 사전학습할 때 사용되었던 이미지들의 형태와 정확히 일치하도록 다음과 같은 연산을 수행할 수 있다.
- 이미지 크기를 조절하는 `center_crop()`
- 픽셀 값을 정규화하는 `normalize()` 또는 스케일을 재조정하는 `rescale()`

`from_pretrained()`를 사용하여 Hub 또는 로컬에 있는 비전 모델로부터 image processors의configuration (image size, normalize 및 rescale 여부 등)을 불러올 수 있다. pretrained model에 대한 configuration 값은 `preprocessor_config.json` 파일에 저장되어 있다.


```python
from transformers import AutoImageProcessor

# 1. configuration 파일(preprocessor_config.json) 로드 및 프로세서 인스턴스화
# # "google/vit-base-patch16-224" 모델이 학습될 때 사용했던 설정을 그대로 가져옴
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
```

    /usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(
    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
    WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
    


    preprocessor_config.json:   0%|          | 0.00/160 [00:00<?, ?B/s]



    config.json: 0.00B [00:00, ?B/s]


    Fast image processor class <class 'transformers.models.vit.image_processing_vit_fast.ViTImageProcessorFast'> is available for this model. Using slow image processor class. To use the fast image processor class set `use_fast=True`.
    

이미지를 이미지 프로세서에 전달하여 픽셀 값으로 변환하고, `return_tensors="pt"`를 설정하여 파이토치 텐서를 반환받을 수 있다.


```python
from PIL import Image
import requests

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/image_processor_example.png"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# 2. 프로세서를 통해 이미지를  pretrained model이 학습 시 사용한 설정에 맞게 전처리한 다음 텐서 형태로 반호나
inputs = image_processor(image, return_tensors="pt")
print(inputs["pixel_values"].shape) # [배치 사이즈, 채널 수(RGB=3), 높이, 너비]
```

    torch.Size([1, 3, 224, 224])
    


```python
image
```




    
![png](output_6_0.png)
    




```python
inputs
```




    {'pixel_values': tensor([[[[ 0.2471,  0.2157,  0.2392,  ...,  0.0510,  0.0980,  0.1059],
              [ 0.2314,  0.2392,  0.2392,  ...,  0.0510,  0.0902,  0.1059],
              [ 0.2235,  0.2235,  0.1843,  ...,  0.0902,  0.0980,  0.0902],
              ...,
              [-0.4039, -0.5294, -0.5216,  ..., -0.6235, -0.6392, -0.6549],
              [-0.3020, -0.5059, -0.5529,  ..., -0.6471, -0.6627, -0.6235],
              [-0.1922, -0.4039, -0.5137,  ..., -0.6627, -0.6706, -0.6235]],
    
             [[ 0.0510,  0.0510,  0.0980,  ..., -0.1059, -0.0824, -0.0745],
              [ 0.0431,  0.0745,  0.0824,  ..., -0.0902, -0.0902, -0.0588],
              [ 0.0196,  0.0431,  0.0510,  ..., -0.0902, -0.1059, -0.0902],
              ...,
              [ 0.3725,  0.3255,  0.3412,  ...,  0.1608,  0.1451,  0.1451],
              [ 0.4275,  0.3255,  0.3176,  ...,  0.1529,  0.1373,  0.1529],
              [ 0.4902,  0.3804,  0.3176,  ...,  0.1451,  0.1529,  0.1765]],
    
             [[-0.0275, -0.0510, -0.0275,  ..., -0.2000, -0.1922, -0.2078],
              [-0.0353, -0.0196, -0.0510,  ..., -0.2078, -0.2157, -0.2000],
              [-0.0510, -0.0588, -0.0980,  ..., -0.1608, -0.2000, -0.2157],
              ...,
              [ 0.9686,  0.9765,  0.9765,  ...,  0.9059,  0.8980,  0.8902],
              [ 0.9765,  0.9765,  0.9765,  ...,  0.8980,  0.8902,  0.8824],
              [ 0.9765,  0.9765,  0.9765,  ...,  0.8980,  0.9059,  0.8824]]]])}



## 1. Image processor classes

이미지 프로세서를 불러오는 방법은 `AutoImageProcessor`를 사용하거나 모델별 이미지 프로세서를 사용하는 방식 두 가지가 있다.

속도가 더 빠른 프로세서를 사용하고 싶다면 torchvision을 설치한 뒤, `use_fast=True`를 추가하면 된다. torchvision을 기반으로 하며, 특히 GPU에서 처리할 때 속도가 훨씬 빠르다. 단, 모델에 따라 지원 여부가 다르다.




```python
# AutoImageProcessor
from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", use_fast=True)

# model-specific image processor
from transformers import ViTImageProcessor

image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
```

`device` 파라미터를 사용해 어떤 장치에서 처리할지 지정할 수 있다.
- 만약 입력값이 텐서라면 그 텐서와 동일한 장치에서
- 그렇지 않은 경우 기본적으로 CPU에서 처리된다.


```python
from torchvision.io import read_image
from transformers import DetrImageProcessorFast

processor = DetrImageProcessorFast.from_pretrained("facebook/detr-resnet-50")
images_processed = processor(image, return_tensors="pt", device="cuda")
```

## 2. 전처리(Preprocess)

`Transformers`의 비전 모델은 입력값으로 파이토치 텐서 형태의 픽셀 값을 받는다.

이미지 프로세서는 **이미지를 바로 이 픽셀 값 텐서(batch size, number of channels, height, width)로 변환**하는 역할을 한다. 이 과정에서 모델이 요구하는 크기로 이미지를 조절하고, 픽셀 값 또한 모델 기준에 맞춰 정규화하거나 재조정한다. 이를 통해 pretrained model이 요구하는 input format을 맞춰줄 수 있다.

일반적으로 모델 성능을 높이기 위해, 보통 이미지는 증강 과정을 거친 뒤 전처리되어 모델에 입력된다. 이때 증강은  Albumentations, Kornia와 같은 라이브러리를 사용할 수 있으며, 이후 전처리 단계에서 이미지 프로세서를 사용하면 된다.


```python
from datasets import load_dataset

dataset = load_dataset("ethz/food101", split="train[:100]")
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
    


    data/train-00000-of-00008.parquet:   0%|          | 0.00/490M [00:00<?, ?B/s]



    data/train-00001-of-00008.parquet:   0%|          | 0.00/464M [00:00<?, ?B/s]



    data/train-00002-of-00008.parquet:   0%|          | 0.00/472M [00:00<?, ?B/s]



    data/train-00003-of-00008.parquet:   0%|          | 0.00/464M [00:00<?, ?B/s]



    data/train-00004-of-00008.parquet:   0%|          | 0.00/475M [00:00<?, ?B/s]



    data/train-00005-of-00008.parquet:   0%|          | 0.00/470M [00:00<?, ?B/s]



    data/train-00006-of-00008.parquet:   0%|          | 0.00/478M [00:00<?, ?B/s]



    data/train-00007-of-00008.parquet:   0%|          | 0.00/486M [00:00<?, ?B/s]



    data/validation-00000-of-00003.parquet:   0%|          | 0.00/423M [00:00<?, ?B/s]



    data/validation-00001-of-00003.parquet:   0%|          | 0.00/413M [00:00<?, ?B/s]



    data/validation-00002-of-00003.parquet:   0%|          | 0.00/426M [00:00<?, ?B/s]



    Generating train split:   0%|          | 0/75750 [00:00<?, ? examples/s]



    Generating validation split:   0%|          | 0/25250 [00:00<?, ? examples/s]



```python
# torchvision의 transforms 모듈을 사용해 이미지 증강

from torchvision.transforms import RandomResizedCrop, ColorJitter, Compose
from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

size = (
    image_processor.size["shortest_edge"] # shortest_edge: 가장 짧은변
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)

```


    preprocessor_config.json:   0%|          | 0.00/160 [00:00<?, ?B/s]



    config.json: 0.00B [00:00, ?B/s]


    Fast image processor class <class 'transformers.models.vit.image_processing_vit_fast.ViTImageProcessorFast'> is available for this model. Using slow image processor class. To use the fast image processor class set `use_fast=True`.
    

`torchvision.transforms`의 `Compose`API는 여러 변환(transform)을 하나로 묶어주는 역할을 한다.

잘라낼 이미지의 크기는 위와 같이 이미지 프로세서에서 가져올 수 있다. 모델에 따라 정확한 높이와 너비가 필요할 때도 있고, 가장 짧은 변(*shortest_edge*) 값만 필요할 때도 있다.


```python
_transforms = Compose([
    RandomResizedCrop(size), # 이미지를 무작위로 자르고 리사이즈
    ColorJitter(brightness=0.5, hue=0.5) # 색상을 무작위로 변경
    ])
```

아래 함수는 준비된 변환 값들(_transforms)을 이미지에 적용하고, RGB 형식으로 바꿔주는 기능을 수행한다. 그 다음, 이렇게 증강된 이미지를 이미지 프로세서에 넣어 픽셀 값을 반환한다.


```python
def transforms(examples):
    images = [_transforms(img.convert("RGB")) for img in examples["image"]]
    examples["pixel_values"] = image_processor(images, do_resize=False, return_tensors="pt")["pixel_values"]
    return examples
```

`set_transform`을 사용하면 증강 및 전처리 기능이 결합된 transforms 함수를 전체 데이터셋에 on-the-fly 방식으로 적용할 수 있다.


```python
dataset.set_transform(transforms)
```


```python
import numpy as np
import matplotlib.pyplot as plt

img = dataset[0]["pixel_values"]
plt.imshow(img.permute(1, 2, 0))
```

    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.0..1.0].
    




    <matplotlib.image.AxesImage at 0x7de23338e5d0>




    
![png](output_21_2.png)
    


이미지 프로세서는 이러한 전처리뿐만 아니라, object detection이나 segmentation과 같은 vision tasks에서 모델의 결과값을 bounding box나 segmentation map처럼 의미 있는 예측으로 바꿔주는 후처리 기능도 갖추고 있다.

## 3. Padding

DETR과 같은 일부 모델은 학습 중에 scale augmentation을 사용하기 때문에, 이로 인해 하나의 배치 내에 있는 이미지들이 서로 다른 크기를 가질 수 있다.

서로 다른 크기의 이미지들은 함께 배치로 묶을 수 없다. 이 문제는 special padding token으로 이미지들을 패딩하여 해결할 수 있다. 다음과 같이 custom collate function을 정의하여 사용하면 된다.


```python
def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch] # 각 이미지 텐서들을 하나의 리스트로

    # image_processor.pad()는 리스트 내 텐서들을 보고, 패딩을 추가한 텐서로 변환한다.
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch
```
