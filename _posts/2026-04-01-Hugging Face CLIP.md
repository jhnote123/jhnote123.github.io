# Hugging Face Contrastive Language-Image Pre-Training (CLIP)

CLIP은 컴퓨터 비전 모델을 학습시킬 때 고정된 수의 카테고리(class 개수)에 국한되는 한계를 해결하기 위해 등장한 멀티모달 모델이다.

4억 개의 (image, text) 쌍으로 사전학습되었으며, 덕분에 downstream tasks로의 zero-shot transfer가 가능하다.

visual features과 text features을 얻기 위해 image encoder와 text encoder를 사용한다. 두 features을 모두 동일한 차원의 latent space로 투영한 다음, 이들의 dot product를 통해 similarity score를 계산한다.



```python
# Pipeline을 이용한 zero-shot classification
import torch
from transformers import pipeline

clip = pipeline(
   task="zero-shot-image-classification",
   model="openai/clip-vit-base-patch32",
   dtype=torch.bfloat16,
   device=0
)
labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]
clip("http://images.cocodataset.org/val2017/000000039769.jpg", candidate_labels=labels)
```

    /usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(
    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
    WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
    


    config.json: 0.00B [00:00, ?B/s]



    pytorch_model.bin:   0%|          | 0.00/605M [00:00<?, ?B/s]



    Loading weights:   0%|          | 0/398 [00:00<?, ?it/s]



    model.safetensors:   0%|          | 0.00/605M [00:00<?, ?B/s]


    CLIPModel LOAD REPORT from: openai/clip-vit-base-patch32
    Key                                  | Status     |  | 
    -------------------------------------+------------+--+-
    vision_model.embeddings.position_ids | UNEXPECTED |  | 
    text_model.embeddings.position_ids   | UNEXPECTED |  | 
    
    Notes:
    - UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
    


    tokenizer_config.json:   0%|          | 0.00/592 [00:00<?, ?B/s]



    vocab.json: 0.00B [00:00, ?B/s]



    merges.txt: 0.00B [00:00, ?B/s]



    tokenizer.json: 0.00B [00:00, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/389 [00:00<?, ?B/s]



    preprocessor_config.json:   0%|          | 0.00/316 [00:00<?, ?B/s]


    The image processor of type `CLIPImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`. 
    




    [{'score': 0.9921875, 'label': 'a photo of a cat'},
     {'score': 0.005218505859375, 'label': 'a photo of a car'},
     {'score': 0.0027923583984375, 'label': 'a photo of a dog'}]




```python
# AutoModel을 이용한 zero-shot classification
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

model = AutoModel.from_pretrained("openai/clip-vit-base-patch32", dtype=torch.bfloat16, attn_implementation="sdpa")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # image-text similarity score
probs = logits_per_image.softmax(dim=1)
most_likely_idx = probs.argmax(dim=1).item()
most_likely_label = labels[most_likely_idx]
print(f"Most likely label: {most_likely_label} with probability: {probs[0][most_likely_idx].item():.3f}")
```


    Loading weights:   0%|          | 0/398 [00:00<?, ?it/s]


    CLIPModel LOAD REPORT from: openai/clip-vit-base-patch32
    Key                                  | Status     |  | 
    -------------------------------------+------------+--+-
    vision_model.embeddings.position_ids | UNEXPECTED |  | 
    text_model.embeddings.position_ids   | UNEXPECTED |  | 
    
    Notes:
    - UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
    

    Most likely label: a photo of a cat with probability: 0.992
    

## 1. CLIPConfig

### 1.1 class transformers.CLIPConfig
Parameters
- **text_config** (`dict`, optional): `CLIPTextConfig`를 초기화하는 데 사용되는 configuration options의 딕셔너리
- **vision_config** (`dict`, optional): `CLIPVisionConfig`를 초기화하는 데 사용되는 configuration options의 딕셔너리  
- **projection_dim** (`int`, optional, defaults to `512`): text 및 vision projection layers의 차
- **logit_scale_init_value** (`float | int`, optional, defaults to 2.6592): `logit_scale` 파라미터의 초깃값
- **initializer_factor** (`float`, optional, defaults to `1.0`): 모든 가중치 행렬을 초기화하기 위한 비율 계수. default는 가중치 크기 범위를 1.0으로 조절

기본값으로 config를 인스턴스화하면 `openai/clip-vit-base-patch32` 모델의 설정과 유사한 설정으로 생성된다.


```python
from transformers import CLIPConfig, CLIPModel

# Initializing a CLIPConfig with openai/clip-vit-base-patch32 style configuration
configuration = CLIPConfig()

# Initializing a CLIPModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
model = CLIPModel(configuration)

# Accessing the model configuration
configuration = model.config
```


```python
# We can also initialize a CLIPConfig from a CLIPTextConfig and a CLIPVisionConfig
from transformers import CLIPTextConfig, CLIPVisionConfig

# Initializing a CLIPText and CLIPVision configuration
config_text = CLIPTextConfig()
config_vision = CLIPVisionConfig()

config = CLIPConfig(text_config=config_text, vision_config=config_vision)
```

## 2. CLIPTextConfig

CLIP 모델의 텍스트 부분의 configuration에 대한 클래스이다. 기본값으로 인스턴스화하면 `openai/clip-vit-base-patch32` 모델의 config와 유사한 설정으로 생성된다.

### 2.1 class transformers.CLIPTextConfig
Parameters
- **vocab_size** (`int`, optional, defaults to `49408`): vocabulary size
- **hidden_size** (`int`, optional, defaults to `512`): hidden representations의 차원
- **intermediate_size** (`int`, optional, defaults to `2048`): MLP representations의 차원
- **projection_dim** (`int`, optional, defaults to `512`): text 및 vision projection layers의 차원
- **num_hidden_layers** (`int`, optional, defaults to `12`): Transformer decoder hidden layers의 개수
- **num_attention_heads** (`int`, optional, defaults to `8`): Transformer decoder의 각 attention layer에 있는 attention heads의 개수
- **max_position_embeddings** (`int`, optional, defaults to `77`): maximum sequence length
- **hidden_act** (`str`, optional, defaults to `quick_gelu`): decoder 내부의 non-linear activation function
- **layer_norm_eps** (`float`, optional, defaults to `1e-05`): layer normalization에서 사용되는 epsilon 값
- **attention_dropout** (optional, defaults to `0.0`): attention probabilities에 대한 dropout ratio
- **initializer_range** (`float`, optional, defaults to `0.02`)
- **initializer_factor** (`float`, optional, defaults to `1.0`)
- **pad_token_id** (`int`, optional, defaults to `1`)
- **bos_token_id** (`int`, optional, defaults to `49406`)
- **eos_token_id** (optional, defaults to `49407`)

## 3. CLIPVisionConfig
### 3.1 class transformers.CLIPVisionConfig
Parameters
- **hidden_size** (`int`, optional, defaults to `768`)
- **intermediate_size** (`int`, optional, defaults to `3072`)
- **projection_dim** (`int`, optional, defaults to `512`)
- **num_hidden_layers** (`int`, optional, defaults to `12`)
- **num_attention_heads** (`int`, optional, defaults to `12`)
- **num_channels** (`int`, optional, defaults to `3`): input channels의 수
- **image_size** (`int`, optional, defaults to `224`): image size(resolution)
- **patch_size** (`int`, optional, defaults to `32`): patch size(resolution)
- **hidden_act** (`str`, optional, defaults to `quick_gelu`)
- **layer_norm_eps** (`float`, optional, defaults to `1e-05`)
- **attention_dropout** (optional, defaults to `0.0`)
- **initializer_range** (`float`, optional, defaults to `0.02`)
- **initializer_factor** (`float`, optional, defaults to `1.0`)

## 4. CLIPTokenizer

byte-level BPE를 기반으로 하는 tokenizer

### 4.1 class transformers.CLIPTokenizer
Parameters
- **vocab** (`str`, `dict` or `list`, optional): tokenizer에 사용할 vocabulary
- **merges** (`str` or `list`, optional): BPE tokenizer에서 사용할 Merges list
- **unk_token** (`str`, optional, defaults to `"<|endoftext|>"`): unknown token. vocabulary에 없는 token은 unk token으로 설정된다.
- **bos_token** (`str`, optional, defaults to `"<|startoftext|>"`)
- **eos_token** (`str`, optional, defaults to `"<|endoftext|>"`)
- **pad_token** (`str`, optional, defaults to `"<|endoftext|>"`)

#### 4.1.1 get_special_tokens_mask
Parameters
- **token_ids_0**: (이미 format이 맞춰졌을 수 있는) sequence의 IDs 리스
- **token_ids_1**: `already_has_special_tokens=True`일 때는 사용되지 않으며, 반드시 `None`이어야 한다.
- **already_has_special_tokens**: sequence에 이미 special tokens이 포함되어 format이 맞춰졌는지 여부

Returns
- special token인 경우 1, 일반 sequence token인 경우 0으로 이루어진 리스트를 반환한다.

#### 4.1.2 save_vocabulary
- **save_directory** (`str`)
- **filename_prefix** (`str` | `None = None`)

## 5. CLIPImageProcessor

### 5.1 class transformers.CLIPImageProcessor
#### 5.1.1 preprocess
Parameters
- **images**: 전처리할 이미지. 0에서 255 사이의 픽셀 값을 가진 single image 또는 images의 배치. 만약 픽셀 값이 0과 1 사이로 맞춰진 image를 전달한다면, `do_rescale=False`로 설정
- **return_tensors**: `pt`로 설정하면 stacked tensors 반환하고, 그렇지 않으면 tensors의 리스트 반환
- **kwargs**

Returns
- **data** (`dict`)
- **tensor_type**

## 6. CLIPProcessor

텍스트와 이미지를 입력하면 `CLIPProcessor`는 `CLIPImageProcessor`와 `CLIPTokenizer`를 하나의 processor로 래핑하여 입력으로 들어온 텍스트와 이미지를 처리한다.

`CLIPImageProcessor`와 `CLIPTokenizer`의 모든 기능을 사용할 수 있다.

### 6.1 class transformers.CLIPProcessor
Parameters
- **image_processor** (`CLIPImageProcessor`)
- **tokenizer** (`CLIPTokenizer`)

#### 6.1.1 _ _call_ _
Parameters
- **images**: PIL 이미지, Numpy 배열 또는 PyTorch 텐서일 수 있다.
- **text**: 인코딩할 시퀀스
- **videos**: 4D Numpy 배열, yTorch 텐서 또는 3D 프레임의 nested list 형태일 수 있다.
- **audio**: 오디오 배열 또는 텐서
- **return_tensors**: `pt`로 설정하면 PyTorch의 `torch.Tensor` 객체 반환. `np`로 설정하면 NumPy의 `np.ndarray` 객체 반환.

Returns
- 처리된 입력값들이 딕셔너리 형태로 담긴 `BatchFeature` 객체

## 7. CLIPModel

### 7.1 class transformers.CLIPModel
Parameters
- **config** (`CLIPConfig`)

최상단에 specific head가 없는  bare Clip 모델

#### 7.1.1 forward
Parameters
- **input_ids** (shape: `(batch_size, sequence_length)`, `torch.LongTensor`, optional): vocabulary에 있는 input sequence tokens의 인덱스
- **pixel_values** (shape: `(batch_size, num_channels, image_size, image_size)`, `torch.FloatTensor`, optional): input images에 해당하는 tensors. `CLIPImageProcessor`를 통해 얻을 수 있다.
- **attention_mask** (shape: `(batch_size, sequence_length)`, `torch.Tensor`, optional): padding token 인덱스에 대해 어텐션 연산을 수행하지 않도록 피하는 마스크. `[0, 1]` 범위에서 선택된다. (1: 마스킹되지 않음, 0: 마스킹됨)
- **position_id** (shape: `(batch_size, sequence_length)`, `torch.LongTensor`, optional): position embeddings에서 각 input sequence tokens의 인덱스
- **return_loss** (`bool`, optional): contrastive loss를 반환할지 여
- **interpolate_pos_encoding** (`bool`, optional, defaults to `False`): pre-trained position embeddings을 interpolate할지 여부

Returns
- **loss** (shape: `(1, )`, `torch.FloatTensor`, optional): `return_loss=True`일 때 반환. image-text similarity에 대한 contrastive loss
- **logits_per_image** (shape: `(image_batch_size, text_batch_size)`, `torch.FloatTensor`, optional): `image_embeds`와 `text_embeds` 사이의 스켕일이 적용된 dot product scores. 이는 text-image similarity scores이다.
- **logits_per_text** (shape: `(text_batch_size, image_batch_size)`, `torch.FloatTensor`, optional): `text_embeds`와 `image_embeds` 사이의 스켕일이 적용된 dot product scores. 이는 text-image similarity scores이다.
- **text_embeds** (shape: `(batch_size, output_dim)`, `torch.FloatTensor`, optional): `CLIPTextModel`의 pooled output에 projection layer를 적용하여 얻은 text embeddings
- **image_embeds** (shape: `(batch_size, output_dim)`, `torch.FloatTensor`, optional): `CLIPVisionModel`의 pooled output에 projection layer를 적용하여 얻은 image embeddings
- **text_model_output**과 **vision_model_output**: projection layer 투영 전 단계에서의 `CLIPTextModel`, `CLIPVisionModel`의 output


```python
import torch
from transformers import AutoProcessor, CLIPModel
from transformers.image_utils import load_image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)

inputs = processor(
    text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
)

with torch.inference_mode():
    outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
```


    Loading weights:   0%|          | 0/398 [00:00<?, ?it/s]


    CLIPModel LOAD REPORT from: openai/clip-vit-base-patch32
    Key                                  | Status     |  | 
    -------------------------------------+------------+--+-
    text_model.embeddings.position_ids   | UNEXPECTED |  | 
    vision_model.embeddings.position_ids | UNEXPECTED |  | 
    
    Notes:
    - UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
    

#### 7.1.2 get_text_features

텐서 형태의 텍스트 데이터를 모델의 텍스트 인코더에 입력할 때 사용하는 메서드

Parameters
- **input_ids** (shape: `(batch_size, sequence_length)`, `torch.Tensor`):
- **attention_mask**
- **position_ids**

Returns
- **last_hidden_state**
- **pooler_output**
- **hidden_states**
- **attentions**



```python
import torch
from transformers import AutoTokenizer, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

with torch.inference_mode():
    text_features = model.get_text_features(**inputs)
```

    /usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(
    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
    WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
    


    config.json: 0.00B [00:00, ?B/s]



    pytorch_model.bin:   0%|          | 0.00/605M [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/605M [00:00<?, ?B/s]



    Loading weights:   0%|          | 0/398 [00:00<?, ?it/s]


    CLIPModel LOAD REPORT from: openai/clip-vit-base-patch32
    Key                                  | Status     |  | 
    -------------------------------------+------------+--+-
    vision_model.embeddings.position_ids | UNEXPECTED |  | 
    text_model.embeddings.position_ids   | UNEXPECTED |  | 
    
    Notes:
    - UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
    


    tokenizer_config.json:   0%|          | 0.00/592 [00:00<?, ?B/s]



    vocab.json: 0.00B [00:00, ?B/s]



    merges.txt: 0.00B [00:00, ?B/s]



    tokenizer.json: 0.00B [00:00, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/389 [00:00<?, ?B/s]


#### 7.1.3 get_image_features
Parameters
- **pixel_values** (shape: `(batch_size, num_channels, image_size, image_size)`, `torch.FloatTensor`): input images에 해당하는 tensors. `CLIPImageProcessor`를 사용하여 얻을 수 있다.
- **interpolate_pos_encoding** (`bool`, optional, defaults to `False`)

Returns
- **last_hidden_state** (shape: `(batch_size, sequence_length, hidden_size)`, `torch.FloatTensor`): 모델의 last layer에서 출력된 hidden states
- **pooler_output** (shape: `(batch_size, hidden_size)`, `torch.FloatTensor`): 첫 번째 토큰([CLS] 토큰 등)의 last layer hidden state
- **hidden_states**
- **attentions**


```python
import torch
from transformers import AutoProcessor, CLIPModel
from transformers.image_utils import load_image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)

inputs = processor(images=image, return_tensors="pt")

with torch.inference_mode():
    image_features = model.get_image_features(**inputs)
```


    Loading weights:   0%|          | 0/398 [00:00<?, ?it/s]


    CLIPModel LOAD REPORT from: openai/clip-vit-base-patch32
    Key                                  | Status     |  | 
    -------------------------------------+------------+--+-
    vision_model.embeddings.position_ids | UNEXPECTED |  | 
    text_model.embeddings.position_ids   | UNEXPECTED |  | 
    
    Notes:
    - UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
    


    preprocessor_config.json:   0%|          | 0.00/316 [00:00<?, ?B/s]


    The image processor of type `CLIPImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`. 
    

## 8. CLIPTextModel

모델 최상단에 어떠한 head나 projection layer가 없는 CLIP의 text model

### 8.1 class transformers.CLIPTextModel
Parameters
- config (`CLIPTextConfig`)

#### 8.1.1 forward
Parameters
- **input_ids** (shape: `(batch_size, sequence_length)`, `torch.Tensor`, optional): vocabulary에 있는 input sequence tokens의 인덱스
- **attention_mask** (shape: `(batch_size, sequence_length)`, `torch.Tensor`, optional): padding token 인덱스에 대해 어텐션 연산을 수행하지 않도록 피하는 마스크. (1: 마스킹되지 않은 토큰, 0: 마스킹된 토큰)
- **position_ids** (shape: `(batch_size, sequence_length)`, `torch.Tensor`, optional): position embeddings에서 각 input sequence tokens의 위치 인덱스

Returns
- **last_hidden_state** (shape: `(batch_size, sequence_length, hidden_size)`, `torch.FloatTensor`)
- **pooler_output** (shape: `(batch_size, hidden_size)`, `torch.FloatTensor`)
- **hidden_states**
- **attentions**


```python
from transformers import AutoTokenizer, CLIPTextModel

model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state
pooled_output = outputs.pooler_output  # pooled (EOS token) states
pooled_output.shape # [2, 512]
```

`CLIPTextModel`외에 `CLIPVisionModel`도 있으며, 최상단에 projection layer(pooled output 위에 linear layer)가 추가된 `CLIPTextModelWithProjection`, `CLIPVisionModelWithProjection`도 있다.

## 9. CLIPForImageClassification

모델 최상단에 image classification head(patch tokens의 final hidden states 위에 얹힌 linear layer)가 포함된 CLIP vision encoder이다.

### 9.1 class transformers.CLIPForImageClassification
Parameters
- **config** (`CLIPConfig`)

#### 9.1.1 forward
Parameters
- **pixel_values** (shape: `(batch_size, num_channels, image_size, image_size)`, `torch.Tensor`, optional): input images에 대한 tensors. `CLIPImageProcessor`를 사용하여 얻을 수 있다.
- **labels** (shape: `(batch_size,)`, torch.LongTensor, optional): image classification/regression loss를 계산하기 위한 labels. `config.num_labels == 1`이면 regression loss, `config.num_labels > 1`이면 classification loss

Returns
- **loss**
- **hidden_states**
- **attentions**


```python
from transformers import AutoImageProcessor, CLIPForImageClassification
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPForImageClassification.from_pretrained("openai/clip-vit-base-patch32")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
#print(model.config.id2label[predicted_label])
```
