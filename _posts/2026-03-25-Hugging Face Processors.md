---
title: "Hugging Face Processors"
date: 2026-03-25 15:00:00 +0900
categories: [Hugging Face]
tags: [processors, huggingface]
---

# Hugging Face Processors

```python
# https://huggingface.co/docs/transformers/main_classes/processors
```

## 1. Processors

Transformers 라이브러리에서 processors는 두 가지 다른 의미를 가질 수 있다:
- Wav2Vec2 (speech and text) 또는 CLIP (text and vision)과 같은 **multi-modal models을 위해 입력값을 전처리하는 객체**
- 라이브러리 구버전에서 GLUE나 SQUAD 데이터셋을 전처리하는 데 사용된 사용 중단된 객체

## 2. Multi-modal processors

모든 multi-modal model은 여러 modalities (text, visio, audio 등)가 그룹화된 데이터를 인코딩하거나 디코딩할 객체가 필요하다.

이 역할은 **processors**라고 불리는 객체들이 담당한다. text modality를 위한 tokenizers, vision을 위한 image processors, audio를 위한 feature extractors 등 두 개 이상의 처리 객체들을 하나로 묶어준다.

이러한 processors은 다음과 같은 class를 상속받는다.

### 2.1 class transformers.ProcessorMixin

이 클래스는 모든 processor classes에 저장 및 불러오기 기능을 제공하기 위해 사용된다.

#### (1) `apply_chat_template`
- 파라미터
  - conversation: 포맷팅할 대화 내용
  - chat_template: conversation을 포맷팅할 때 사용할 Jinja 템플릿

- 토크나이저에 있는 `apply_chat_template` 메서드와 유사하게, 이 메서드는 입력된 conversations에 Jinja 템플릿을 적용하여 토큰화할 수 있는 단일 문자열로 변환한다.
- input은 다음과 같은 형식이어야 한다


```python
conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://www.ilankelman.org/stopsigns/australia.jpg"
            },
            {
                "type": "text",
                "text": "Please describe this image in detail."
            },
        ],
    },
]
```

여기서 각 메시지의 content는 **텍스트와 (선택적으로) 이미지 또는 비디오 입력으로 구성된 리스트이다.
- 이 예시의 경우, content 안에 `[{"type": "image"}, {"type": "text"}]`로 텍스트와 이미지를 묶어서 제공한다.

이미지, 비디오, URL 또는 로컬 경로를 넣을 수 있으며, 이는 return_dict=True일 때 모델이 인식하는 이미지 텐서인 pixel_values를 구성하는 데 사용된다.

만약 미디어가 제공되지 않으면, 포맷된 텍스트만 반환받게 된다.

#### (2) `batch_decode`
- 이 메서드는 받은 모든 인자들을 사전학습된 토크나이저(PreTrainedTokenizer)의 `batch_decode()` 메서드로 그대로 전달한다.

#### (3) `decode`
- decode도 받은 모든 안지달을 사전학습된 토크나이저의 `decode()` 메서드로 그대로 전달한다.

#### (4) `from_args_and_dict`
- 파라미터들이 담긴 딕셔너리로부터 ProcessingMixin 객체를 생성한다.
- 파라미터
  - processor_dict: 프로세서 객체를 생성하는 데 사용될 딕셔너리. `to_dict` 메서드로 사전학습된 체크포인트에서 가져올 수 있다.
  - kwargs: 프로세서를 초기화할 때 추가로 넣을 파라미터들
- 이 파라미터들을 바탕으로 만들어진 프로세서 객체를 반환

#### (5) `from_pretrained`
- pretrained model과 연결된 processor를 생성
- 파라미터
  - pretrained_model_name_or_path: 다음 중 하나: string(model ID)/ directory path / path or url(저장된 JSON config file 경로)
- 이 메서드는 단순히 feature extractor, image processor, tokenizer의 `from_pretrained()` 메서드를 한꺼번에 호출하는 역할




```python
from transformers import AutoProcessor

# CLIP 모델
# 단 한 줄로 tokenizer와 image processor를 동시에 다운로드
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
```

    /usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(
    


    preprocessor_config.json:   0%|          | 0.00/316 [00:00<?, ?B/s]


    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
    WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
    


    tokenizer_config.json:   0%|          | 0.00/592 [00:00<?, ?B/s]



    config.json: 0.00B [00:00, ?B/s]


    The image processor of type `CLIPImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`. 
    


    vocab.json: 0.00B [00:00, ?B/s]



    merges.txt: 0.00B [00:00, ?B/s]



    tokenizer.json: 0.00B [00:00, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/389 [00:00<?, ?B/s]


#### (6) `get_processor_dict`
- pretrained_model_name_or_path로부터, from_args_and_dict를 사용하여 ProcessingMixin 타입의 프로세서를 생성하는 데 사용될 파라미터들의 딕셔너리를 찾아낸다.

#### (7) `post_process_image_text_to_text`
- VLM(Vision-Language Model)의 출력값을 후처리하여 텍스트로 디코딩한다.
- 파라미터
  - generated_outputs (torch.Tensor or np.ndarray): 모델의 `generate` 함수를 통해 얻은 output. `(batch_size, sequence_length)` 또는 `(sequence_length,)` 형태의 tensor
  - skip_special_tokens (default=True): output에서 special tokens 제거 여부. 이 값은 내장된 토크나이저의 `decode` 메서드에 그대로 전달된다.
  - **kwargs:  토크나이저의 `decode` 메서드에 추가로 넘겨줄 기타 설정값들

#### (8) `post_process_multimodal_output`
- multimodal model의 output을 후처리하여 요청한 modality의 output을 반환한다.
- 파라미터
  - generated_outputs: post_process_image_text_to_text와 동일
  - skip_special_tokens: post_process_image_text_to_text와 동일. 토크나이저의 `batch_decode` 메서드로 전달
  - generation_mode (str, optional): 출력할 modality를 지정하는 generation mode. `["text", "image", "audio"]` 중 하나
  - **kwargs: 토크나이저의 `batch_decode` 메서드에 추가로 넘겨줄 인자들


#### (9) `push_to_hub`

#### (10) `register_for_auto_class`
- processor를 AutoProcessor 시스템에 등록할 때 사용

#### (11) `save_pretrained`
- 나중에 `from_pretrained()`를 통해 다시 불러올 수 있도록, processor의 attributes (feature extractor, tokenizer…)을 지정된 디렉토리에 저장

#### (12) `to_dict` / `to_json_file` / `to_json_string`
- 프로세서 인스턴스를 딕셔너리/json file/json string으로 변환




## 3. Processing kwargs
processor의 `__call__` 메서드는 modality별로 필요한 설정값들을 keyword arguments (kwargs)로 받아서 처리한다. 이때 어떤 설정값들을 넣을 수 있는지 미리 약속해둔 가이드라인으로 TypedDict 클래스를 사용한다.

TypedDict 클래스에서 각 modality에 대해 사용 가능한 keyword arguments을 정의하여 사용한다.

model-specific processors은 이러한 클래스들을 상속받아 자신만의 fields을 추가하거나 기존 설정을 override할 수 있다.


### 3.1 class transformers.ProcessorMixin
`transformers.ProcessingKwargs`는 프로세서에 전달되는 kwargs을 위한 기본 클래스이다.

만약 모델이 기본 클래스에 없는 specific kwargs을 요구하거나 기존 key에 대해 다른 default values을 가져야 하는 경우, `ProcessingKwargs`를 상속받는 `ModelProcessorKwargs` 클래스를 만들어 다음 두 가지를 제공해야 한다.
- (1) 모델이 입력을 처리하는 데 필요한 추가적인 typed keys
- (2) `_defaults` 속성 아래에 정의된 기존 keys의 default values

새로운 옵션을 추가할 경우, 다음과 같이 type hinting을 명확하게 달아서 클래스를 정의하면 된다.


```python
# adding a new image kwarg for this model
class ModelImagesKwargs(ImagesKwargs, total=False):
    new_image_kwarg: Optional[bool]

class ModelProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: ModelImagesKwargs
    _defaults = {
        "images_kwargs": {
            "new_image_kwarg": False
        },
        "text_kwargs": {
            "padding": "max_length"
        }
    }
```
