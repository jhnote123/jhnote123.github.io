# Introduction to Vision Language Models


```python
# https://huggingface.co/learn/smol-course/unit3/1?utm_source=chatgpt.com
```

Vision Language Models (VLMs)은 이미지와 텍스트를 동시에 이해할 수 있으며, image captioning, visual question answering (VQA), multimodal reasoning과 같은 다양한 tasks을 수행할 수 있다.

VLM은 LLM처럼 next token prediction으로 학습되며, 시각적 정보를 처리하는 능력이 추가된 모델이다.


## 1. What are Vision Language Models?

**일반적인 VLM 아키텍처**는 시각적 특징을 추출하는 **이미지 인코더(image encoder)**, 시각적 표현과 텍스트 표현을 align하는 **프로젝션 레이어(projection layer)**, 그리고 텍스트를 처리하거나 생성하는 **언어 모델(language model)**로 구성된다. 이를 통해 VLM은 시각적 요소와 언어적 개념을 연결할 수 있다.


## 2. Adapting Vision Language Models for specific needs
VLM을 fine-tuning한다는 것은 pre-trained model을 dataset이나 task에 맞게 조정하는 것을 의미한다.

core tools과 techniques은 LLM에 사용되는 것들과 크게 다르지 않다. 다만  VLM에서는 data representation이 특히 중요하다.
- 모델이 시각적 정보와 텍스트 정보를 효과적으로 결합할 수 있도록 적절한 이미지를 구성해야 한다.  
- 또 다른 중요한 요소는 모델 크기이다. VLM은 종종 LLM보다 훨씬 크기 떄문에 효율성이 매우 중요하다.

## 3. Evaluating Vision Language Models

VLM 평가에 널리 사용되는 benchmarks은 다음과 같다.
- MMMU & MMMU-Pro: 예술, 과학, 공학 등 다양한 도메인에 걸친 reasoning을 요구하는 대규모 벤치마크
- MMBench: OCR, localization, reasoning과 같은 skills을 테스트하는 3,000개 이상의 single-choice questions로 구성된 벤치마크
- MMT-Bench: recognition, localization, reasoning, planning을 포함한 expert-level의 multimodal tasks에 초점을 둔 벤치마크  

전문적인 skills을 테스트하기 위해 설계된 domain-specific benchmarks도 존재한다.
- MathVista: 이미지의 context 내에서 mathematical reasoning 평가
- AI2D: 다이어그램(diagram) 이해에 대한 평가
- ScienceQA: 과학 분야의 question answering
- OCRBench: document understanding와 OCR capabilities

마지막으로, 간소화된 평가 워크플로우를 위한 OpenVLM Leaderboard가 있으며, 명령어 하나로 여러 벤치마크에 걸쳐 VLM을 평가할 수 있는 toolkit을 제공한다.
