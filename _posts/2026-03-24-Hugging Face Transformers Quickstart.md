---
title: "Hugging Face Transformers Quickstart"
date: 2026-03-24 20:00:00 +0900
categories: [Hugging Face]
tags: [transformers, huggingface, quickstart]
---


# Hugging Face Transformers Quickstart


```python
# https://huggingface.co/docs/transformers/en/quicktour
```


```python
#!pip install -U transformers datasets evaluate accelerate timm
```


```python
from huggingface_hub import notebook_login

notebook_login()
```

## 1. Pretrained models


```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
```


    config.json:   0%|          | 0.00/684 [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/3.09G [00:00<?, ?B/s]



    Loading weights:   0%|          | 0/338 [00:00<?, ?it/s]



    generation_config.json:   0%|          | 0.00/138 [00:00<?, ?B/s]



    tokenizer_config.json: 0.00B [00:00, ?B/s]



    vocab.json: 0.00B [00:00, ?B/s]



    merges.txt: 0.00B [00:00, ?B/s]



    tokenizer.json: 0.00B [00:00, ?B/s]


model과 preprocessor를 불러올 때 `AutoClass` API를 사용하면, pretrained weights와 configuration file의 이름 또는 경로를 기반으로, 각 task에 맞는 적절한 model architecture를 자동으로 추론하여 선택.

Hub에 있는 weights와 configuration file을 model과 preprocessor class로 불러오려면 `from_pretrained()` 메서드를 사용.

model을 불러올 때, 다음 parameters을 설정하면 모델이 최적으로 로드.
- `device_map="auto"`: model weights를 가장 빠른 device(예: GPU)에 자동으로 먼저 할당
- `dtype="auto"`: model weights가 저장되어 있는 원본 데이터 타입 그대로 초기화. PyTorch는 기본적으로 `torch.float32`로 weights를 불러온다.


```python

```


```python
model_inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="pt").to(model.device)
generated_ids = model.generate(**model_inputs, max_length=30)
tokenizer.batch_decode(generated_ids)[0]
```

    Setting `pad_token_id` to `eos_token_id`:151643 for open-end generation.
    Both `max_new_tokens` (=2048) and `max_length`(=30) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    




    "The secret to baking a good cake is 100% pure, unadulterated, organic ingredients. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we don't add any artificial preservatives, colors, or flavors. We use only the finest ingredients, and we"



model에 text를 넣고 결과물을 얻는 단계는:
- (1) inputs: `tokenizer`를 사용하여 text를 토큰화하고 PyTorch tensors로 반환
- (2) generated_ids: inputs을 `generate()` 메서드에 전달하여 model이 생성한 response 반환
- (3) `decode()`를 사용하여 generated_ids를 다시 text로 변환  


```python

```

## 2. Pipeline

`pipeline()`은 어떤 모델이든 쉽게 inference를 수행할 수 있게 해주는 추상화 도구. 특정 task를 위한 모델과 그에 알맞은 토크나이저를 자동으로 인스턴스화.


```python
# text generation
from transformers import pipeline
from accelerate import Accelerator

device = Accelerator().device

pipeline = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B", device=device)
```


    Loading weights:   0%|          | 0/338 [00:00<?, ?it/s]


inference에 사용할 수 있는 accelerator를 자동으로 설정하려면 `Accelerator`


```python
pipeline("The secret to baking a good cake is ", max_length=50)
```

    Passing `generation_config` together with generation-related arguments=({'max_length'}) is deprecated and will be removed in future versions. Please pass either a `generation_config` object OR all generation parameters explicitly, but not both.
    Both `max_new_tokens` (=256) and `max_length`(=50) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    




    [{'generated_text': 'The secret to baking a good cake is 2 hours in the oven; the secret to baking a good life is 21 hours in the oven. By this reasoning, baking the perfect cake in 6 hours would require 12 hours of effort (and 12 hours of oven time!). Not a great example of a productivity metric, right? But, as with many things, it’s only a bad example if you have a flawed productivity metric (or two!). If you have a good productivity metric, it doesn’t matter how many hours in the oven it takes to bake a good cake.\nSo what about a better productivity metric? The best productivity metric is the one that provides you with the most bang for your buck. In other words, the one that tells you the most about how productive you are given the time you’re putting in. It’s a fair question to ask how you measure productivity, and it’s a question that can be answered differently depending on who you ask. But there’s one type of productivity metric that is always effective and always useful for measuring productivity: time.\nWhat is Time?\nTime is a unique productivity metric in that it is one of the only things that we can measure and that is constant. For example, if you want to know how productive you are, you could go'}]




```python

```

## 3. Trainer

`Trainer`는 PyTorch models을 위한 완전한 training 및 evaluation loop. 수동으로 학습 루프를 작성할 때 수반되는 많은 반복적인 코드를 추상화하여, 더 빠르게 학습 시작.


```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
dataset = load_dataset("rotten_tomatoes")
```


    config.json:   0%|          | 0.00/483 [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/268M [00:00<?, ?B/s]



    Loading weights:   0%|          | 0/100 [00:00<?, ?it/s]


    [1mDistilBertForSequenceClassification LOAD REPORT[0m from: distilbert/distilbert-base-uncased
    Key                     | Status     | 
    ------------------------+------------+-
    vocab_layer_norm.bias   | UNEXPECTED | 
    vocab_layer_norm.weight | UNEXPECTED | 
    vocab_projector.bias    | UNEXPECTED | 
    vocab_transform.bias    | UNEXPECTED | 
    vocab_transform.weight  | UNEXPECTED | 
    pre_classifier.bias     | MISSING    | 
    pre_classifier.weight   | MISSING    | 
    classifier.weight       | MISSING    | 
    classifier.bias         | MISSING    | 
    
    [3mNotes:
    - UNEXPECTED[3m	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
    - MISSING[3m	:those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.[0m
    


    tokenizer_config.json:   0%|          | 0.00/48.0 [00:00<?, ?B/s]



    vocab.txt: 0.00B [00:00, ?B/s]



    tokenizer.json: 0.00B [00:00, ?B/s]



    README.md: 0.00B [00:00, ?B/s]



    train.parquet:   0%|          | 0.00/699k [00:00<?, ?B/s]



    validation.parquet:   0%|          | 0.00/90.0k [00:00<?, ?B/s]



    test.parquet:   0%|          | 0.00/92.2k [00:00<?, ?B/s]



    Generating train split:   0%|          | 0/8530 [00:00<?, ? examples/s]



    Generating validation split:   0%|          | 0/1066 [00:00<?, ? examples/s]



    Generating test split:   0%|          | 0/1066 [00:00<?, ? examples/s]



```python

```


```python
def tokenize_dataset(dataset):
    return tokenizer(dataset["text"])

dataset = dataset.map(tokenize_dataset, batched=True)
```


    Map:   0%|          | 0/8530 [00:00<?, ? examples/s]



    Map:   0%|          | 0/1066 [00:00<?, ? examples/s]



    Map:   0%|          | 0/1066 [00:00<?, ? examples/s]


`map` 메서드를 사용하여 tokenize_dataset 함수를 전체 데이터셋에 적용. batched=True를 통해 여러 개를 묶어서 병렬 처리 가능


```python

```


```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # padding 처리
```

배치 생성을 위해 data collator를 불러오고, 여기에 토크나이저를 전달.


```python

```


```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="distilbert-rotten-tomatoes",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    push_to_hub=True,
)
```

training process를 사용자 정의하려면 `TrainingArguments` class 사용. training, evaluation 등을 위한 많은 옵션 제공: batch size, learning rate, mixed precision, ...


```python

```

마지막으로, 이 모든 것들을 `Trainer`에 전달하고 `train()`을 호출하여 학습 시작.


```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
)

trainer.train()
```



    <div>

      <progress value='2134' max='2134' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [2134/2134 03:07, Epoch 2/2]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>500</td>
      <td>0.459214</td>
    </tr>
    <tr>
      <td>1000</td>
      <td>0.388816</td>
    </tr>
    <tr>
      <td>1500</td>
      <td>0.262353</td>
    </tr>
    <tr>
      <td>2000</td>
      <td>0.276663</td>
    </tr>
  </tbody>
</table><p>



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]





    TrainOutput(global_step=2134, training_loss=0.34027406179469216, metrics={'train_runtime': 188.0326, 'train_samples_per_second': 90.729, 'train_steps_per_second': 11.349, 'total_flos': 195974132394480.0, 'train_loss': 0.34027406179469216, 'epoch': 2.0})




```python

```


```python
trainer.push_to_hub()
```

`push_to_hub()`로 학습이 끝난 모델과 토크나이저를 Hub에 공유


```python

```
