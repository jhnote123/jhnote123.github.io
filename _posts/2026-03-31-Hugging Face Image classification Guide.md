---
title: "Hugging Face Vision Transformer (ViT)"
date: 2026-03-27 19:00:00 +0900
categories: [Hugging Face]
tags: [vision transformer, vit, image classification, huggingface]
---

# Hugging Face Image classification task guide


```python
# https://huggingface.co/docs/transformers/tasks/image_classification
```

Image classification은 이미지에 label이나 class를 할당하는 task이다.

text나 audio classification과 달리, 입력값은 image를 구성하는 pixel 값들이다.

이 가이드에서는 다음 방법들을 설명한다:
- 1. 이미지 속 음식 분류를 위해 Food-101 데이터셋으로 ViT를 fine-tuning
- 2. fine-tuned model을 inference에 사용하는 방법

## 1. Load dataset


```python
# !pip install transformers datasets evaluate accelerate pillow torchvision scikit-learn trackio
```


```python
from datasets import load_dataset

# Load Food-101 dataset
food = load_dataset("ethz/food101", split="train[:5000]")
```


```python
food = food.train_test_split(test_size=0.2) # 80%는 train, 20%는 test
food["train"][0]
# image: PIL image of the food item
# label: label class of the food item
```




    {'image': <PIL.Image.Image image mode=RGB size=384x512>, 'label': 53}




```python
import pandas as pd

labels = food["train"].features["label"].names
labels_series = pd.Series(labels)
print(labels_series.nunique()); print(labels_series.unique())
```

    101
    ['apple_pie' 'baby_back_ribs' 'baklava' 'beef_carpaccio' 'beef_tartare'
     'beet_salad' 'beignets' 'bibimbap' 'bread_pudding' 'breakfast_burrito'
     'bruschetta' 'caesar_salad' 'cannoli' 'caprese_salad' 'carrot_cake'
     'ceviche' 'cheesecake' 'cheese_plate' 'chicken_curry'
     'chicken_quesadilla' 'chicken_wings' 'chocolate_cake' 'chocolate_mousse'
     'churros' 'clam_chowder' 'club_sandwich' 'crab_cakes' 'creme_brulee'
     'croque_madame' 'cup_cakes' 'deviled_eggs' 'donuts' 'dumplings' 'edamame'
     'eggs_benedict' 'escargots' 'falafel' 'filet_mignon' 'fish_and_chips'
     'foie_gras' 'french_fries' 'french_onion_soup' 'french_toast'
     'fried_calamari' 'fried_rice' 'frozen_yogurt' 'garlic_bread' 'gnocchi'
     'greek_salad' 'grilled_cheese_sandwich' 'grilled_salmon' 'guacamole'
     'gyoza' 'hamburger' 'hot_and_sour_soup' 'hot_dog' 'huevos_rancheros'
     'hummus' 'ice_cream' 'lasagna' 'lobster_bisque' 'lobster_roll_sandwich'
     'macaroni_and_cheese' 'macarons' 'miso_soup' 'mussels' 'nachos'
     'omelette' 'onion_rings' 'oysters' 'pad_thai' 'paella' 'pancakes'
     'panna_cotta' 'peking_duck' 'pho' 'pizza' 'pork_chop' 'poutine'
     'prime_rib' 'pulled_pork_sandwich' 'ramen' 'ravioli' 'red_velvet_cake'
     'risotto' 'samosa' 'sashimi' 'scallops' 'seaweed_salad'
     'shrimp_and_grits' 'spaghetti_bolognese' 'spaghetti_carbonara'
     'spring_rolls' 'steak' 'strawberry_shortcake' 'sushi' 'tacos' 'takoyaki'
     'tiramisu' 'tuna_tartare' 'waffles']
    


```python
# Create dictionary
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

print(id2label[str(79)])
```

    prime_rib
    

## 2. Preprocess

다음 단계는 image를 tensor로 만들기 위해 image processor를 사용하는 것이다. 여기서는 ViT image processor를 불러온다.


```python
from transformers import AutoImageProcessor

checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
```

    Fast image processor class <class 'transformers.models.vit.image_processing_vit_fast.ViTImageProcessorFast'> is available for this model. Using slow image processor class. To use the fast image processor class set `use_fast=True`.
    

모델이 과적합에 robust해지도록 이미지에 몇 가지 image transformations을 적용한다.

아래는 이미지 픽셀 값의 평균과 표준편차로 정규화를 한 다음, 이미지의 임의의 부분을 자르고(crop), 크기를 조정(resize)하는 과정을 나타낸 것이다.


```python
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)
_transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])
```


```python
image_processor.size["height"], image_processor.size["width"]
```




    (224, 224)



그런 다음 transformation을 적용하고 모델의 입력값인 이미지의 `pixel_values`를 반환하는 전처리 함수를 만든다.


```python
def transforms(examples):
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples
```

정의한 전처리 함수를 데이터셋 전체에 적용하려면, Datasets의 `with_transform` 메서드를 사용하면 된다.


```python
food = food.with_transform(transforms)
```

`DefaultDataCollator`를 사용하여 examples의 batch를 만든다.

Transformers의 다른 data collator들과 달리, `DefaultDataCollator`는 padding과 같은 추가적인 전처리를 적용하지 않는다.
- 앞선 `RandomResizedCrop` 단계에서 모든 사진을 224 x 224라는 똑같은 크기로 맞춰두었기 때문에, 추가적인 작업 없이 개별 데이터를 배치 단위로 묶어주는 가장 기본적인 `DefaultDataCollator`를 사용


```python
from transformers import DefaultDataCollator

data_collator = DefaultDataCollator()
```

## 3. Evaluate

Evaluate 라이브러리를 통해 사용할 metric을 불러온다. 여기서는 `accuracy`를 사용한다.


```python
# https://jhnote123.github.io/posts/Hugging-Face-Evaluate-Quick-Tour/
```


```python
import evaluate

accuracy = evaluate.load("accuracy")
```


    Downloading builder script: 0.00B [00:00, ?B/s]


그런 다음 예측값(predictions)과 정답 라벨(labels)을 `compute` 함수에 전달하여 accuracy를 계산하는 함수를 만든다.


```python
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)
```

## 4. Train

`AutoModelForImageClassification`을 사용하여 ViT 모델을 불러온다. 이때, `num_labels`에 라벨의 총개수를 지정하고, 앞서 만든 dictionary들도 지정한다. 모델 출력값(숫자 ID)과 실제 클래스 명칭(문자열)을 자동으로 연결한다.


```python
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)
```


    Loading weights:   0%|          | 0/198 [00:00<?, ?it/s]


    ViTForImageClassification LOAD REPORT from: google/vit-base-patch16-224-in21k
    Key                 | Status     | 
    --------------------+------------+-
    pooler.dense.weight | UNEXPECTED | 
    pooler.dense.bias   | UNEXPECTED | 
    classifier.bias     | MISSING    | 
    classifier.weight   | MISSING    | 
    
    Notes:
    - UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
    - MISSING	:those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
    

`TrainingArguments`에 training hyperparameters를 정의한다.

`remove_unused_columns=False`로 설정하는 이유는 `image`열이 통째로 삭제되는 것을 방지하기 위해서이다. `image` 열이 없으면 `pixel_values`를 만들 수 없다.


```python
training_args = TrainingArguments(
    output_dir="my_awesome_food_model", # 모델을 저장할 위치
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    warmup_steps=0.1,
    logging_steps=10,
   # report_to="trackio", # 학습 로그 기록
    run_name="food101",
    load_best_model_at_end=False,
    metric_for_best_model="accuracy",
    push_to_hub=False, # 모델을 Hub에 업로드할지 여부
)
```

위에서 정의한 학습 인자들(training_args)을 `Trainer`에 전달


```python
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=food["train"],
    eval_dataset=food["test"],
    processing_class=image_processor,
    compute_metrics=compute_metrics,
)

trainer.train()
```



    <div>

      <progress value='189' max='189' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [189/189 09:14, Epoch 3/3]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>2.749011</td>
      <td>2.556767</td>
      <td>0.831000</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.871300</td>
      <td>1.815923</td>
      <td>0.863000</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.592312</td>
      <td>1.621790</td>
      <td>0.892000</td>
    </tr>
  </tbody>
</table><p>



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]





    TrainOutput(global_step=189, training_loss=2.4589004819355313, metrics={'train_runtime': 558.6828, 'train_samples_per_second': 21.479, 'train_steps_per_second': 0.338, 'total_flos': 9.307289843712e+17, 'train_loss': 2.4589004819355313, 'epoch': 3.0})




```python
trainer.save_model()
```


    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]



```python
# trainer.push_to_hub()
```

## 5. Inference

fine-tuning이 끝났으며, 이를 inference에 사용할 수 있다.


```python
# inference에 사용할 이미지 로드

ds = load_dataset("ethz/food101", split="validation[:10]")
image = ds["image"][0]
```

fine-tuned model을 inference에 사용해 보는 가장 간단한 방법은 다음과 같이 `pipeline()` 안에 넣어 사용하는 것이다.


```python
from transformers import pipeline

classifier = pipeline("image-classification", model="my_awesome_food_model")
classifier(image)
```


    Loading weights:   0%|          | 0/200 [00:00<?, ?it/s]





    [{'label': 'beignets', 'score': 0.30939388275146484},
     {'label': 'ramen', 'score': 0.01662697084248066},
     {'label': 'prime_rib', 'score': 0.015273718163371086},
     {'label': 'bruschetta', 'score': 0.013892950490117073},
     {'label': 'pork_chop', 'score': 0.012161750346422195}]




```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

image_processor = AutoImageProcessor.from_pretrained("my_awesome_food_model")
inputs = image_processor(image, return_tensors="pt")

model = AutoModelForImageClassification.from_pretrained("my_awesome_food_model")
with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
model.config.id2label[predicted_label]
```


    Loading weights:   0%|          | 0/200 [00:00<?, ?it/s]





    'beignets'


