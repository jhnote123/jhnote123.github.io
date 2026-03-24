# Hugging Face Evaluate Quick Tour


```python
# https://huggingface.co/docs/evaluate/a_quick_tour?utm_source=chatgpt.com
```


```python
# !pip install evaluate
```

## 1. Types of evaluations

Evaluate는 모델을 다양한 측면에서 평가할 수 있도록, 각 평가 목적에 맞는 도구를 제공
- (1) **Metric**: 지표는 모델 성능 평가에 사용. 일반적으로 모델의 predictions과 ground truth labels을 함께 사용. https://huggingface.co/evaluate-metric
- (2) **Comparison**: 두 모델을 비교하는 데 사용. https://huggingface.co/evaluate-comparison  
- (3) **Measurement**: 데이터셋의 properties 확인할 수 있음. https://huggingface.co/evaluate-measurement

각각의 metric, comparison,, measurement 도구는 별도의 모듈로 나뉘어져 있지만, 이들 모두 `evaluate.load()`라는 하나의 함수로 불러올 수 있다.


## 2. Load
metric, comparison, measurement는 `evaluate.load` 함수를 통해 불러올 수 있다.


```python
import evaluate
accuracy = evaluate.load("accuracy")
```

    /usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(
    


    Downloading builder script: 0.00B [00:00, ?B/s]



```python
accuracy
```




    EvaluationModule(name: "accuracy", module_type: "metric", features: {'predictions': Value('int32'), 'references': Value('int32')}, usage: """
    Args:
        predictions (`list` of `int`): Predicted labels.
        references (`list` of `int`): Ground truth labels.
        normalize (`boolean`): If set to False, returns the number of correctly classified samples. Otherwise, returns the fraction of correctly classified samples. Defaults to True.
        sample_weight (`list` of `float`): Sample weights Defaults to None.
    
    Returns:
        accuracy (`float` or `int`): Accuracy score. Minimum possible value is 0. Maximum possible value is 1.0, or the number of examples input, if `normalize` is set to `True`.. A higher score means higher accuracy.
    
    Examples:
    
        Example 1-A simple example
            >>> accuracy_metric = evaluate.load("accuracy")
            >>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0])
            >>> print(results)
            {'accuracy': 0.5}
    
        Example 2-The same as Example 1, except with `normalize` set to `False`.
            >>> accuracy_metric = evaluate.load("accuracy")
            >>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0], normalize=False)
            >>> print(results)
            {'accuracy': 3.0}
    
        Example 3-The same as Example 1, except with `sample_weight` set.
            >>> accuracy_metric = evaluate.load("accuracy")
            >>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0], sample_weight=[0.5, 2, 0.7, 0.5, 9, 0.4])
            >>> print(results)
            {'accuracy': 0.8778625954198473}
    """, stored examples: 0)



다음과 같이 `module_type`을 통해 사용할 평가 도구를 명시하여 불러올 수도 있다.


```python
word_length = evaluate.load("word_length", module_type="measurement")
```


    Downloading builder script: 0.00B [00:00, ?B/s]


    [nltk_data] Downloading package punkt_tab to /root/nltk_data...
    [nltk_data]   Unzipping tokenizers/punkt_tab.zip.
    


```python
word_length
```




    EvaluationModule(name: "word_length", module_type: "measurement", features: {'data': Value('string')}, usage: """
    Args:
        `data`: a list of `str` for which the word length is calculated.
        `tokenizer` (`Callable`) : the approach used for tokenizing `data` (optional).
            The default tokenizer is `word_tokenize` from NLTK: https://www.nltk.org/api/nltk.tokenize.html
            This can be replaced by any function that takes a string as input and returns a list of tokens as output.
    
    Returns:
        `average_word_length` (`float`) : the average number of words in the input list of strings.
    
    Examples:
        >>> data = ["hello world"]
        >>> wordlength = evaluate.load("word_length", module_type="measurement")
        >>> results = wordlength.compute(data=data)
        >>> print(results)
        {'average_word_length': 2}
    """, stored examples: 0)



## 3. Community modules

Evaluate에 내장되어 구현된 모듈들 외에도, metric implementation의 repository ID를 지정하여 다른 사람들이 만든 모듈을 불러올 수 있다.


```python
element_count = evaluate.load("lvwerra/element_count", module_type="measurement")
```


    Downloading builder script: 0.00B [00:00, ?B/s]



```python
element_count
```




    EvaluationModule(name: "element_count", module_type: "measurement", features: {'data': Value('int64')}, usage: """
    Calculates number of elements in dataset
    Args:
        data: list of elements.
    Returns:
        element_count: number of elements in dataset,
    Examples:
        >>> measure = evaluate.load("lvwerra/element_count")
        >>> measure.compute(["a", "b", "c")
        {"element_count": 3}
    """, stored examples: 0)



## 4. List available modules

`list_evaluation_modules()`를 사용하여 Hub에 어떤 모듈들이 사용 가능한지 확인할 수 있다.


```python
evaluate.list_evaluation_modules(
  module_type="comparison",
  include_community=True,
  with_details=True)
```




    [{'name': 'ncoop57/levenshtein_distance',
      'type': 'comparison',
      'community': True,
      'likes': 0},
     {'name': 'kaleidophon/almost_stochastic_order',
      'type': 'comparison',
      'community': True,
      'likes': 1},
     {'name': 'NeuraFusionAI/Arabic-Evaluation',
      'type': 'comparison',
      'community': True,
      'likes': 0},
     {'name': 'PaxxStacks/TicTacToedown',
      'type': 'comparison',
      'community': True,
      'likes': 0},
     {'name': 'sodyb/SimpleBench',
      'type': 'comparison',
      'community': True,
      'likes': 0}]



## 5. Module attributes
모든 평가 모듈에는 그 안에 어떤 정보들이 들어있는지 확인할 수 있는 attributes이 포함되어 있으며, 이는 `EvaluationModuleInfo` 객체에 저장되어 있다.

Attribute
- (1) description: 평가 모듈이 어떻게 작동하는지에 대한 짧은 설명
- (2) citation
- (3) features: 사용할 도구가 요구하는 **입력 데잍터의 형식**
- (4) inputs_description: 상세 설명
- (5) homepage: 해당 도구의 공식 홈페이지
- (6) license: 라이센서 정보
- (7) codebase_urls: 모듈이 구현된 원본 코드 링크
- (8) reference_urls: 기타 추가적인 URL들


```python
accuracy = evaluate.load("accuracy")
```


```python
accuracy.description
```




    '\nAccuracy is the proportion of correct predictions among the total number of cases processed. It can be computed with:\nAccuracy = (TP + TN) / (TP + TN + FP + FN)\n Where:\nTP: True positive\nTN: True negative\nFP: False positive\nFN: False negative\n'




```python
accuracy.citation
```




    '\n@article{scikit-learn,\n  title={Scikit-learn: Machine Learning in {P}ython},\n  author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.\n         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.\n         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and\n         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},\n  journal={Journal of Machine Learning Research},\n  volume={12},\n  pages={2825--2830},\n  year={2011}\n}\n'




```python
accuracy.features
# accuracy를 계산하려면,
# predictions과 references라는 두 가지 데이터가 필요하며, 각각 int32로 넣어주어야 함을 확인할 수 있음
```




    {'predictions': Value('int32'), 'references': Value('int32')}



`Evaluate`는 다양한 입력 형식(Python lists, NumPy arrays, PyTorch tensors, etc.)을 허용하며, 저장 및 계산을 위해 이를 적절한 형식으로 자동 변환한다.

## 6. Compute

`Evaluate`는 두 가지 계산 방식을 제공한다.
- All-in-one
- Incremental

incremental approach에서는 `EvaluationModule.add()` 또는 `EvaluationModule.add_batch()`를 사용하여 필요한 입력값들을 모듈에 차곡차곡 추가하고, 마지막에 `EvaluationModule.compute()`를 호출하여 최종 점수를 계산한다.

대안은 모든 입력값을 한 번에 모두 `compute()` 함수에 전달하는 것이다.  

## 7. How to compute - All-in-one

즉, 가장 간단한 방법은 필요한 입력값들과 함께 `compute()`를 호출하는 것이다.

`features`속성에서 확인했던 입력값들을 다음과 같이 `compute()`메서드에 간단히 전달하기만 하면 된다.


```python
accuracy.compute(references=[0,1,0,1], predictions=[1,0,0,1])
```




    {'accuracy': 0.5}



경우에 따라 예측값을 반복적으로 쌓아가거나 여러 대의 GPU를 사용하는 환경에서 처리해야 할 때가 있는데, 이런 경우에는 `add()` 또는 `add_batch()`가 유용하다.

## 8. Calculate a single metric or a batch of metrics - Incremental


많은 평가 파이프라인에서 for 루프와 같이 반복적으로 예측값을 생성한다. 이런 경우 다음과 같이 `add()`를 통해 예측값을 저장해두었다가 마지막에 `compute()`에 전달하여 최종 점수를 한 번에 계산할 수 있다.


```python
for ref, pred in zip([0,1,0,1], [1,0,0,1]):
    accuracy.add(references=ref, predictions=pred) # 1개씩 평가 모듈에 정답과 예측값을 누적
accuracy.compute() # 누적이 끝난 후, 최종 점수를 한 번에 계산
```




    {'accuracy': 0.5}



예측값과 정답을 배치 단위로 다룰 때는 `add_batch()`를 사용하면 된다.


```python
import numpy as np

refs, preds = np.array([[0, 1], [0, 1]]), np.array([[1, 0], [0, 1]])
refs.shape, preds.shape # [batch size=2, 2]
```




    ((2, 2), (2, 2))




```python
for r, p in zip(refs, preds):
    accuracy.add_batch(references=r, predictions=p)
accuracy.compute()
```




    {'accuracy': 0.5}



`add_batch()`는 모델에서 배치 단위로 예측값을 가져와야 할 때 유용하게 사용할 수 있다.


```python
for model_inputs, gold_standards in evaluation_dataset:
    predictions = model(model_inputs)
    metric.add_batch(references=gold_standards, predictions=predictions)
metric.compute()
```

## 9. Combining several evaluations

모델의 다양한 측면을 평가하는 여러 가지 지표들을 함께 평가하려 할 때, 예를 들어 classification에서는 모델의 성능을 더 파악하기 위해 accuracy 외에도 F1-score, recall, precision을 함께 계산하는 것이 일반적이다.

여러 지표를 각각 불러와서 순차적으로 호출할 수도 있지만, 더 편리한 방법은 다음과 같이 `combine()` 함수를 사용하여 이들을 하나로 묶는 것이다.


```python
clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
```


    Downloading builder script: 0.00B [00:00, ?B/s]



    Downloading builder script: 0.00B [00:00, ?B/s]



    Downloading builder script: 0.00B [00:00, ?B/s]



```python
clf_metrics.compute(predictions=[0, 1, 0], references=[0, 1, 1])
```




    {'accuracy': 0.6666666666666666,
     'f1': 0.6666666666666666,
     'precision': 1.0,
     'recall': 0.5}



## 10. Save and push to the Hub

평가 결과는 `evaluate.save()`로 저장할 수 있다.

특정 파일 이름이나 폴더 경로를 전달할 수 있으며, 후자의 경우, 자동으로 생성된 파일 이름으로 결과가 저장된다.


```python
result = accuracy.compute(references=[0,1,0,1], predictions=[1,0,0,1])

hyperparams = {"model": "bert-base-uncased"}
evaluate.save("./results/", experiment="run 42", **result, **hyperparams)
```




    PosixPath('results/result-2026_03_24-12_41_05.json')




```python
{
    "experiment": "run 42",
    "accuracy": 0.5,
    "model": "bert-base-uncased",
    "_timestamp": "2022-05-30T22:09:11.959469",
    "_git_commit_hash": "123456789abcdefghijkl",
    "_evaluate_version": "0.1.0",
    "_python_version": "3.9.12 (main, Mar 26 2022, 15:51:15) \n[Clang 13.1.6 (clang-1316.0.21.2)]",
    "_interpreter_path": "/Users/leandro/git/evaluate/env/bin/python"
}
```

`evaluate.push_to_hub()` 함수를 사용하면, Hub에 있는 모델의 repository에 report도 가능하다.


```python
evaluate.push_to_hub(
  model_id="huggingface/gpt2-wikitext2",  # model repository on hub
  metric_value=0.5,                       # metric value
  metric_type="bleu",                     # metric name, e.g. accuracy.name
  metric_name="BLEU",                     # pretty name which is displayed
  dataset_type="wikitext",                # dataset name on the hub
  dataset_name="WikiText",                # pretty name
  dataset_split="test",                   # dataset split used
  task_type="text-generation",            # task id, see https://github.com/huggingface/evaluate/blob/main/src/evaluate/config.py#L154-L192
  task_name="Text Generation"             # pretty name for task
)
```

## 11. Evaluator

`evaluate.evaluator()`로 평가 과정을 자동화할 수 있다. 모델의 예측값을 필요로 하던 `EvaluationModules`의 지표들과 달리 **오직 모델, 데이터셋, 평가지표**만을 필요로 한다.

inference 과정이 내부적으로 처리되기 때문에, 주어진 지표로 데이터셋에 대해 모델을 평가하는 것이 훨씬 간단하다. 이를 가능하게 하기 위해 `transformers`의 `pipeline` 추상화를 사용한다.


```python
from transformers import pipeline, TFPreTrainedModel
from datasets import load_dataset
from evaluate import evaluator
import evaluate

pipe = pipeline("text-classification", model="lvwerra/distilbert-imdb", device=0)
data = load_dataset("imdb", split="test").shuffle().select(range(1000))
metric = evaluate.load("accuracy")
```


    Loading weights:   0%|          | 0/104 [00:00<?, ?it/s]



```python
# '텍스트 분류' 전용 자동 평가기를 만듭니다.
task_evaluator = evaluator("text-classification")

# 준비한 3가지를 한 번에 넣고 실행합니다!
results = task_evaluator.compute(
    model_or_pipeline=pipe,
    data=data,
    metric=metric,
    label_mapping={"NEGATIVE": 0, "POSITIVE": 1} # 모델의 출력 단어를 숫자로 번역
)

print(results)
```

    {'accuracy': 0.916, 'total_time_in_seconds': 10.966879519999566, 'samples_per_second': 91.18364054026186, 'latency_in_seconds': 0.010966879519999567}
    

단순히 지표 값 하나만 계산하는 것으로는 한 모델이 다른 모델보다 유의미하게 더 나은 성능을 내는지 판단하기에 충분하지 않은 경우가 많다.

`evaluate`는 부트스트래핑을 통해 신뢰 구간과 표준 오차를 계산하여 점수가 얼마나 안정적인지 추정할 수 있다.


```python
results = task_evaluator.compute(model_or_pipeline=pipe, data=data, metric=metric,
                       label_mapping={"NEGATIVE": 0, "POSITIVE": 1},
                       strategy="bootstrap", n_resamples=200)

print(results)
```

    {'accuracy': {'confidence_interval': (np.float64(0.8926477890014252), np.float64(0.928)), 'standard_error': np.float64(0.009037386056469452), 'score': 0.916}, 'total_time_in_seconds': 12.98525015300038, 'samples_per_second': 77.01045326176789, 'latency_in_seconds': 0.012985250153000378}
    

## 12. Visualization - `radar_plot()`


```python
import evaluate
from evaluate.visualization import radar_plot

data = [
   {"accuracy": 0.99, "precision": 0.8, "f1": 0.95, "latency_in_seconds": 33.6},
   {"accuracy": 0.98, "precision": 0.87, "f1": 0.91, "latency_in_seconds": 11.2},
   {"accuracy": 0.98, "precision": 0.78, "f1": 0.88, "latency_in_seconds": 87.6},
   {"accuracy": 0.88, "precision": 0.78, "f1": 0.81, "latency_in_seconds": 101.6}
   ]
model_names = ["Model 1", "Model 2", "Model 3", "Model 4"]
plot = radar_plot(data=data, model_names=model_names)
plot.show()
```


    
![png](output_47_0.png)
    


## 13. Running evaluation on a suite of tasks

모델의 다운스트림 성능에 대해, 다양하고 서로 다른 tasks에 대해 모델을 평가하는 것이 유용할 수 있다.

`EvaluationSuite`는 여러 tasks의 collection에 대한 모델 평가를 가능하게 한다.


```python
import evaluate
from evaluate.evaluation_suite import SubTask

# 평가 스위트 만들기
class Suite(evaluate.EvaluationSuite):
    def __init__(self, name):
        super().__init__(name)
        # 모델 하나로 IMDb와 SST2 평가
        self.suite = [
            SubTask(
                task_type="text-classification",
                data="imdb",
                split="test[:1]",
                args_for_task={
                    "metric": "accuracy",
                    "input_column": "text",
                    "label_column": "label",
                    "label_mapping": {
                        "LABEL_0": 0.0,
                        "LABEL_1": 1.0
                    }
                }
            ),
            SubTask(
                task_type="text-classification",
                data="sst2",
                split="test[:1]",
                args_for_task={
                    "metric": "accuracy",
                    "input_column": "sentence",
                    "label_column": "label",
                    "label_mapping": {
                        "LABEL_0": 0.0,
                        "LABEL_1": 1.0
                    }
                }
            )
        ]
```

평가는 `EvaluationSuite`를 불러온 뒤, 모델이나 파이프라인과 함꼐 `run()` 메서드를 호출하여 실행할 수 있다.


```python
from evaluate import EvaluationSuite
suite = EvaluationSuite.load('mathemakitten/sentiment-evaluation-suite')
results = suite.run("huggingface/prunebert-base-uncased-6-finepruned-w-distil-mnli")
```


    Downloading builder script: 0.00B [00:00, ?B/s]



    Map:   0%|          | 0/10 [00:00<?, ? examples/s]


    WARNING:evaluate.evaluator.base:`data` is a preloaded Dataset! Ignoring `subset` and `split`.
    


    config.json: 0.00B [00:00, ?B/s]



    pytorch_model.bin:   0%|          | 0.00/438M [00:00<?, ?B/s]



    Loading weights:   0%|          | 0/201 [00:00<?, ?it/s]



    model.safetensors:   0%|          | 0.00/438M [00:00<?, ?B/s]



    tokenizer_config.json:   0%|          | 0.00/39.0 [00:00<?, ?B/s]



    vocab.txt: 0.00B [00:00, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/112 [00:00<?, ?B/s]



    README.md: 0.00B [00:00, ?B/s]



    data/train-00000-of-00001.parquet:   0%|          | 0.00/3.11M [00:00<?, ?B/s]



    data/validation-00000-of-00001.parquet:   0%|          | 0.00/72.8k [00:00<?, ?B/s]



    data/test-00000-of-00001.parquet:   0%|          | 0.00/148k [00:00<?, ?B/s]



    Generating train split:   0%|          | 0/67349 [00:00<?, ? examples/s]



    Generating validation split:   0%|          | 0/872 [00:00<?, ? examples/s]



    Generating test split:   0%|          | 0/1821 [00:00<?, ? examples/s]



    Map:   0%|          | 0/10 [00:00<?, ? examples/s]


    WARNING:evaluate.evaluator.base:`data` is a preloaded Dataset! Ignoring `subset` and `split`.
    


    Loading weights:   0%|          | 0/201 [00:00<?, ?it/s]



```python
results
```




    [{'accuracy': 0.3,
      'total_time_in_seconds': 0.2743247339994923,
      'samples_per_second': 36.45314753141621,
      'latency_in_seconds': 0.02743247339994923,
      'task_name': 'imdb',
      'data_preprocessor': '<function Suite.__init__.<locals>.<lambda> at 0x7a4c7998f9c0>'},
     {'accuracy': 0.0,
      'total_time_in_seconds': 0.1975849290001861,
      'samples_per_second': 50.61114757386471,
      'latency_in_seconds': 0.01975849290001861,
      'task_name': 'sst2',
      'data_preprocessor': '<function Suite.__init__.<locals>.<lambda> at 0x7a4c7998f920>'}]


