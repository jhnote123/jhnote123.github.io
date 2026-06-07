---
title: "Hugging Face Expanding Chat Templates with Tools"
date: 2026-04-01 16:00:00 +0900
categories: [Hugging Face]
---

# Expanding Chat Templates with Tools


```python
# https://huggingface.co/docs/transformers/main/chat_template_tools_and_documents
```

`apply_chat_template`에 필수로 전달해야 하는 유일한 argument는 `messages`이며, 원하는 keyword argument를 추가로 전달할 수 있다.

기본적인 호출은 다음과 같다.
```python
tokenizer.apply_chat_template(messages)
```

도구를 함께 전달하려면 다음과 같이 사용한다.
```python
tokenizer.apply_chat_template(
    messages,
    tools=tools
)
```

단, `tools`를 전달한다고 모든 모델이 함수 호출을 지원하는 것은 아니다. 모델의 채팅 템플릿이 `tools` 인자를 실제로 처리하도록 작성되어 있어야 한다.

#### Tool use / function calling
"tool use" 기능을 지원하는 LLM은 answer를 생성하기 전에 external tools로 등록된 functions을 호출할 수 있다.

tool-use model에 tools을 전달할 때는 functions의 list를 tools 인자로 전달하면 된다.

```python
import datetime

def current_time():
    """Get the current local time as a string."""
    return str(datetime.now())

def multiply(a: float, b: float):
    """
    A function that multiplies two numbers

    Args:
        a: The first number to multiply
        b: The second number to multiply
    """
    return a * b

tools = [current_time, multiply]

model_input = tokenizer.apply_chat_template(
    messages,
    tools=tools
)
```


위와 같은 형식으로 function을 작성해야 tool로 올바르게 파싱될 수 있다.

function이 tool로 올바르게 작동되려면 다음 규칙을 따라야 한다.
- (1) function의 이름은 기능을 알 수 있도록 명확해야 한다.
  - `run_m` 같은 모호한 이름보다는 `multiply` 같은 이름이 좋다.
- (2) 모든 매개변수에는 타입 힌트가 있어야 한다.
```python
def multiply(a: float, b: float):
```
- (3) function에는 standard Google style의 docstring이 있어야 한다.
  - 즉, 함수에 대한 초기 설명 다음에 인자를 설명하는 `Args:` 블록이 와야 한다. 단, 함수에 인자가 없는 경우는 예외이다.
  - 그리고 `Args:` 블록에는 타입을 작성하지 않는다.
    - 올바른 형식: `a: The first number to multiply`
    - 권장하지 않는 형식: `a (int): The first number to multiply`
```python
def multiply(a: float, b: float):
    """
    A function that multiplies two numbers

    Args:
        a: The first number to multiply
        b: The second number to multiply
    """
```
- (4) 반환 타입과 `Returns:` 블록은 선택 사항이다.
```python
def multiply(a: float, b: float) -> float:
    """
    A function that multiplies two numbers.

    Args:
        a: The first number
        b: The second number

    Returns:
        The multiplication result.
    """
```


#### Passing tool results to the model

위의 샘플 코드는 모델에서 사용할 수 있는 tools을 나열하는 단계이다. 모델이 실제로 tools을 사용하려면 다음과 같이 해야 한다.
- (1) 모델의 output을 파싱하여 tool name(s)와 arguments을 추출한다.
- (2) 모델이 생성한 tool call(s)을 대화 기록에 추가한다.
- (3) 해당 arguments을 사용하여 function(s)을 실행한다.
- (4) function(s) 실행 결과를 대화 기록에 추가한다.

즉, 모델이 function을 직접 실행하는 것이 아니다. 모델은 호출한 함수와 인자를 텍스트 또는 JSON으로 출력할 뿐이고, 실제 실행은 프로그램이 담당한다.

#### A complete tool use example



```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="auto")
```

    /usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:112: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(
    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
    WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
    


    config.json:   0%|          | 0.00/659 [00:00<?, ?B/s]



    tokenizer_config.json:   0%|          | 0.00/7.30k [00:00<?, ?B/s]



    vocab.json:   0%|          | 0.00/2.78M [00:00<?, ?B/s]



    merges.txt:   0%|          | 0.00/1.67M [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/7.03M [00:00<?, ?B/s]


    [transformers] `torch_dtype` is deprecated! Use `dtype` instead!
    


    model.safetensors:   0%|          | 0.00/988M [00:00<?, ?B/s]



    Loading weights:   0%|          | 0/290 [00:00<?, ?it/s]



    generation_config.json:   0%|          | 0.00/242 [00:00<?, ?B/s]



```python
## tools의 list 정의
## 아래 함수들은 실제 API를 호출하지 않는 더미 함수

# 온도 함수는 항상 22.0 반환
def get_current_temperature(location: str, unit: str) -> float:
    """
    Get the current temperature at a location.

    Args:
        location: The location to get the temperature for, in the format "City, Country"
        unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])
    Returns:
        The current temperature at the specified location in the specified units, as a float.
    """
    return 22.  # A real function should probably actually get the temperature!

# 풍속 함수는 항상 6.0 반환
def get_current_wind_speed(location: str) -> float:
    """
    Get the current wind speed in km/h at a given location.

    Args:
        location: The location to get the temperature for, in the format "City, Country"
    Returns:
        The current wind speed at the given location in km/h, as a float.
    """
    return 6.  # A real function should probably actually get the wind speed!

tools = [get_current_temperature, get_current_wind_speed]
```


```python
## 대화 구성

messages = [
  {
      "role": "system",
      "content": (
          "You are a bot that responds to weather queries."
          "You should reply with the unit used in the queried location."
          )
  },
  {
      "role": "user",
      "content": "Hey, what's the temperature in Paris right now?"
  }
]
```


```python
# 채팅 템플릿 적용 및 첫 번째 응답 생성

inputs = tokenizer.apply_chat_template(
    messages,
    tools=tools,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)

inputs = {k: v.to(model.device) for k, v in inputs.items()}

out = model.generate(
    **inputs,
    max_new_tokens=128
)

# 입력 길이 이후만 디코딩해서 모델의 응답만 출력
print(tokenizer.decode(out[0][len(inputs["input_ids"][0]):]))
```

    <tool_call>
    {"name": "get_current_temperature", "arguments": {"location": "Paris, France", "unit": "celsius"}}
    </tool_call><|im_end|>
    

```python
<tool_call>
{"name": "get_current_temperature", "arguments": {"location": "Paris, France", "unit": "celsius"}}
</tool_call><|im_end|>
```

모델이 function docstring이 요청한 형식으로 유효한 arguments로 function을 호출했음을 볼 수 있다. 프랑스의 온도는 celsius가 적절하다고 판단했다.


```python
## 다음으로, 모델의 tool call을 대화 기록에 추가한다.

tool_call = {
    "name": "get_current_temperature",
    "arguments": {
        "location": "Paris, France",
        "unit": "celsius"
    }
}

## 이 messages는“assistant가 이 함수를 호출하기로 결정했다”는 사실을 대화 기록에 저장한다.
messages.append({
    "role": "assistant",
    "tool_calls": [
        {
            "type": "function",
            "function": tool_call
        }
    ]
})
```


```python
messages
```




    [{'role': 'system',
      'content': 'You are a bot that responds to weather queries.You should reply with the unit used in the queried location.'},
     {'role': 'user',
      'content': "Hey, what's the temperature in Paris right now?"},
     {'role': 'assistant',
      'tool_calls': [{'type': 'function',
        'function': {'name': 'get_current_temperature',
         'arguments': {'location': 'Paris, France', 'unit': 'celsius'}}}]}]




```python
## 대화에 tool call을 추가했으니 함수를 호출하고 결과를 대화에 추가할 수 있다

# 이 예시에서는 항상 22.0을 반환하는 더미 함수만 사용하고 있으므로 해당 결과를 직접 추가한 것이다.
messages.append({"role": "tool", "name": "get_current_temperature", "content": "22.0"})
```


```python
## 마지막으로 assistant가 함수 출력을 읽고 user와 계속 채팅하도록 한다.

inputs = tokenizer.apply_chat_template(
    messages,
    tools=tools,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = {k: v.to(model.device) for k, v in inputs.items()}
out = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(out[0][len(inputs["input_ids"][0]):]))
```

    The current temperature in Paris is 22°C.<|im_end|>
    

이 예시는 더미 도구를 한 번 호출하는 단순한 사례지만, 같은 방식으로 다음과 같은 기능들도 구현할 수 있다.
- 여러 개의 real tools 호출
- 긴 대화
- 실시간 정보 조회
- 계산기
- 데이터베이스 접근


#### Understanding tool schemas

`apply_chat_template`의 `tools` 인자에 전달된 각 function은 JSON 스키마로 변환되며, 이 JSON 스키마가 채팅 템플릿을 통해 모델에 전달된다.

즉, tool-use model은 function을 직접 보지 않으며, 그 안의 실제 코드를 볼 수 없다. tool-use model은 tool이 무엇을 하고 어떻게 작동하는지가 아니라, tool을 어떻게 사용하는지(예: 함수에 필요한 매개변수와 타입 등)에 초점을 둔다.


function이 앞에서 설명한 규격을 따르면 JSON 스키마 생성은 자동으로 처리된다. 만약, 문제가 발생하거나 변환 과정을 직접 제어하고 싶다면 `get_json_schema`를 통해 수동으로 변환을 처리할 수 있다. 아래 코드는 수동 스키마 변환에 대한 예시이다.


```python
from transformers.utils import get_json_schema

def multiply(a: float, b: float):
    """
    A function that multiplies two numbers

    Args:
        a: The first number to multiply
        b: The second number to multiply
    """
    return a * b

schema = get_json_schema(multiply)
schema
```




    {'type': 'function',
     'function': {'name': 'multiply',
      'description': 'A function that multiplies two numbers',
      'parameters': {'type': 'object',
       'properties': {'a': {'type': 'number',
         'description': 'The first number to multiply'},
        'b': {'type': 'number', 'description': 'The second number to multiply'}},
       'required': ['a', 'b']}}}



- `type: "function"`은 함수형 도구라는 의미
- 'name'은 모델이 호출할 함수의 이름
- 'description'은 함수가 하는 일에 대한 설명
- `arameters.type: "object"`는 인자를 JSON 객체로 전달한다는 의미
- `properties`는 각 인자의 타입과 설명
- `required`는 반드시 제공해야 하는 인자 리스트

필요하면 이러한 스키마를 수정하거나, `get_json_schema`를 전혀 사용하지 않고 직접 처음부터 작성할 수도 있다.

작성한 JSON 스키마는 `apply_chat_template`의 `tools`에 직접 전달할 수 있다. 그러므로 복잡한 함수를 더 정밀하게 정의할 수 있다.

그러나 스키마가 복잡할수록 모델이 혼동할 가능성도 커진다. 그래서 가능하면 인자를 최소한으로 유지하는 간단한 함수를 사용하는 것이 좋다.

아래는 스키마를 직접 정의하고, 이를 전달하여 `apply_chat_template`을 적용하는 예시이다.


```python
# A simple function that takes no arguments
current_time = {
    "type": "function",
    "function": {
        "name": "current_time",
        "description": "Get the current local time as a string.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
}

# A more complete function that takes two numerical arguments
multiply = {
    "type": "function",
    "function": {
        "name": "multiply",
        "description": "A function that multiplies two numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "The first number to multiply"
                },
                "b": {
                    "type": "number",
                    "description": "The second number to multiply"
                }
            },
            "required": ["a", "b"]
        }
    }
}
```


```python
# 직접 작성한 스키마를 모델에게 전달
model_input = tokenizer.apply_chat_template(
    messages,
    tools=[current_time, multiply]
)
```
