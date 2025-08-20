import requests
import json
import time
from langserve import RemoteRunnable
import base64
import httpx
from langchain_core.messages import HumanMessage

secret_key = json.load(open('api_data/secret_key.json', 'r'))

def v3api(models, content, layer_num=0):
    url = secret_key[models]['url']
    key = secret_key[models]['key']
    messages = [
            {
                "role": "user",
                "content": content
            }
        ]

    if models in ['o1-mini-2024-09-12']:
        max_token_symbol = "max_completion_tokens"
    else:
        max_token_symbol = "max_tokens"
        
    payload = json.dumps({
        "model": models if models != 'deepseek-chat' else "/root/.cache/huggingface/deepseek",
        "messages": messages,
        max_token_symbol: 4090,
        "temperature": 1,
        "stream": False
    })

    headers = {
        'Authorization': f'Bearer {key}',
        'Content-Type': 'application/json'
    }

    try: 
        response = requests.request("POST", url, headers=headers, data=payload)
        response = json.loads(response.text)
        # print(response)
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(e)
        layer_num += 1
        time.sleep(30 * layer_num)
        ans = v3api(models, content, layer_num)
        return ans


if __name__ == "__main__":
    content = '你好，请解释给定图片中的内容'
    models = 'deepseek-chat'
    print(v3api(models, content))
