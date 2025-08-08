import time
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import base64

def convert_image_to_data_url(image_path):
    # 读取图片文件并编码为 Base64
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    # 构建 data URL
    mime_type = "image/png"  # 根据实际图片类型设置，比如 image/jpeg 或 image/png
    data_url = f"data:{mime_type};base64,{base64_image}"
    return data_url

# 模型映射
model_map = {
    "deepseek-r1": "deepseek/deepseek-r1",
    "gpt-4o-mini": "openai/gpt-4o-mini-search-preview",
    "gpt-4.5-preview": "openai/gpt-4.5-preview",
    "deepseek-v3-0324": "deepseek/deepseek-chat-v3-0324",
    "claude-3.7-sonnet": "anthropic/claude-3.7-sonnet",
    "qwen2.5-coder-7b-instruct": "qwen/qwen2.5-coder-7b-instruct",
    "shisa-v2-llama3.3-70b": "shisa-ai/shisa-v2-llama3.3-70b:free",
    "qwen3-235b-a22b": "qwen/qwen3-235b-a22b",
    "mistral-medium-3":"mistralai/mistral-medium-3",
    "gemini-2.5-pro-preview":"google/gemini-2.5-pro-preview",
    "gpt-o3":"openai/o3",#没法测
    "gpt-4.1":"openai/gpt-4.1",#需要本地vpn
    "claude-3.7-sonnet":"anthropic/claude-3.7-sonnet",
    "claude-3.7-sonnet-thinking":"anthropic/claude-3.7-sonnet:thinking",
    "deepseek-prover-v2":"deepseek/deepseek-prover-v2",#纯数学模型
    "gemini-2.5-flash-preview-thinking":"google/gemini-2.5-flash-preview:thinking",
    "llama-4-maverick":"meta-llama/llama-4-maverick",
    "llama-4-scout":"meta-llama/llama-4-scout",
    "grok-3":"x-ai/grok-3-beta",
    "gemini-2.5-flash-preview":"google/gemini-2.5-flash-preview",
    "grok-3-mini":"x-ai/grok-3-mini-beta",
    "claude-sonnet-4":"anthropic/claude-sonnet-4",# 测了吗？
    "deepseek-r1-0528":"deepseek/deepseek-r1-0528",
    "magistral-medium-thinking-2506":"mistralai/magistral-medium-2506:thinking",
    # "magistral-medium-2506":"mistralai/magistral-medium-2506"
    "ernie-4.5-300b-a47b":"baidu/ernie-4.5-300b-a47b",
    "grok-4":"x-ai/grok-4",# 测
    "kimi-k2":"moonshotai/kimi-k2", # 测
    "qwen3-235b-a22b-07":"qwen/qwen3-235b-a22b-07-25",#no thingking
    "qwen3-235b-a22b-thinking-2507":"qwen/qwen3-235b-a22b-thinking-2507"

}

# 请求函数
def openreuters_fetch_response(model_key, content, image_url=None):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
    }

    # 构造消息内容
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": content}]
        }
    ]

    if image_url is None:
        messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
    else:
        image_base64_list = [{"type": "image_url", "image_url": {"url": convert_image_to_data_url(image)}} for image in image_url]
        for image in image_url:
            if not os.path.exists(image):
                print("!!!!!!!!!!!!!!!!!No image path!!!!!!!!!!!!!!")
                assert False
        messages = [
            {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": content
                    }
                ] + image_base64_list
            }
        ]
    payload = {
        "model": model_map[model_key],
        "messages": messages
    }

    try:
        response = requests.post(url=url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        print(data)
        answer = data['choices'][0]['message']['content']
        return answer
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(60)
        response = requests.request("POST", url, headers=headers, data=payload)
        response = json.loads(response.text)
        return response['choices'][0]['message']['content']

# 主函数
if __name__ == '__main__':
    # model_key = 'claude-3.7-sonnet'
    # model_key = 'grok-3'
    # model_key = 'gemini-2.5-pro-preview'
    # messages = '**测试概述：**\n您即将参加一场**化学**测试，测试形式为**单项选择题**。请按照以下说明完成测试。\n\n---\n\n**你的任务：**\n现在，请**仅仅**输出对应正确答案的字母。\n不要包括任何额外的文本或解释。(例如：`A`，`B`，`C`...)：\n\n问题：以下是一道单项选择题，请从选项中选择一个正确答案。\n请仅输出对应的字母，例如：A。\n\n\n题目：常温下，浓度均为$${1.0mol/L}$$的$${HX}$$溶液、$${HY}$$溶液，分别加水稀释.稀释后溶液的$${pH}$$随浓度的变化如图所示，下列叙述正确的是：() \n\nA. $${HX}$$是强酸，溶液每稀释$${10}$$倍$${.pH}$$始终增大$${1}$$\nB. 常温下$${HY}$$的电离常数为$${1.010^{-4}}$$\nC. 溶液中水的电离程度：a点大于b点\nD. 消耗同浓度的$${NaOH}$$溶液体积：a点大于b点\n\n答案：'
    path = "/ssd/projects/scihorizon/benchmark_data/202501/multimodal/images/CMMU/9bd0f5de5a01c94a45a39b1278049b2e20f84d69c683d366bb575a13b9307696/question_0.png"
    path = "/ssd/projects/scihorizon/benchmark_data/202501/multimodal/images/private/14cc3d4317288f84c2c8503dc79039559799d062548f50b5a783915509ad227f/question_0.png"
    image_urls = [path]
     # benchmark_data/202501/multimodal/images/CMMU/9bd0f5de5a01c94a45a39b1278049b2e20f84d69c683d366bb575a13b9307696
    #image_urls = ['/ssd/projects/scihorizon/benchmark_data/benchmark_data/202501/multimodal/images/CMMU/982d00b58c8c5e64cf33a1b67975975c67082f770a5d6465fee37895131d07b0/question_0.png']
    image_urls = None
    messages = "用中文描述这张图片"
    # for model_key in ['qwen3-235b-a22b', 'gemini-2.5-pro-preview', 'claude-3.7-sonnet-thinking', 'gpt-4.1', 'llama-4-maverick', 'llama-4-scout', 'grok-3']:
    for model_key in ["kimi-k2"]:
        print("[model name]:",model_key)
        answer = openreuters_fetch_response(model_key, messages, image_urls)
        print(answer)
