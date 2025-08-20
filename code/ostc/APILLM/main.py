from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import pandas as pd
from utils import v3api
from tqdm import tqdm
from config import args
import os
import json
import random

final_ans = pd.read_csv(args.input_path, sep='\t')
candidate_list = pd.read_csv(args.known_label, sep='\t')
candidate_list['node'] = candidate_list['node'].apply(lambda x: x.replace('_', ' '))

candidate_list = candidate_list['node'].tolist()
candidate_list = ['It is an novel class'] + candidate_list
candidate_list = [f"{i:03}.{v}" for i, v in enumerate(candidate_list)]

choose_idx = len(final_ans) // args.sample_num

def gen(text):
    if args.dataset_name != 'thucnews':
        ans = f"""
            You are an expert in text classification with a focus on Out-of-Distribution (OOD) detection. Given a piece of text and a list of candidate categories, identify the most suitable category for the text from the provided list.

            Text to Classify:
            {text}.

            Questions: Given the candidate categories: {candidate_list}, which category does the text belong to? 
            Please strictly follow the following format for your answer: 
            
            Answer: xxx.xxxx.
            """
    else:
        ans = f"""
            您是一位专注于分布外（OOD）检测的文本分类专家。给定一段文本和一个候选类别列表，请从提供的列表中识别出该文本最合适的类别。

            待分类文本：
            {text}。

            问题：给定候选类别：{candidate_list}，该文本属于哪个类别？
            请严格按照以下格式回答：

            答案：xxx.xxxx。
            """
    return ans

final_ans['prompt'] = final_ans['text'].apply(gen)

start_time = time.time()
run_num = 0  # 记录实际执行的 API 数量
futures = []  # 存储提交的 Future 任务

def process_task(model_name, content, file_path):
    result = v3api(model_name, content)
    with open(file_path, 'w') as w:
        w.write(result)
    return result

# 统计任务数量

compute_idx_list = []
for idx in range(len(final_ans)):
    if idx % choose_idx != 0:
        continue
    output_file = f"{args.output_dir}/{idx}.txt"
    if os.path.exists(output_file):
        continue
    compute_idx_list.append(idx)


# 初始化 tqdm 进度条
with tqdm(total=len(compute_idx_list), desc="Processing", dynamic_ncols=True) as pbar:
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for idx in compute_idx_list:
            output_file = f"{args.output_dir}/{idx}.txt"
            # input_file = f"{args.input_dir}/{idx}.txt"
            # with open(input_file, 'w') as w:
            #     w.write(final_ans['prompt'][idx])
            future = executor.submit(process_task, args.model_name, final_ans['prompt'][idx], output_file)
            futures.append(future)

        for future in as_completed(futures):
            run_num += 1
            pbar.update(1)  # 实时更新进度条
            pbar.set_description(f'current_time: {time.time() - start_time:.2f}s, num: {run_num}')  # 避免 print 影响进度条
