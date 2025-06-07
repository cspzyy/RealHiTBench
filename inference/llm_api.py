from typing import List
from openai import OpenAI
from utils.common_util import *
import re
import time

def get_llm_response(messages: List[str], opt):
    client = OpenAI(api_key=opt.api_key, base_url=opt.base_url)
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": messages[i]} for i in range(len(messages))]
    chat_completion = client.chat.completions.create(
        messages=messages,
        max_tokens = 4096
    )
    return chat_completion.choices[0].message.content
def get_mlm_response(messages, image_file, opt):
    
    client = OpenAI(api_key=opt.api_key, base_url=opt.base_url)
    image = encode_image(image_file)
    messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": messages},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}"
                        }
                    }
                ]
            }
        ]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=opt.model,
        temperature=0,
    )
    return chat_completion.choices[0].message.content

def get_final_answer(messages, answer_format, opt, sleep_time=5, max_retry=3):
    retry = 0
    while retry < max_retry:
        response = get_llm_response(messages, opt)
        #messages.append(response)
        # 判断响应中是否包含 "Final Answer"
        if "[Final Answer]:" in response:
            final_answer = response.split("[Final Answer]:")[-1].strip()
            return final_answer
        elif "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
            return final_answer
        else:
            retry += 1
            print("No 'Final Answer' found, requesting again...")
            messages[len(messages)-1] += 'Note: Please check your output format. The ouput format should be: [Final Answer]: "' + answer_format
            time.sleep(sleep_time)
        if retry == max_retry: return response

def get_final_answer_mlm(messages, answer_format, image_file, opt, sleep_time=5, max_retry=5):
    retry = 0
    while retry < max_retry:
        response = get_mlm_response(messages, image_file, opt)
        if "[Final Answer]:" in response:
            final_answer = response.split("[Final Answer]:")[-1].strip()
            return final_answer
        elif "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
            return final_answer
        else:
            retry += 1
            print("No 'Final Answer' found, requesting again...")
            messages += '\n Note: Please check your output format. You do not need to do much explaining, just give the final answer in the given format: "[Final Answer]: '+ answer_format
            time.sleep(sleep_time)
        if retry == max_retry: return response

def get_final_answer_complexity(messages, opt, sleep_time=5, max_retry=5):
    retry = 0
    while retry < max_retry:
        response = get_llm_response(messages, opt)
        messages.append(response)
        pattern = r"\[(.*?)\]: ([0-9.]+)"
        matches = re.findall(pattern, response)
        keys = ['ColumnHeaderHierarchy', 'RowHeaderHierarchy', 'ContentPartition', 'StructureCompound', 'VisualComplexity']
        result_dict = {key: float(value) for key, value in matches}
        flag = True
        for key in keys: 
            if key not in result_dict.keys():
                flag =False
        if flag:
            return result_dict
        else:
            retry += 1
            print("No 'Final Answer' found, requesting again...")
            messages.append('Note: Please check your output format. Your output should follow the format strictly: "[TableSize]: xxx, [ColumnHeaderHierarchy]: xxx, [RowHeaderHierarchy]: xxx, [ContentPartition]: xxx, [StructureCompound]: xxx, [VisualComplexity]: xxx')
            time.sleep(sleep_time)
        if retry == max_retry: return {}


