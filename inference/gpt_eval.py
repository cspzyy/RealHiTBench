from typing import List
from openai import OpenAI
import re
import time

def get_gpt_response(messages: List[str], opt):
    
    client = OpenAI(api_key=opt.api_key, base_url=opt.base_url)
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": messages[i]} for i in range(len(messages))]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="gpt-4o",
    )
    return chat_completion.choices[0].message.content

def get_eval_score(messages, opt, sleep_time=5, max_retry=5):
    retry = 0
    while retry < max_retry:
        response = get_gpt_response(messages, opt)
        messages.append(response)
        if "[Score]:" in response:
            eval_score = response.split("[Score]:")[-1].strip()
            if "/" in eval_score: eval_score = eval_score.split("/")[0].strip()
            return eval_score
        elif "Score:" in response:
            eval_score = response.split("Score:")[-1].strip()
            if "/" in eval_score: eval_score = eval_score.split("/")[0].strip()
            return eval_score
        else:
            retry += 1
            print("No 'Score' found, requesting again...")
            messages.append('Note: Please check your output format. You do not need to do much explaining, just give the final answer in the given format: "[Score]: xx/100".')
            time.sleep(sleep_time)
        if retry == max_retry: return response



