from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from typing import List
from utils.common_util import *
import time
import re
import os

device = 'cuda'

def load_model(opt):
    tokenizer = AutoTokenizer.from_pretrained(opt.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        opt.model_dir, 
        torch_dtype=torch.bfloat16,
        device_map='auto',
        offload_folder="offload_dir",
        offload_state_dict=True,
    )
    return tokenizer, model

def get_llm_response(messages: List[str], tokenizer, model):
    model.eval()
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": messages[i]} for i in range(len(messages))]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_input = tokenizer([text], return_tensors='pt').to(device)
    attention_mask = torch.ones(model_input.input_ids.shape, dtype=torch.long, device=device)
    generated_ids = model.generate(
        model_input.input_ids,
        max_new_tokens=4096,
        attention_mask=attention_mask,  
        pad_token_id=tokenizer.eos_token_id,
        temperature = 0.6,
        use_cache = True
    )

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_input.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def get_tablellm_response(messages: List[str], tokenizer, model):
    model.eval()
    prompt = messages[0]
    
    model_input = tokenizer([prompt], return_tensors='pt').to(device)
    attention_mask = torch.ones(model_input.input_ids.shape, dtype=torch.long, device=device)
    generated_ids = model.generate(
        model_input.input_ids,
        max_new_tokens=1024,
        attention_mask=attention_mask,  
        pad_token_id=tokenizer.eos_token_id,
        temperature =0.001
    )
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_input.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    return response

def get_tablellama_answer(messages: List[str], tokenizer, model, query):
    model.eval()
    PROMPT_DICT = {
        "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input_seg}\n\n### Question:\n{question}\n\n### Response:"
        ),
    }
    latex_file_path = os.path.abspath(f'../data/latex')
    
    question_prompt = PROMPT_DICT["prompt_input"].format(instruction = messages[0], input_seg = "", question = query['Question'])
    input_part1 = len(tokenizer.encode(question_prompt)) + 100
    table_content = read_file(f'{latex_file_path}/{query["FileName"]}.txt')
    truncated_table_tokens = tokenizer.encode(table_content, max_length = 8192 - input_part1, truncation=True, add_special_tokens=False)
    truncated_table_content = tokenizer.decode(truncated_table_tokens, skip_special_tokens=True)
    prompt = PROMPT_DICT["prompt_input"].format(instruction = messages[0], input_seg = truncated_table_content, question = query['Question'])
    model_input = tokenizer(prompt, return_tensors='pt').to(device)
    output = model.generate(
        **model_input,
        max_new_tokens=1024
    )
    out = tokenizer.decode(output[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    response = out.split(prompt)[1].strip().split("</s>")[0]
    return response

def get_final_answer(messages, answer_format, tokenizer, model, sleep_time=5, max_retry=3):
    retry = 0
    while retry < max_retry:
        response = get_llm_response(messages, tokenizer, model)
        messages.append(response)
        if "[Final Answer]:" in response:
            final_answer = response.split("[Final Answer]:")[-1].strip()
            return final_answer
        elif "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
            return final_answer
        else:
            retry += 1
            print("No 'Final Answer' found, requesting again...")
            messages[len(messages)-1] += '\nNote: Please check your output format. You should follow the given format: "[Final Answer]: '
            time.sleep(sleep_time)
        if retry == max_retry: return ""
        
def get_final_answer_tablellama(messages, answer_format, tokenizer, model, query, sleep_time=5, max_retry=5):
    retry = 0
    while retry < max_retry:
        response = get_tablellama_answer(messages, tokenizer, model, query)
        messages.append(response)
        if "[Final Answer]:" in response:
            final_answer = response.split("[Final Answer]:")[-1].strip()
            return final_answer
        elif "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
            return final_answer
        else:
            retry += 1
            print("No 'Final Answer' found, requesting again...")
            messages.append('Note: Please check your output format. You do not need to do much explaining, just give the final answer in the given format: "[Final Answer]: '+ answer_format)
            time.sleep(sleep_time)
        if retry == max_retry: return response