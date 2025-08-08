from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, LlavaForConditionalGeneration, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from typing import List
from utils.common_util import *
import time
import re
import os

from PIL import Image
import requests
import torch
from torchvision import io
from typing import Dict
from transformers import Qwen2VLForConditionalGeneration
from vllm import LLM, SamplingParams

# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import get_model_name_from_path
# from llava.eval.run_llava import eval_model

from typing import Any

device = 'cuda'

def load_model(opt):
    config = AutoConfig.from_pretrained(opt.model_dir)
    model_type = config.model_type.lower()

    tokenizer = AutoTokenizer.from_pretrained(opt.model_dir, use_fast=False)
    
    model = AutoModelForCausalLM.from_pretrained(
        opt.model_dir, 
        torch_dtype=torch.bfloat16,
        device_map='auto',
        offload_folder="offload_dir",
        offload_state_dict=True,
    )
    return tokenizer, model

def load_llama_vl_model(opt):
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     opt.model_dir,
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )
    model = LLM(model=opt.model_dir,tensor_parallel_size=4,dtype="bfloat16",enable_prefix_caching=True,enable_chunked_prefill=True)
    # tokenizer = AutoTokenizer.from_pretrained(opt.model_dir)
    return model

def load_llava_model(opt):
    model_path = opt.model_dir
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path= model_path,
        model_base= None,
        model_name=get_model_name_from_path(model_path),
    )
    return model, image_processor

def load_qwen_vl_model(opt):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        opt.model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(opt.model_dir)
    return model, processor

def get_llm_response(messages: List[str], tokenizer, model):
    model.eval()
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": messages[i]} for i in range(len(messages))]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_input = tokenizer([text], return_tensors='pt').to(device)
    device = next(model.parameters()).device
    model_input = model_input.to(device)
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

def get_llama_vl_response(messages, image_file, model, tokenizer):
    prompt = "USER: <image>\n"+messages+"\nASSISTANT"
    image = Image.open(image_file).convert('RGB')
    max_edge = max(image.size)
    image = image.resize((max_edge, max_edge))
    inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image
            },
        }
    sampling_params = SamplingParams(temperature=0, max_tokens=32000,stop_token_ids=[tokenizer.eos_token_id])
    outputs = model.generate(inputs,sampling_params)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)

def get_llava_response(prompt, image_file, tokenizer, model, image_processor, opt):
    model_path = opt.model_dir
    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0.2,
        "top_p": None,
        "num_beams": 1,
        'max_new_tokens': 4096
    })()
    response = eval_model(args, tokenizer, model, image_processor)
    return response

def get_qwen_vl_response(messages, image_file, model, processor):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": messages},
                {
                    "type": "image",
                }
            ],
        }
    ]
    image = Image.open(image_file).convert('RGB')
    max_edge = max(image.size)
    image = image.resize((max_edge, max_edge))
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt], images=[image], padding=True, return_tensors="pt"
    )
    inputs = inputs.to("cuda")
    output_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    return output_text

def get_tablellm_response(messages: List[str], tokenizer, model):
    model.eval()
    prompt = messages[0]
    
    model_input = tokenizer([prompt], return_tensors='pt').to(device)
    device = next(model.parameters()).device
    model_input = model_input.to(device)
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
    device = next(model.parameters()).device
    model_input = model_input.to(device)
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
        
def get_multimodal_final_answer(messages, image_file, answer_format, tokenizer, model, image_processor, opt, sleep_time=5, max_retry=5):
    retry = 0
    while retry < max_retry:
        config = AutoConfig.from_pretrained(opt.model_dir)
        model_type = config.model_type.lower()

        response: Any = ""

        if model_type == 'llama3_2_vl':
            response = get_llama_vl_response(messages, image_file, model, tokenizer)
        elif model_type == 'llava':
            response = get_llava_response(messages, image_file, tokenizer, model, image_processor, opt)
        elif model_type == 'qwen2_vl':
            response = get_qwen_vl_response(messages, image_file, model, image_processor)
        else:
            raise ValueError(f"The model-specific loading script has not yet been configured; please consult the model's documentation.")

        # response = get_llava_response(messages, image_file, tokenizer, model, image_processor, opt)
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
            messages = messages + '\n Note: Please check your output format. You do not need to do much explaining, just give the final answer in the given format: "[Final Answer]: '+ answer_format
            time.sleep(sleep_time)
        if retry == max_retry: return response

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