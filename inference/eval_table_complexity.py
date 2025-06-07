import sys,os
import concurrent.futures
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

import datetime
import json
import ast
import argparse
import pandas as pd
from tqdm import tqdm
import base64
import time
from eval_complexity_prompt import *
#from gpt_eval_prompt import Eval_Prompt
from llm_api import *
# from llm_local import *
from gpt_eval import get_eval_score
from metrics.qa_metrics import QAMetric
from utils.common_util import *
from utils.chart_process import *
from utils.chart_metric_util import *
from matplotlib import pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def gen_solution(opt):

    start_time  = datetime.datetime.now()

    input_folder = ""
    output_folder = ""
    all_eval_results = []
    for file_name in tqdm(os.listdir(input_folder)):
        print("----------------------------- Current Id: {} --------------------------".format(id))
        file_name = file_name.split(".")[0]
        prompt = Eval_Prompt.format_map({
            'Table': read_txt_file(f'{input_folder}/{file_name}.html')
        })
        messages = [prompt]
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(get_final_answer_complexity, messages, opt)
                result_dict = future.result(timeout=120)
                ColumnHeaderHierarchy = result_dict.get('ColumnHeaderHierarchy', 0)
                RowHeaderHierarchy = result_dict.get('RowHeaderHierarchy', 0)
                ContentPartition = result_dict.get('ContentPartition', 0)
                StructureCompound = result_dict.get('StructureCompound', 0)
                VisualComplexity = result_dict.get('VisualComplexity', 0)
                FinalScore = (ColumnHeaderHierarchy + RowHeaderHierarchy + ContentPartition + StructureCompound + VisualComplexity)/5
                eval_result = {
                    'Id': id,
                    'File_Name': file_name,
                    'ColumnHeaderHierarchy': ColumnHeaderHierarchy,
                    'RowHeaderHierarchy': RowHeaderHierarchy,
                    'ContentPartition': ContentPartition,
                    'StructureCompound': StructureCompound,
                    'VisualComplexity': VisualComplexity,
                    'FinalScore': FinalScore
                }
                print(eval_result)
                all_eval_results.append(eval_result)
        except concurrent.futures.TimeoutError:
            print(id)
            continue
        except Exception as e:
            print(e)
            continue

    df = pd.DataFrame(all_eval_results)
    end_time = datetime.datetime.now()
    df.to_csv(f'{output_folder}/AITQA_{end_time}.csv', index=False)
    print(f"Total time taken: {end_time - start_time}")

def parse_option():
    parser = argparse.ArgumentParser("command line arguments for generation.")
    parser.add_argument('--model', type=str, default='', help='model name')
    parser.add_argument('--api_key', type=str, default="", help='the api key of model')
    parser.add_argument('--base_url', type=str, default="", help='the base url of model')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_option()
    print(opt)
    
    gen_solution(opt)
