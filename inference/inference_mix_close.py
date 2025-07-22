import sys,os
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
from answer_prompt_mix import *
from gpt_eval_prompt import Eval_Prompt
from gpt_eval import *
from llm_api import *
from metrics.qa_metrics import QAMetric
from utils.common_util import *
from utils.chart_process import *
from utils.chart_metric_util import *
from matplotlib import pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

file_extensions = {
    "latex": "txt",
    "markdown": "md",
    "csv": "csv",
    "html": "html"
}

@timeout(15)
def execute(c):
    exec(c)

@timeout(20)
def exec_and_get_y_reference(answer_code, chart_type):
    ecr_1 = False
    python_code, eval_code = build_eval_code(answer_code, chart_type)
    print("Code:", python_code)
    if python_code == "":
        return "", False
    try:
        python_code = surround_pycode_with_main(python_code)
        execute(python_code)
        ecr_1 = True
        plt.close('all')
    except Exception as e:
        print("Python Error: ",e)
        ecr_1 = False
        return "", False
    if ecr_1:
        pass
    try: 
        from io import StringIO
        output = StringIO()
        stdout = sys.stdout
        try:
            sys.stdout = output
            chart_eval_code = surround_pycode_with_main(eval_code)
            execute(chart_eval_code)
        except Exception as e:
            print("Eval Error: ",e)
            return "", True
        finally:
            sys.stdout = stdout
        output_value = output.getvalue()
        print("OUTPUT VALUE: ",output_value)
    except Exception as e:
        print("Eval Error: ",e)
        output_value = ''

    if output_value != '':
        parsed_prediction = output_value.strip()
    else:
        parsed_prediction = ''
    plt.close('all')
    return parsed_prediction, ecr_1

def build_messages(query, answer_format, opt):
    file_format = opt.format
    file_path = os.path.abspath(f'../data/{file_format}')
    question_type = query['QuestionType']
    if question_type == 'Data Analysis': TASK_PROMPT = Answer_Prompt[query['SubQType']]
    else : TASK_PROMPT = Answer_Prompt[question_type]
    QUESTION_PROMPT = User_Prompt.format_map({
        'Question': query['Question'],
        'Table': read_file(f'{file_path}/{query["FileName"]}.{file_extensions[opt.format]}'),
        "Answer_format": answer_format
    })
    messages = TASK_PROMPT + "\n" + QUESTION_PROMPT
    return messages

def get_answer_format(query):
    answer_format = ""
    if query['SubQType'] == 'Exploratory Analysis': answer_format = "CorrelationRelation, CorrelationCoefficient"
    elif query['QuestionType'] == 'Visualization': answer_format = "import pandas as pd \n import matplotlib.pyplot as plt \n ... plt.show()"
    else: answer_format = "AnswerName1, AnswerName2..."
    return answer_format

def gen_solution(opt):

    start_time  = datetime.datetime.now()
    dataset_path = os.path.abspath(f'../data')
    with open(f'{dataset_path}/QA_final.json', 'r') as fp:
        dataset = json.load(fp)
        querys = dataset['queries']  
    
    output_file_path = os.path.abspath(f'../result')
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)
        os.chmod(output_file_path, 0o777)  

    output_file_path = f'{output_file_path}/close_source'
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path) 
        os.chmod(output_file_path, 0o777)

    file_path = os.path.abspath(f'../data/{opt.format}')
    image_file_path = os.path.abspath(f'../data/image')
    table_file_path = os.path.abspath(f'../data/tables')

    all_eval_results = []
    for query in tqdm(querys):
        try:
            print("----------------------------- Current query: {} --------------------------".format(query['id']))
            question_type = query['QuestionType']
            answer_format = get_answer_format(query)
            messages = build_messages(query, answer_format, opt)
            image_file = f'{image_file_path}/{query["FileName"]}.png'
            
            metric_scores = {}
            if question_type == 'Visualization':
                response = get_final_answer_mlm(messages, answer_format, image_file, opt)
                reference = query['ProcessedAnswer']
                chart_type = query['SubQType'].split()[0]
                python_code = re.sub(r"'[^']*\.xlsx'", "'"+table_file_path+"/"+query['FileName']+".xlsx'", response)
                python_code = python_code.replace("table.xlsx", table_file_path+"/"+query['FileName']+".xlsx")
                prediction, ecr_1 = exec_and_get_y_reference(python_code, chart_type)
                metric_scores['ECR'] = ecr_1
                if prediction != '':
                    try:
                        prediction = ast.literal_eval(prediction)
                        reference = ast.literal_eval(reference)
                        if chart_type == 'PieChart': metric_scores['Pass'] = compute_pie_chart_metric(reference, prediction)
                        else: metric_scores['Pass'] = compute_general_chart_metric(reference, prediction)
                    except Exception as e:
                        metric_scores['Pass'] = 'False'
                else:
                    metric_scores['Pass'] = 'None'
            else:
                reference = query['FinalAnswer']
                if question_type == 'Data Analysis':
                    response = get_final_answer_mlm(messages, answer_format, image_file, opt)
                    if query['SubQType'] == 'Summary Analysis' or query['SubQType'] == 'Anomaly Analysis':
                        eval_prompt = Eval_Prompt[query['SubQType']].format_map({
                            'Question': query['Question'],
                            'Table': read_file(f'{file_path}/{query["FileName"]}.{file_extensions[opt.format]}'),
                            'Reference_Answer': query['FinalAnswer'],
                            'User_Answer': response
                        })
                    else:
                        eval_prompt = Eval_Prompt[query['SubQType']].format_map({
                            'Question': query['Question'],
                            'Reference_Answer': query['FinalAnswer'],
                            'User_Answer': response
                        })
                    prediction = response
                    eval_score = get_eval_score([eval_prompt], opt)
                    metric_scores = qa_metric.compute([reference], [prediction])
                    metric_scores['GPT_EVAL'] = eval_score
                elif question_type == 'Structure Comprehending':
                    image_file = f'{image_file_path}/{query["FileName"]}.png'
                    messages = build_messages(query, answer_format, opt)
                    reference = get_final_answer_mlm(messages, answer_format, image_file, opt)
                    query["FileName"] = query["FileName"] + "_swap"
                    image_file = f'{image_file_path}/{query["FileName"]}.png'
                    messages = build_messages(query, answer_format, opt)
                    response = get_final_answer_mlm(messages, answer_format, image_file, opt)
                    prediction = response
                    metric_scores = qa_metric.compute([reference], [prediction])
                else:
                    response = get_final_answer_mlm(messages, answer_format, image_file, opt)
                    prediction = response
                    metric_scores = qa_metric.compute([reference], [prediction])
            
            eval_result = {
                'Id': query['id'],
                'QuestionType': query['QuestionType'],
                'Model_Answer': response,
                'Correct_Answer': query['FinalAnswer'],
                'F1': metric_scores.get('F1', ''),
                'EM': metric_scores.get('EM', ''),
                'ROUGE-L': metric_scores.get('ROUGE-L', ''),
                'SacreBLEU': metric_scores.get('SacreBLEU', ''),
                'GPT_EVAL': metric_scores.get('GPT_EVAL', ''),
                'ECR': metric_scores.get('ECR', ''),
                'Pass': metric_scores.get('Pass', '')
            }
            all_eval_results.append(eval_result)
        
        except Exception as e:
            print(str(e))

    print(all_eval_results)
    
    print(output_file_path)
    end_time = datetime.datetime.now()
    print(f"Total time taken: {end_time - start_time}")

    df = pd.DataFrame(all_eval_results)
    df.to_csv(f'{output_file_path}/{opt.model}_mix_{end_time}.csv', index=False)

    

def parse_option():
    parser = argparse.ArgumentParser("command line arguments for generation.")
    
    parser.add_argument('--model', type=str, default='', help='model name')
    parser.add_argument('--api_key', type=str, default="", help='the api key of model')
    parser.add_argument('--base_url', type=str, default="", help='the base url of model')
    parser.add_argument('--local_rank', dest='local_rank', default=0, type=int, help='node rank for distributed training')
    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    opt = parse_option()
    print(opt)
    qa_metric = QAMetric()
    
    gen_solution(opt)
