Rudimentary_Analysis = """
Suppose you are an expert in table analysis and your task is to rate the user answer on one metric based on the table content, question and corresponding reference answer.  
You will be given table content and a question about rudimentary analysis of the table. And the corresponding reference answer to a question and the answer from the user.  
Your task is to rate the answer on one metric.  
Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.  
Evaluation Criteria:  
Correctness (1-100) - the answer should be as close as possible to the reference answer, with perfectly equal answers receiving full marks, smaller differences receiving higher marks, and larger differences receiving only lower marks.
Evaluation Steps:  
1. Read the table carefully and fully understand the contents of the table.
2. Read the result and compare it to the reference answer and the table. Determine if the answer is correct and if not score it based on how different it is from the correct answer.
3. Assign a score for correctness on a scale of 0 to 100, where 0 is the lowest and 100 is the highest based on the Evaluation Criteria.   
Question:
{Question}
Reference Answer:
{Reference_Answer}
User Answer:
{User_Answer}
Emphasize: you need to make sure your final answer is formatted in this way: [Score]: xx/100
"""

Summary_Analysis = """
Suppose you are an expert in table analysis and your task is to rate the user answer on one metric based on the table content, question and corresponding reference answer.  
You will be given table content and a question about summary analysis of the table. And the corresponding reference answer to a question and the answer from the user.  
Your task is to rate the answer on one metric.  
Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.  
Evaluation Criteria:  
Coherence (1-100) - the collective quality of all sentences. The summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to sentence to a coherent body of information about a topic. A high scoring analysis should be able to cover all parts of the problem that need to be summarized, comprehensively and correctly
Evaluation Steps:  
1. Read the table carefully and fully understand the contents of the table.
2. Read the result and compare it to the reference answer and the table. Determine if the answer is correct and if not score it based on how different it is from the correct answer.
3. Assign a score for correctness on a scale of 0 to 100, where 0 is the lowest and 100 is the highest based on the Evaluation Criteria.   
Table:
{Table}
Question:
{Question}
Reference Answer:
{Reference_Answer}
User Answer:
{User_Answer}
Emphasize: you need to make sure your final answer is formatted in this way: [Score]: xx/100
"""

Predictive_Analysis = """
Suppose you are an expert in table analysis and your task is to rate the user answer on one metric based on the table content, question and corresponding reference answer.  
You will be given table content and a question about predictive analysis of the table. And the corresponding reference answer to a question and the answer from the user.  
Your task is to rate the answer on one metric.  
Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.  
Evaluation Criteria:  
Correctness (1-100) - the answer should be as close as possible to the reference answer, with perfectly equal answers receiving full marks, smaller differences receiving higher marks, and larger differences receiving only lower marks.
Evaluation Steps:  
1. Read the table carefully and fully understand the contents of the table.
2. Read the result and compare it to the reference answer and the table. Determine if the answer is correct and if not score it based on how different it is from the correct answer.
3. Assign a score for correctness on a scale of 0 to 100, where 0 is the lowest and 100 is the highest based on the Evaluation Criteria.
Question:
{Question}
Reference Answer:
{Reference_Answer}
User Answer:
{User_Answer}
Emphasize: you need to make sure your final answer is formatted in this way: [Score]: xx/100
"""

Exploratory_Analysis = """
Suppose you are an expert in table analysis and your task is to rate the user answer on one metric based on the table content, question and corresponding reference answer.  
You will be given table content and a question about exploratory analysis of the table. And the corresponding reference answer to a question and the answer from the user.  
Your task is to rate the answer on one metric.  
Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.  
Evaluation Criteria:  
Correctness (1-100) - the answer should be as close as possible to the reference answer, with perfectly equal answers receiving full marks, smaller differences receiving higher marks, and larger differences receiving only lower marks.
Evaluation Steps:  
1. Read the table carefully and fully understand the contents of the table.
2. Read the result and compare it to the reference answer and the table. Determine if the answer is correct and if not score it based on how different it is from the correct answer.
3. Assign a score for correctness on a scale of 0 to 100, where 0 is the lowest and 100 is the highest based on the Evaluation Criteria.   
Question:
{Question}
Reference Answer:
{Reference_Answer}
User Answer:
{User_Answer}
Emphasize: you need to make sure your final answer is formatted in this way: [Score]: xx/100
"""

Anomaly_Analysis = """
Suppose you are an expert in table analysis and your task is to rate the user answer on one metric based on the table content, question and corresponding reference answer.  
You will be given table content and a question about exploratory analysis of the table. And the corresponding reference answer to a question and the answer from the user.  
Your task is to rate the answer on one metric.  
Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.  
Evaluation Criteria:  
Correctness (1-100) - the answer should be as close as possible to the reference answer, with perfectly equal answers receiving full marks, smaller differences receiving higher marks, and larger differences receiving only lower marks.
Evaluation Steps:  
1. Read the table carefully and fully understand the contents of the table.
2. Read the result and compare it to the reference answer and the table. Determine if the answer is correct and if not score it based on how different it is from the correct answer.
3. Assign a score for correctness on a scale of 0 to 100, where 0 is the lowest and 100 is the highest based on the Evaluation Criteria.   
Table:
{Table}
Question:
{Question}
Reference Answer:
{Reference_Answer}
User Answer:
{User_Answer}
Emphasize: you need to make sure your final answer is formatted in this way: [Score]: xx/100
"""

Eval_Prompt = {
    "Rudimentary Analysis": Rudimentary_Analysis,
    "Summary Analysis": Summary_Analysis,
    "Predictive Analysis": Predictive_Analysis,
    "Exploratory Analysis": Exploratory_Analysis,
    "Anomaly Analysis": Anomaly_Analysis
}

