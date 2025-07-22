Fact_Checking = """
# Role Play
Suppose you are an expert in table analysis and your task is to provide answers to questions based on the content of the table.
# Chain-of-thought
Let’s think step by step as follows and make the most of your strengths as a table analysis expert：
1、Fully understand the question and extract the necessary information from it.
2、Clearly and comprehensively understanding the content of  the table, including the structure of the table, the meaning and formatting of each row and column header（Note: There is usually summative cell in the table, such as all, combine, total, sum, average, mean, etc. Please pay careful attention to the flag information in the row header and column header, this information can help you to skip many operations.）
3、Based on the question, select the row and column headers in the table that are most relevant to it and find the corresponding cells based on them.
4、According to the requirements of the question, perform statistical, calculation, ranking, or other operations on the cells you selected, and output of the answer in the format specified by the definition.
# Output Control
1、The final answer should be one-line and strictly follow the format: "[Final Answer]: AnswerName1, AnswerName2...". 
2、Ensure the "AnswerName" is a number or entity name, as short as possible, without any explanation. Give the final answer to the question directly without any explanation. If the question is judgmental, please answer 'Yes' or 'No'.
I will give you a table in {format} format and a question, please use the table to answer the question in specified format: [Final Answer]: AnswerName1, AnswerName2...
"""

Numerical_Reasoning = """
# Role Play
Suppose you are an expert in table analysis and your task is to provide answers to questions based on the content of the table.
# Chain-of-thought 
Let’s think step by step as follows and make the most of your strengths as a table analysis expert：
1、Fully understand the question and extract the necessary information from it.
2、Clearly and comprehensively understanding the content of  the table, including the structure of the table, the meaning and formatting of each row and column header（Note: There is usually summative cell in the table, such as all, combine, total, sum, average, mean, etc. Please pay careful attention to the flag information in the row header and column header, this information can help you to skip many operations.）
3、Based on the question, select the row and column headers in the table that are most relevant to it and find the corresponding cells based on them.
4、According to the requirements of the question, perform statistical, calculation, ranking, or other operations on the cells you selected, and output of the answer in the format specified by the definition.
# Output Control
1、The final answer should be one-line and strictly follow the format: "[Final Answer]: AnswerName1, AnswerName2...". 
2、Ensure the "AnswerName" is a number or entity name, as short as possible, without any explanation. Give the final answer to the question directly without any explanation. Note: If the answer involves decimals, always keep it to two decimals.
I will give you a table in {format} format and a question, please use the table to answer the question in specified format: [Final Answer]: AnswerName1, AnswerName2...
"""

Rudimentary_Analysis = """
# Role Play
Suppose you are an expert in table analysis and your task is to provide answers to questions based on the content of the table.
# Chain-of-thought 
Let’s think step by step as follows and make the most of your strengths as a table analysis expert：
1、Fully understand the question and extract the necessary information from it.
2、Clearly and comprehensively understanding the content of  the table, including the structure of the table, the meaning and formatting of each row and column header（Note: There is usually summative cell in the table, such as all, combine, total, sum, average, mean, etc. Please pay careful attention to the flag information in the row header and column header, this information can help you to skip many operations.）
3、Based on the question, select the row and column headers in the table that are most relevant to it and find the corresponding cells based on them.
4、According to the requirements of the question, perform statistical, calculation, ranking, or other operations on the cells you selected, and output of the answer in the format specified by the definition.
# Output Control
1、You only need to output the final answer without any interpretation. 
2、The final answer should be one-line and strictly follow the format: "[Final Answer]: AnswerName1, AnswerName2...". Note: If the answer involves decimals, always keep it to two decimals.
3、The "AnswerName" should represent the primary result of the rudimentary analysis, such as a number or an entity name, expressed as concisely as possible. Provide the final answer directly without additional explanation or extra output.
I will give you a table in {format} format and a question, please use the table to answer the question in specified format: [Final Answer]: AnswerName1, AnswerName2...
"""

Summary_Analysis = """
# Role Play
Suppose you are an expert in table analysis and your task is to provide answers to questions based on the content of the table.
# Chain-of-thought 
Let’s think step by step as follows and make the most of your strengths as a table analysis expert：
1、Fully understand the question and extract the necessary information from it.
2、Clearly and comprehensively understanding the content of  the table, including the structure of the table, the meaning and formatting of each row and column header（Note: There is usually summative cell in the table, such as all, combine, total, sum, average, mean, etc. Please pay careful attention to the flag information in the row header and column header, this information can help you to skip many operations.）
3、Based on the question, select the row and column headers in the table that are most relevant to it and find the corresponding cells based on them.
4、According to the requirements of the question, perform statistical, calculation, ranking, or other operations on the cells you selected, and output of the answer in the format specified by the definition.
# Output Control
1、You only need to output the final answer without any interpretation. 
2、The final answer should be one-line and strictly follow the format: "[Final Answer]: TableSummary". 
3、The "TableSummary" should provide a concise summary of the table, including a brief description of its content, the main columns, and any basic insights derived. Provide the final answer directly without additional explanation or extra output.
I will give you a table in {format} format and a question, please use the table to answer the question in specified format: [Final Answer]: TableSummary.
"""

Predictive_Analysis = """
# Role Play
Suppose you are an expert in table analysis and your task is to provide answers to questions based on the content of the table.
# Chain-of-thought 
Let’s think step by step as follows and make the most of your strengths as a table analysis expert：
1、Fully understand the question and extract the necessary information from it.
2、Clearly and comprehensively understanding the content of  the table, including the structure of the table, the meaning and formatting of each row and column header（Note: There is usually summative cell in the table, such as all, combine, total, sum, average, mean, etc. Please pay careful attention to the flag information in the row header and column header, this information can help you to skip many operations.）
3、Based on the question, select the row and column headers in the table that are most relevant to it and find the corresponding cells based on them.
4、According to the requirements of the question, perform statistical, calculation, ranking, or other operations on the cells you selected, and output of the answer in the format specified by the definition.
# Output Control
1、You only need to output the final answer without any interpretation. 
2、The final answer should be one-line and strictly follow the format: "[Final Answer]: AnswerName1, AnswerName2...". Note: If the answer involves decimals, always keep it to two decimals.
3、The "AnswerName" should summarize the primary result of the analysis in a concise manner, such as a number, an entity name, or a trend description (e.g., No clear trend, Increasing trend, Decreasing trend).  Provide the final answer directly without additional explanation or extra output. Note: If the final answer has multiple decimals, retain two decimals. 
I will give you a table in {format} format and a question, please use the table to answer the question in specified format: [Final Answer]: AnswerName1, AnswerName2...
"""

Exploratory_Analysis = """
# Role Play
Suppose you are an expert in table analysis and your task is to provide answers to questions based on the content of the table.
# Chain-of-thought 
Let’s think step by step as follows and make the most of your strengths as a table analysis expert：
1、Fully understand the question and extract the necessary information from it.
2、Clearly and comprehensively understanding the content of  the table, including the structure of the table, the meaning and formatting of each row and column header（Note: There is usually summative cell in the table, such as all, combine, total, sum, average, mean, etc. Please pay careful attention to the flag information in the row header and column header, this information can help you to skip many operations.）
3、Based on the question, select the row and column headers in the table that are most relevant to it and find the corresponding cells based on them.
4、According to the requirements of the question, perform statistical, calculation, ranking, or other operations on the cells you selected, and output of the answer in the format specified by the definition.
# Output Control
1、You only need to output the final answer without any interpretation. 
2、The final answer should be one-line and strictly follow the format: "[Final Answer]: CorrelationRelation, CorrelationCoefficient." Note: If the answer involves decimals, always keep it to two decimals.
3、Ensure that: the correlation coefficient should be a float number with two decimal places; the correlation relation can only be "No correlation" with the correlation coefficient between -0.3 to +0.3, "Weak positive correlation" with the correlation coefficient between +0.3 to +0.7, "Weak negative correlation" with the correlation coefficient between -0.3 to -0.7, "Strong positive correlation" with the correlation coefficient between +0.7 to +1, or "Strong negative correlation" with the correlation coefficient between -0.7 to -1. If the question is about impact analysis, the "AnswerName" should be a entity name or a impact description(No clear impact, Negtive impact or Positive impact), as short as possible, without any explanation. If the question is about causal analysis, the "AnswerName" should be a brief explanation of the causal analysis results as concise as possible. Note: If the final answer has multiple decimals, retain two decimals. 
I will give you a table in {format} format and a question, please use the table to answer the question in specified format: [Final Answer]: CorrelationRelation, CorrelationCoefficient
"""

Anomaly_Analysis = """
# Role Play
Suppose you are an expert in table analysis and your task is to provide answers to questions based on the content of the table.
# Chain-of-thought 
Let’s think step by step as follows and make the most of your strengths as a table analysis expert：
1、Fully understand the question and extract the necessary information from it.
2、Clearly and comprehensively understanding the content of  the table, including the structure of the table, the meaning and formatting of each row and column header（Note: There is usually summative cell in the table, such as all, combine, total, sum, average, mean, etc. Please pay careful attention to the flag information in the row header and column header, this information can help you to skip many operations.）
3、Based on the question, select the row and column headers in the table that are most relevant to it and find the corresponding cells based on them.
4、According to the requirements of the question, perform statistical, calculation, ranking, or other operations on the cells you selected, and output of the answer in the format specified by the definition.
# Output Control
1、You only need to output the final answer without any interpretation. 
2、The final answer should be one-line and strictly follow the format: "[Final Answer]: Conclusion."
3、The "Conclusion" should provide a concise conclusion of the table anomaly. Provide the final answer directly without additional explanation or extra output.
I will give you a table in {format} format and a question, please use the table to answer the question in specified format: [Final Answer]: Conclusion
"""

Visulization = """
# Role Play
Suppose you are an expert in table analysis and your task is to generate the code to questions based on the content of the table.
# Chain-of-thought 
Let’s think step by step as follows and make the most of your strengths as a table analysis expert：
1、Fully understand the question and extract the necessary information from it.
2、Clearly and comprehensively understanding the content of  the table, including the structure of the table, the meaning and formatting of each row and column header（Note: There is usually summative cell in the table, such as all, combine, total, sum, average, mean, etc. Please pay careful attention to the flag information in the row header and column header, this information can help you to skip many operations.）
3、Based on the question, select the row and column headers in the table that are most relevant to it and find the corresponding cells based on them.
4、According to the requirements of the question, perform statistical, calculation, ranking, or other operations on the cells you selected, and output of the answer in the format specified by the definition.
# Output Control
1、The final answer should follow the format below and ensure the first three code lines is exactly the same with the following code block: [Final Answer]: import pandas as pd \n import matplotlib.pyplot as plt \n ... plt.show(). 
2、You only need to output the final code without any interpretation, make sure that your code can be run directly without any syntax errors.
3、Please make sure the table is named “table.xlsx”, and the pandas and matplotlib libraries have been successfully introduced.
4、Ensure that the X-axis used for drawing in the code is arranged in ascending alphabetical or numerical order. Ensure the last line in python code can only be "plt.show()", no other from. Give the final answer to the question directly without any explanation.
I will give you a table in {format} format and a question, please use the table to answer the question in specified format: [Final Answer]: import pandas as pd \n import matplotlib.pyplot as plt \n ... plt.show()
"""

Structure_Comprehending = """
# Role Play
Suppose you are an expert in table analysis and your task is to provide answers to questions based on the content of the table.
# Chain-of-thought 
Let’s think step by step as follows and make the most of your strengths as a table analysis expert：
1、Fully understand the question and extract the necessary information from it.
2、Clearly and comprehensively understanding the content of  the table, including the structure of the table, the meaning and formatting of each row and column header（Note: There is usually summative cell in the table, such as all, combine, total, sum, average, mean, etc. Please pay careful attention to the flag information in the row header and column header, this information can help you to skip many operations.）
3、Based on the question, select the row and column headers in the table that are most relevant to it and find the corresponding cells based on them.
4、According to the requirements of the question, perform statistical, calculation, ranking, or other operations on the cells you selected, and output of the answer in the format specified by the definition.
# Output Control
1、The final answer should be one-line and strictly follow the format: "[Final Answer]: AnswerName1, AnswerName2...". 
2、Ensure the "AnswerName" is a number or entity name, as short as possible, without any explanation. Give the final answer to the question directly without any explanation. If the question is judgmental, please answer 'Yes' or 'No'.
I will give you a table in {format} format and a question, please use the table to answer the question in specified format: [Final Answer]: AnswerName1, AnswerName2...
"""

User_Prompt = """ 
Let's get start! 
# Table
{Table}
# Question
{Question}
Emphasize: you need to make sure your final answer is formatted in this way: [Final Answer]: {Answer_format}
"""

Model_First_Response = """
Got it! Please provide the table in csv format along with the question.
"""

Answer_Prompt = {
    "Fact Checking": Fact_Checking,
    "Numerical Reasoning": Numerical_Reasoning,
    "Structure Comprehending": Structure_Comprehending,
    "Rudimentary Analysis": Rudimentary_Analysis,
    "Summary Analysis": Summary_Analysis,
    "Predictive Analysis": Predictive_Analysis,
    "Exploratory Analysis": Exploratory_Analysis,
    "Anomaly Analysis": Anomaly_Analysis,
    "Visualization": Visulization
}