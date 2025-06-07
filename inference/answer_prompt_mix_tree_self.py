Generate_Tree = """
You are tasked with performing detailed table analysis. Your task is to generate a hierarchical tree structure for the top-row and left-column headers based on a complex table.
(1). Task Description
[Reasoning Steps]
Your thought process is as follows:

- Understand the Table Structure: Provide a comprehensive description of the table, including the various levels of row and column headers and their corresponding meanings. Construct two distinct hierarchical trees: one for the row headers and one for the column headers. Each tree should accurately represent the levels and relationships of the headers.

- Traverse the Table: Analyze each row and column header to extract its content, indentation, and positions in the table. Identify merged cells and indentation, as they often indicate hierarchical relationships. Determine the parent-child relationships based on these visual cues and arrange the data under the correct parent node in both row and column header trees.

- Validate the Hierarchical Relationships: Iterate through both the row header tree and column header tree. Verify that the parent-child relationships are accurate and that the nodes are correctly placed within their respective hierarchies.

(2). Node Definition
You will be provided with a table in Image format. The table may contain complex structures, such as merged or nested cells. Your task is to encode each node of table header as a tuple T(t1, t2, t3, t4)
The first element t1 indicates it represents row header (R) or column header (C), along with its corresponding level. 
The second element t2 and third element t3 represent its start and end positions, while the fourth element t4 contains the value from the table. For example, a tuple (R0, 1, 2, City) indicates that it is a row header (R) at level 0, spanning from row 1 to row 2, with the value City.
Please Convert the table headers to list L=[T1, T2, ...]

(3). Tree Generate
1. Divide the tuples list L into groups based on their levels, such that all tuples with the same level are grouped together. Add a special ROOT node for rows and columns, each with a level of "-1".
2. For each tuple A in L. If the start and end positions of A are equal, mark A as a leaf node.
3. Otherwise, compare its T2 and T3 values with every closest higher-level and same flag tuple B. If tuple A is within the range of tuple B, then B is the parent-header of A.
4. Repeat steps 2 and 3 iteratively until all tuples in L are linked to their respective parent nodes (Tuples without parent node are linked to the ROOT node), forming a hierarchical Table-Header Tree H.

(4) Next, we will provide a latex format table and corresponding image for you to analyze the hierarchical structure for the table and please organize the table header tuples as a tree, which can help you better understand the table structure.
Let's get started!
[TABLE]
"""

Fact_Checking = """
You are a table analyst. Your task is to first extract relevant keywords based on the questions posed, identify related content from the previous tables, and match it with the corresponding headers in the structure tree. Then you need to answer questions based on the provided table content. 

(1) Your guidelines of thinking and output format control:
- Understand the Question: Begin by carefully reading the question to extract the essential information needed for answering. This helps ensure that you focus on the right aspects of the table in the next steps.
- Analyze the Table Content: Thoroughly examine the structure tree and original content of the table, paying close attention to both row and column headers, which may include special indicators such as "Total," "Sum," "Average," or other summary metrics. Be mindful of any rows or columns dedicated to aggregates, as these can provide quick answers without the need for detailed calculations. It's also crucial to recognize that the table might have complex structures, such as merged cells or semantic nesting, which could influence the interpretation of the data.
- Identify Relevant Data: With a clear understanding of the question, identify the rows and columns in the table that are most relevant to the inquiry. This involves locating the cells that correspond to the relevant headers, ensuring the selected data is directly related to the question at hand.
- Perform Necessary Analysis or Calculations: Once the relevant data is identified, perform any required operations, such as statistical analysis, mathematical calculations, ranking, or other necessary procedures. This will help you derive the needed insights and provide a comprehensive answer.

(2) Your output action pattern and output format.
Your output should follow a React-like pattern of thinking, which includes one or more cycles of "Thought/Action/Result", ultimately leading to a "Final Answer" on the last line.
[Action Patterns]
- Thought: Consider the next action based on the result of the previous one.
- Action: The action should always be a single processing action.
- Result: Simulate the action result, analyze the result, and decide whether to continue or stop.
(This "Thought/Action/Result" cycle can repeat multiple times.)

Verify the table, observations, and question thoroughly before providing the final answer.
When answering, if the final answer comes from the original format in the table, please use the original format from the table without modifying it.
Below is an example of an output format. You need to first output the relevant keywords, headers, and content related with the question, then go through a multi-round interaction of thought/action/result, and finally provide the final answer.
[Output Example]
Relevant Keywords: Keywords related to the table in question
Relavant Table Headers: column/row headers related with the question
Relavant Content: related table content
Thought: Your first round of thinking.
Action: The action of your first round.
Result: The observation and result of your first round of simulation.
Thought: Your second round of thinking. The 'Thought/Action/Result' cycle can repeat 1 or more times until the final answer is reached.
Action: The action of your second round.
Result: The observation and result of your second round of simulation.
Final Answer: Your output result, following the format "Final Answer: AnswerName1, AnswerName2...". The "AnswerName" should be a number or entity name, as short as possible.

Let's get start!
[Question]
{Question}
"""



Numerical_Reasoning = """
You are a table analyst. Your task is to first extract relevant keywords based on the questions posed, identify related content from the previous tables, and match it with the corresponding headers in the structure tree. Then you need to answer questions based on the provided table content. 

(1) Your guidelines of thinking and output format control:
- Understand the Question: Begin by carefully reading the question to extract the essential information needed for answering. This helps ensure that you focus on the right aspects of the table in the next steps.
- Analyze the Table Content: Thoroughly examine the structure tree and original content of the table, paying close attention to both row and column headers, which may include special indicators such as "Total," "Sum," "Average," or other summary metrics. Be mindful of any rows or columns dedicated to aggregates, as these can provide quick answers without the need for detailed calculations. It's also crucial to recognize that the table might have complex structures, such as merged cells or semantic nesting, which could influence the interpretation of the data.
- Identify Relevant Data: With a clear understanding of the question, identify the rows and columns in the table that are most relevant to the inquiry. This involves locating the cells that correspond to the relevant headers, ensuring the selected data is directly related to the question at hand.
- Perform Necessary Analysis or Calculations: Once the relevant data is identified, perform any required operations, such as statistical analysis, mathematical calculations, ranking, or other necessary procedures. This will help you derive the needed insights and provide a comprehensive answer.

(2) Your output action pattern and output format.
Your output should follow a React-like pattern of thinking, which includes one or more cycles of "Thought/Action/Result", ultimately leading to a "Final Answer" on the last line.
[Action Patterns]
- Thought: Consider the next action based on the result of the previous one.
- Action: The action should always be a single processing action.
- Result: Simulate the action result, analyze the result, and decide whether to continue or stop.
(This "Thought/Action/Result" cycle can repeat multiple times.)

Verify the table, observations, and question thoroughly before providing the final answer.
When answering, if the final answer comes from the original format in the table, please use the original format from the table without modifying it.
Below is an example of an output format. You need to first output the relevant keywords, headers, and content related with the question, then go through a multi-round interaction of thought/action/result, and finally provide the final answer.
[Output Example]
Relevant Keywords: Keywords related to the table in question
Relavant Table Headers: column/row headers related with the question
Relavant Content: related table content
Thought: Your first round of thinking.
Action: The action of your first round.
Result: The observation and result of your first round of simulation.
Thought: Your second round of thinking. The 'Thought/Action/Result' cycle can repeat 1 or more times until the final answer is reached.
Action: The action of your second round.
Result: The observation and result of your second round of simulation.
Final Answer: Your output result, following the format "Final Answer: AnswerName1, AnswerName2...". The "AnswerName" should be a number or entity name, as short as possible.

Let's get start!
[Question]
{Question}
"""

Structure_Comprehending = """
You are a table analyst. Your task is to first extract relevant keywords based on the questions posed, identify related content from the previous tables, and match it with the corresponding headers in the structure tree. Then you need to answer questions based on the provided table content. 

(1) Your guidelines of thinking and output format control:
- Understand the Question: Begin by carefully reading the question to extract the essential information needed for answering. This helps ensure that you focus on the right aspects of the table in the next steps.
- Analyze the Table Content: Thoroughly examine the structure tree and original content of the table, paying close attention to both row and column headers, which may include special indicators such as "Total," "Sum," "Average," or other summary metrics. Be mindful of any rows or columns dedicated to aggregates, as these can provide quick answers without the need for detailed calculations. It's also crucial to recognize that the table might have complex structures, such as merged cells or semantic nesting, which could influence the interpretation of the data.
- Identify Relevant Data: With a clear understanding of the question, identify the rows and columns in the table that are most relevant to the inquiry. This involves locating the cells that correspond to the relevant headers, ensuring the selected data is directly related to the question at hand.
- Perform Necessary Analysis or Calculations: Once the relevant data is identified, perform any required operations, such as statistical analysis, mathematical calculations, ranking, or other necessary procedures. This will help you derive the needed insights and provide a comprehensive answer.

(2) Your output action pattern and output format.
Your output should follow a React-like pattern of thinking, which includes one or more cycles of "Thought/Action/Result", ultimately leading to a "Final Answer" on the last line.
[Action Patterns]
- Thought: Consider the next action based on the result of the previous one.
- Action: The action should always be a single processing action.
- Result: Simulate the action result, analyze the result, and decide whether to continue or stop.
(This "Thought/Action/Result" cycle can repeat multiple times.)

Verify the table, observations, and question thoroughly before providing the final answer.
When answering, if the final answer comes from the original format in the table, please use the original format from the table without modifying it.
Below is an example of an output format. You need to first output the relevant keywords, headers, and content related with the question, then go through a multi-round interaction of thought/action/result, and finally provide the final answer.
[Output Example]
Relevant Keywords: Keywords related to the table in question
Relavant Table Headers: column/row headers related with the question
Relavant Content: related table content
Thought: Your first round of thinking.
Action: The action of your first round.
Result: The observation and result of your first round of simulation.
Thought: Your second round of thinking. The 'Thought/Action/Result' cycle can repeat 1 or more times until the final answer is reached.
Action: The action of your second round.
Result: The observation and result of your second round of simulation.
Final Answer: Your output result, following the format "Final Answer: AnswerName1, AnswerName2...". The "AnswerName" should be a number or entity name, as short as possible.

Let's get start!
[Question]
{Question}
"""





"""
You are assigned the task of answering questions based on the provided table. Follow these steps to ensure accuracy and efficiency:
(1). Your Chain of Thought:
1.1 Understand the Question: Carefully read the question to extract the relevant details needed for the answer.
1.2 Analyze the Table: Gain a clear understanding of the table's structure and content, including:
  -Row and column headers, particularly any summative or aggregate values such as "Total", "Sum", "Average", etc.
  -Flags or special notations that could simplify the analysis.
1.3 Select Relevant Data: Identify the rows and columns that are directly relevant to the question. Locate the corresponding cells based on the selected headers.
1.4 Perform Calculations or Operations: Based on the question, carry out the necessary calculations, ranking, or statistical operations on the identified data cells.
1.5 Provide the Final Answer: Present the answer strictly in the required format, as specified below.
(2). Output Control:
2.1 The response must be a one-line answer in the format:
    "[Final Answer]: AnswerName1, AnswerName2..."
2.2 Ensure that AnswerName is the shortest, most appropriate term (a number or entity name), with no explanation or elaboration.
2.3 If the answer involves decimals, round them to two decimal places.

You will be provided with a table in LaTeX format and a question. Use the table to generate your answer in the exact format requested.
"""


Rudimentary_Analysis = """
You are a table analyst. Your task is to first extract relevant keywords based on the questions posed, identify related content from the previous tables, and match it with the corresponding headers in the structure tree. Then you need to answer questions based on the provided table content. 

(1) Your guidelines of thinking and output format control:
- Understand the Question: Begin by carefully reading the question to extract the essential information needed for answering. This helps ensure that you focus on the right aspects of the table in the next steps.
- Analyze the Table Content: Thoroughly examine the structure tree and original content of the table, paying close attention to both row and column headers, which may include special indicators such as "Total," "Sum," "Average," or other summary metrics. Be mindful of any rows or columns dedicated to aggregates, as these can provide quick answers without the need for detailed calculations. It's also crucial to recognize that the table might have complex structures, such as merged cells or semantic nesting, which could influence the interpretation of the data.
- Identify Relevant Data: With a clear understanding of the question, identify the rows and columns in the table that are most relevant to the inquiry. This involves locating the cells that correspond to the relevant headers, ensuring the selected data is directly related to the question at hand.
- Perform Necessary Analysis or Calculations: Once the relevant data is identified, perform any required operations, such as statistical analysis, mathematical calculations, ranking, or other necessary procedures. This will help you derive the needed insights and provide a comprehensive answer.

(2) Your output action pattern and output format.
Your output should follow a React-like pattern of thinking, which includes one or more cycles of "Thought/Action/Result", ultimately leading to a "Final Answer" on the last line.
[Action Patterns]
- Thought: Consider the next action based on the result of the previous one.
- Action: The action should always be a single processing action.
- Result: Simulate the action result, analyze the result, and decide whether to continue or stop.
(This "Thought/Action/Result" cycle can repeat multiple times.)

Verify the table, observations, and question thoroughly before providing the final answer.
When answering, if the final answer comes from the original format in the table, please use the original format from the table without modifying it.
Below is an example of an output format. You need to first output the relevant keywords, headers, and content related with the question, then go through a multi-round interaction of thought/action/result, and finally provide the final answer.
[Output Example]
Relevant Keywords: Keywords related to the table in question
Relavant Table Headers: column/row headers related with the question
Relavant Content: related table content
Thought: Your first round of thinking.
Action: The action of your first round.
Result: The observation and result of your first round of simulation.
Thought: Your second round of thinking. The 'Thought/Action/Result' cycle can repeat 1 or more times until the final answer is reached.
Action: The action of your second round.
Result: The observation and result of your second round of simulation.
Final Answer: Your output result, following the format "Final Answer: AnswerName1, AnswerName2...". The "AnswerName" should be the primary result of the rudimentary analysis, such as a number or an entity name, expressed as concisely as possible.

Let's get start!
[Question]
{Question}
"""

Summary_Analysis = """
You are a table analyst. Your task is to first extract relevant keywords based on the questions posed, identify related content from the previous tables, and match it with the corresponding headers in the structure tree. Then you need to answer questions based on the provided table content. 

(1) Your guidelines of thinking and output format control:
- Understand the Question: Begin by carefully reading the question to extract the essential information needed for answering. This helps ensure that you focus on the right aspects of the table in the next steps.
- Analyze the Table Content: Thoroughly examine the structure tree and original content of the table, paying close attention to both row and column headers, which may include special indicators such as "Total," "Sum," "Average," or other summary metrics. Be mindful of any rows or columns dedicated to aggregates, as these can provide quick answers without the need for detailed calculations. It's also crucial to recognize that the table might have complex structures, such as merged cells or semantic nesting, which could influence the interpretation of the data.
- Identify Relevant Data: With a clear understanding of the question, identify the rows and columns in the table that are most relevant to the inquiry. This involves locating the cells that correspond to the relevant headers, ensuring the selected data is directly related to the question at hand.
- Perform Necessary Analysis or Calculations: Once the relevant data is identified, perform any required operations, such as statistical analysis, mathematical calculations, ranking, or other necessary procedures. This will help you derive the needed insights and provide a comprehensive answer.

(2) Your output action pattern and output format.
Your output should follow a React-like pattern of thinking, which includes one or more cycles of "Thought/Action/Result", ultimately leading to a "Final Answer" on the last line.
[Action Patterns]
- Thought: Consider the next action based on the result of the previous one.
- Action: The action should always be a single processing action.
- Result: Simulate the action result, analyze the result, and decide whether to continue or stop.
(This "Thought/Action/Result" cycle can repeat multiple times.)

Verify the table, observations, and question thoroughly before providing the final answer.
When answering, if the final answer comes from the original format in the table, please use the original format from the table without modifying it.
Below is an example of an output format. You need to first output the relevant keywords, headers, and content related with the question, then go through a multi-round interaction of thought/action/result, and finally provide the final answer.
[Output Example]
Relevant Keywords: Keywords related to the table in question
Relavant Table Headers: column/row headers related with the question
Relavant Content: related table content
Thought: Your first round of thinking.
Action: The action of your first round.
Result: The observation and result of your first round of simulation.
Thought: Your second round of thinking. The 'Thought/Action/Result' cycle can repeat 1 or more times until the final answer is reached.
Action: The action of your second round.
Result: The observation and result of your second round of simulation.
Final Answer: Your output result, following the format "Final Answer: TableSummary". The "TableSummary" should provide a concise summary of the table, including a brief description of its content, the main columns, and any basic insights derived.

Let's get start!
[Question]
{Question}
"""

Predictive_Analysis = """
You are a table analyst. Your task is to first extract relevant keywords based on the questions posed, identify related content from the previous tables, and match it with the corresponding headers in the structure tree. Then you need to answer questions based on the provided table content. 

(1) Your guidelines of thinking and output format control:
- Understand the Question: Begin by carefully reading the question to extract the essential information needed for answering. This helps ensure that you focus on the right aspects of the table in the next steps.
- Analyze the Table Content: Thoroughly examine the structure tree and original content of the table, paying close attention to both row and column headers, which may include special indicators such as "Total," "Sum," "Average," or other summary metrics. Be mindful of any rows or columns dedicated to aggregates, as these can provide quick answers without the need for detailed calculations. It's also crucial to recognize that the table might have complex structures, such as merged cells or semantic nesting, which could influence the interpretation of the data.
- Identify Relevant Data: With a clear understanding of the question, identify the rows and columns in the table that are most relevant to the inquiry. This involves locating the cells that correspond to the relevant headers, ensuring the selected data is directly related to the question at hand.
- Perform Necessary Analysis or Calculations: Once the relevant data is identified, perform any required operations, such as statistical analysis, mathematical calculations, ranking, or other necessary procedures. This will help you derive the needed insights and provide a comprehensive answer.

(2) Your output action pattern and output format.
Your output should follow a React-like pattern of thinking, which includes one or more cycles of "Thought/Action/Result", ultimately leading to a "Final Answer" on the last line.
[Action Patterns]
- Thought: Consider the next action based on the result of the previous one.
- Action: The action should always be a single processing action.
- Result: Simulate the action result, analyze the result, and decide whether to continue or stop.
(This "Thought/Action/Result" cycle can repeat multiple times.)

Verify the table, observations, and question thoroughly before providing the final answer.
When answering, if the final answer comes from the original format in the table, please use the original format from the table without modifying it.
Below is an example of an output format. You need to first output the relevant keywords, headers, and content related with the question, then go through a multi-round interaction of thought/action/result, and finally provide the final answer.
[Output Example]
Relevant Keywords: Keywords related to the table in question
Relavant Table Headers: column/row headers related with the question
Relavant Content: related table content
Thought: Your first round of thinking.
Action: The action of your first round.
Result: The observation and result of your first round of simulation.
Thought: Your second round of thinking. The 'Thought/Action/Result' cycle can repeat 1 or more times until the final answer is reached.
Action: The action of your second round.
Result: The observation and result of your second round of simulation.
Final Answer: Your output result, following the format "Final Answer: AnswerName1, AnswerName2...". The "AnswerName" should summarize the primary result of the analysis in a concise manner, such as a number, an entity name, or a trend description (e.g., No clear trend, Increasing trend, Decreasing trend, Stabilize).

Let's get start!
[Question]
{Question}
"""

Exploratory_Analysis = """
You are a table analyst. Your task is to first extract relevant keywords based on the questions posed, identify related content from the previous tables, and match it with the corresponding headers in the structure tree. Then you need to answer questions based on the provided table content. 

(1) Your guidelines of thinking and output format control:
- Understand the Question: Begin by carefully reading the question to extract the essential information needed for answering. This helps ensure that you focus on the right aspects of the table in the next steps.
- Analyze the Table Content: Thoroughly examine the structure tree and original content of the table, paying close attention to both row and column headers, which may include special indicators such as "Total," "Sum," "Average," or other summary metrics. Be mindful of any rows or columns dedicated to aggregates, as these can provide quick answers without the need for detailed calculations. It's also crucial to recognize that the table might have complex structures, such as merged cells or semantic nesting, which could influence the interpretation of the data.
- Identify Relevant Data: With a clear understanding of the question, identify the rows and columns in the table that are most relevant to the inquiry. This involves locating the cells that correspond to the relevant headers, ensuring the selected data is directly related to the question at hand.
- Perform Necessary Analysis or Calculations: Once the relevant data is identified, perform any required operations, such as statistical analysis, mathematical calculations, ranking, or other necessary procedures. This will help you derive the needed insights and provide a comprehensive answer.

(2) Your output action pattern and output format.
Your output should follow a React-like pattern of thinking, which includes one or more cycles of "Thought/Action/Result", ultimately leading to a "Final Answer" on the last line.
[Action Patterns]
- Thought: Consider the next action based on the result of the previous one.
- Action: The action should always be a single processing action.
- Result: Simulate the action result, analyze the result, and decide whether to continue or stop.
(This "Thought/Action/Result" cycle can repeat multiple times.)

Verify the table, observations, and question thoroughly before providing the final answer.
When answering, if the final answer comes from the original format in the table, please use the original format from the table without modifying it.
Below is an example of an output format. You need to first output the relevant keywords, headers, and content related with the question, then go through a multi-round interaction of thought/action/result, and finally provide the final answer.
[Output Example]
Relevant Keywords: Keywords related to the table in question
Relavant Table Headers: column/row headers related with the question
Relavant Content: related table content
Thought: Your first round of thinking.
Action: The action of your first round.
Result: The observation and result of your first round of simulation.
Thought: Your second round of thinking. The 'Thought/Action/Result' cycle can repeat 1 or more times until the final answer is reached.
Action: The action of your second round.
Result: The observation and result of your second round of simulation.
Final Answer: Your output result, following the format "Final Answer: CorrelationRelation, CorrelationCoefficient". CorrelationCoefficient should be a float number; CorrelationRelation can only be "No correlation" with the correlation coefficient between -0.3 to +0.3, "Weak positive correlation" with the correlation coefficient between +0.3 to +0.7, "Weak negative correlation" with the correlation coefficient between -0.3 to -0.7, "Strong positive correlation" with the correlation coefficient between +0.7 to +1, or "Strong negative correlation" with the correlation coefficient between -0.7 to -1.
If the question is about impact analysis, the "Final Answer" should be a entity name or a impact description(No clear impact, Negtive impact or Positive impact), as short as possible, without any explanation. If the question is about causal analysis, the "Final Answer" should be a brief explanation of the causal analysis results as concise as possible. 

Let's get start!
[Question]
{Question}
"""

Anomaly_Analysis = """
You are a table analyst. Your task is to first extract relevant keywords based on the questions posed, identify related content from the previous tables, and match it with the corresponding headers in the structure tree. Then you need to answer questions based on the provided table content. 

(1) Your guidelines of thinking and output format control:
- Understand the Question: Begin by carefully reading the question to extract the essential information needed for answering. This helps ensure that you focus on the right aspects of the table in the next steps.
- Analyze the Table Content: Thoroughly examine the structure tree and original content of the table, paying close attention to both row and column headers, which may include special indicators such as "Total," "Sum," "Average," or other summary metrics. Be mindful of any rows or columns dedicated to aggregates, as these can provide quick answers without the need for detailed calculations. It's also crucial to recognize that the table might have complex structures, such as merged cells or semantic nesting, which could influence the interpretation of the data.
- Identify Relevant Data: With a clear understanding of the question, identify the rows and columns in the table that are most relevant to the inquiry. This involves locating the cells that correspond to the relevant headers, ensuring the selected data is directly related to the question at hand.
- Perform Necessary Analysis or Calculations: Once the relevant data is identified, perform any required operations, such as statistical analysis, mathematical calculations, ranking, or other necessary procedures. This will help you derive the needed insights and provide a comprehensive answer.

(2) Your output action pattern and output format.
Your output should follow a React-like pattern of thinking, which includes one or more cycles of "Thought/Action/Result", ultimately leading to a "Final Answer" on the last line.
[Action Patterns]
- Thought: Consider the next action based on the result of the previous one.
- Action: The action should always be a single processing action.
- Result: Simulate the action result, analyze the result, and decide whether to continue or stop.
(This "Thought/Action/Result" cycle can repeat multiple times.)

Verify the table, observations, and question thoroughly before providing the final answer.
When answering, if the final answer comes from the original format in the table, please use the original format from the table without modifying it.
Below is an example of an output format. You need to first output the relevant keywords, headers, and content related with the question, then go through a multi-round interaction of thought/action/result, and finally provide the final answer.
[Output Example]
Relevant Keywords: Keywords related to the table in question
Relavant Table Headers: column/row headers related with the question
Relavant Content: related table content
Thought: Your first round of thinking.
Action: The action of your first round.
Result: The observation and result of your first round of simulation.
Thought: Your second round of thinking. The 'Thought/Action/Result' cycle can repeat 1 or more times until the final answer is reached.
Action: The action of your second round.
Result: The observation and result of your second round of simulation.
Final Answer: Your output result, following the format "Final Answer: TableAnomaly".
If there are no anomalies, you should output "No anomalies are detected in the table." Otherwise, you should output "X anomaly detected: Anomaly1, Anomaly2 ... AnomalyX" and explain your reasons later

Let's get start!
[Question]
{Question}
"""

Visulization = """
You are a table analyst. Your task is to first extract relevant keywords based on the questions posed, identify related content from the previous tables, and match it with the corresponding headers in the structure tree. Then you need to answer questions based on the provided table content. 

(1) Your guidelines of thinking and output format control:
Your complete thought process is as follows:
- Understand the Question: Begin by carefully reading the question to extract the essential information needed for answering. This helps ensure that you focus on the right aspects of the table in the next steps.
- Analyze the Table Content: Thoroughly examine the structure and content of the table, paying close attention to both row and column headers, which may include special indicators such as "Total," "Sum," "Average," or other summary metrics. Be mindful of any rows or columns dedicated to aggregates, as these can provide quick answers without the need for detailed calculations. It's also crucial to recognize that the table might have complex structures, such as merged cells or semantic nesting, which could influence the interpretation of the data.
- Identify Relevant Data: With a clear understanding of the question, identify the rows and columns in the table that are most relevant to the inquiry. This involves locating the cells that correspond to the relevant headers, ensuring the selected data is directly related to the question at hand.
- Perform Necessary Analysis or Calculations: Once the relevant data is identified, perform any required operations, such as statistical analysis, mathematical calculations, ranking, or other necessary procedures. This will help you derive the needed insights and provide a comprehensive answer.

(2) Your output action pattern and output format.
Your output should follow a React-like pattern of thinking, which includes one or more cycles of "Thought/Action/Result", ultimately leading to a "Final Answer" on the last line.
[Action Patterns]
- Thought: Consider the next action based on the result of the previous one.
- Action: The action should always be a single processing action.
- Result: Simulate the action result, analyze the result, and decide whether to continue or stop.
(This "Thought/Action/Result" cycle can repeat multiple times.)

Verify the table, observations, and question thoroughly before providing the final answer.
Below is an example of an output format.
[Output Example]
Relevant Keywords: Keywords related to the table in question
Relavant Table Headers: column/row headers related with the question
Relavant Content: related table content
Thought: Your first round of thinking.
Action: The action of your first round.
Result: The observation and result of your first round of simulation.
Thought: Your second round of thinking. The 'Thought/Action/Result' cycle can repeat 1 or more times until the final answer is reached.
Action: The action of your second round.
Result: The observation and result of your second round of simulation.
Final Answer: Your output result, following the format "Final Answer: Your python code..."
Your code should follow the format below and ensure the code format is exactly the same with the following code block: import pandas as pd import matplotlib.pyplot as plt ... plt.show(). Ensure the code can generate the chart correctly.

Let's get start!  
[Question]
{Question}

"""



User_Prompt = """ 
(1) We first provide the structural information of the top column headers and left row headers in the table as follows:
The top column header and left row header of the table can each be represented as a separate tree structure. Each node in the tree corresponds to a specific table header or categorization level, which ultimately maps to a particular data cell. Intermediate nodes are used to further categorize parent nodes, while leaf nodes represent specific classifications that directly link to data cells. The nodes are represented as tuples.

1.1 A tuple (T, T1, T2, T3, T4, T5) represents a column header, where:
T: Indicates that it is a column header.
T1: Denotes the level of the header.
T2 and T3: Indicate the start and end column of the header.
T4: Specifies the row position of the header.
T5: Contains the content of the header.

1.2 A tuple (L, L1, L2, L3, L4, L5) represents a row header, where:
L: Indicates that it is a row header.
L1: Denotes the level of the header.
L2 and L3: Indicate the start and end row of the header.
L4: Specifies the column position of the header.
L5: Contains the content of the header.
The tuple of a column header includes T1, which represents the level of the header. Level 0 represents the highest level, with higher values of T1 corresponding to lower levels. If the range (T2, T3) of tuple A is within the range (T2, T3) of tuple B (with possible equality), a parent-child relationship exists between A and B, where A is a sub-header of B and B is the parent header of A. Additionally, A's T1 must be smaller than B's T1. The lowest level headers have T2 equal to T3. This principle similarly applies to row headers.

Examples:
The tuple (T, 1, 0, 0, 0, g) represents a column header at level 1, spanning from column 0 to column 0, located on row 0, with the content "g".
The tuple (L, 0, 6, 6, 1, karlsruher sc) represents a row header at level 0, located on column 1, spanning rows 6 to 6, with the content "karlsruher sc".

I will now provide the column headers, row headers, and the raw LaTeX data of the table:
Column Header
{ColumnHeader}

Row Header
{RowHeader}

(2) The following is the LaTeX representation of the raw table data:


{Table}




(3) Please answer the following question based on the provided table structure. When integrating the question with the table contents, please ensure to validate your answer and output it in the format:
[Final Answer]: {Answer_format}

Question:
{Question}
"""


Model_Second_Response = """
Got it! Please provide the table in LaTeX format along with the question.
"""

Model_First_Response = """
Got it! Please provide the column header and row header about the table and the question. I will choose the most relevant column header and row header about the question.
"""

Tree_Prompt = """
The top column header and the left row header in the table can be constructed as two separate trees, each node of the tree represents a table header or categorization level that ultimately locates to a specific data cell. Intermediate nodes are used to further categorize the parent nodes, and leaves represent a specific certain classification that is directly associated with a data cell. The nodes of the tree are represented by tuple.
1. A tuple (T, T1, T2, T3, T4, T5) represents a column header, where T indicates it's a column header, T1 denotes its level, T2 and T3 indicate the start and end column of the header, T4 indicate the row position of the header, and T5 specifies the content. 
2. A tuple (L, L1, L2, L3, L4, L5) represents a row header, where L indicates it's a row header, L1 denotes its level, L2 and L3 indicate the start and end row of the header, L4 indicate the column position of the header, and L5 specifies the content. 
3. The tuple of a column header contains T1, representing the level of the header, with 0 being the highest level and larger T1 indicating lower levels. If the T2 and T3 of tuple A are between T2 and T3 of tuple B (can be equal), then there is a parent-child relationship between A and B, A is a sub-header of B, B is a parent-header of A, and A's T1 must be smaller than B's T1. The lowest level header's tuple has T2=T3. Similarly for row headers. The specific tuples are in Table Content. 
4. For examples: The tuple (T, 1, 0, 0, 0, g) denotes a column header with level 1, spanning from column 0 to column 0, located on row 0, with the content "g". The tuple (L, 0, 6, 6, 1, karlsruher sc) denotes a row header with level 0, located on column 1, spanning from row 6 to row 6, with the content "karlsruher sc". 
I will give you the column headers and row headers of the table, and the question you need to slove.
#Column Header
{ColumnHeader}
#Row Header
{RowHeader}
#Question
{Question}
Fully understand the question and extract the necessary information from it and select 5-10 row and column headers that are most relevant to the question.
"""

Answer_Prompt = {
    "Fact Checking": Fact_Checking,
    "Numerical Reasoning": Numerical_Reasoning,
    "Rudimentary Analysis": Rudimentary_Analysis,
    "Summary Analysis": Summary_Analysis,
    "Predictive Analysis": Predictive_Analysis,
    "Exploratory Analysis": Exploratory_Analysis,
    "Anomaly Analysis": Anomaly_Analysis,
    "Visualization": Visulization,
    "Generate_Tree": Generate_Tree,
    "Structure Comprehending": Structure_Comprehending
}