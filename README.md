
# Do Large Language Models Pay Similar Attention Like Human Programmers When Generating Code?
Large Language Models (LLMs) have recently been widely used for code generation. Due to the complexity and opacity of LLMs, little is known about how these models generate code. We made the first attempt to bridge this knowledge gap by investigating whether LLMs attend to the same parts of a task description as human programmers during code generation. An analysis of six LLMs, including GPT-4, on two popular code generation benchmarks revealed a consistent misalignment between LLMs' and programmers' attention. We manually analyzed 211 incorrect code snippets and found five attention patterns that can be used to explain many code generation errors. Finally, a user study showed that model attention computed by a perturbation-based method is often favored by human programmers. Our findings highlight the need for human-aligned LLMs for better interpretability and programmer trust. Data and codes to replicate our work are included in this repository.

[^fn1]: Mark Chen et al. 2021. Evaluating Large Language Models Trained on Code.
(2021).
[^fn2]: Jacob Austin, Augustus Odena, et al. 2021. Program Synthesis with Large Language Models.

  <img src="https://github.com/BonanKou/Empirical_Study_On_Attention_Alignment_CodeLLMs/blob/main/image/example_misalignment-1.png" alt="drawing" width="400"/>
  
## Data
The most critical artifact in this repository is our human-annotated dataset with programmer attention. Since this study's rigor relies mainly on the quality of the human annotation, we will briefly explain how the dataset is constructed in this section. For more details, please refer to our paper.

### Definition of Human Attention
When solving a programming task by reading a prompt in natural language, programmers will not pay equal attention to every word in the prompt. In fact, some words (i.e., keywords) are more informative than others and require more attention (e.g., "return"). The goal of the labelers who constructed this dataset is to select such keywords. 

### Data Format
A pickled version of our dataset is provided in the `dataset` folder. To unpickle the file, import the `pickle` library and run the following:

`pickle.load("dataset.txt", "rb")`

Our dataset contains code generation prompts for 1,111 Python tasks from HumanEval (164) and MBPP (947) datasets. The dataset includes human-labeled keywords highlighted in different colors, which indicates four categories of keywords:
-   Green: ***datatype*** keywords that describe the types of data that the code should input or output, such as “string”, “number”, or “list”.
    
-   Cyan: ***conditional*** keywords that describe the operations that the code should perform on the data, such as “compare”, “sort”, “filter”, or “search”.
    
-   Yellow: ***property*** keywords that signal the conditions under which the code should execute, such as “if”, “else”, “when”, and “for”

-   Red: ***operator*** keywords that suggest important properties of the manipulated data and operations, including quantifiers (e.g., “all”, “one”), adjectives (e.g., “first”, “closer”), and adverbs (e.g., “every”, “none”).

The `highlight` field in each item of the dictionary contains a list of dictionaries, where each dictionary represents a highlighted keyword. Each dictionary contains two fields: place, which represents the character indices of the keyword in the prompt, and color, which represents the color code for the keyword.

### Labeling Procedure
The first two authors, with over five years of programming experience in Python, manually labeled the words and phrases they considered essential to solving a programming task in the task description.

Before the labeling process, the two labelers went through 164 programming tasks in HumanEval to familiarize themselves with the programming tasks and the code solutions. During the labeling process, they first independently labeled 20 task descriptions. This first round of labeling had a Cohen's Kappa score of 0.68. The two labelers discussed the disagreements and continued to label the rest of the HumanEval dataset. The labeling process ended with a Cohen's Kappa score of 0.73.

### Data Validation
As programming is a cognitively demanding task, different programmers may approach the same problem differently. Therefore, it is hard to claim annotations in our dataset can represent how a large pool of programmers select keywords from programming prompts. To verify our labels are generally accepted, we invited a third labeler to review our labeling process. The third labeler independently labeled 164 prompts from the HumanEval dataset without looking at the labels of the first two authors. The Fleiss’ Kappa score between the labels of the three labelers is 0.64, indicating a substantial agreement. This result shows labels made by the first two labelers are reasonable and can be accepted by other programmers. 


## Experiment Setup
Our study includes 12 attention calculation methods from four categories: six self-attention-based methods, four gradient-based methods, and two perturbation-based methods. While this repository contains attention calculated by all 12 methods on six models (only perturbation-based methods can be applied to GPT-4), readers can also run the experiment from scratch.

### For Perturbation-based Methods

~~~sh
cd perturbation_based
~~~

#### Prerequisites

Small modifications are needed in order to run SHAP on CodeLLMs. To install our modified ver. of SHAP.

~~~sh
cd shap_lib
pip install -e .
~~~

#### To run the experiments with *Perturbation masking* and *SHAP*

~~~sh
./bert_exp.sh
./shap_exp.sh
~~~

#### To convert *SHAP* results into json files

~~~sh
python3 shap_result_conversion.py
~~~

### For Attention-based Methods
To run the attention-based methods, we can just run the .ipynb file in that folder. Remember that you can choose which layer you want by editing the code.

### For Gradient-based Methods
To run the attribution-based methods, we can just run the .ipynb file in that folder. For convenience, we use the ecco library to compute the gradient. Inside this library, it actually used the Captum library, a library for XAI, to compute the gradient.

## Experiment Results
Our study reveals several important findings about LLMs for code. First, there is a consistent misalignment between model and programmer attention. 

#### To reproduce content of this table

Secondly, after averaging all five models (except for GPT-4), BERT_masking produces the best attention alignment to human programmers among all methods. Gradient-based methods are generally better than self-attention-based methods. For gradient-based and self-attention-based methods, attention distributions in the coding stage produce higher alignment. For self-attention-based methods, attention distributions in the last layer produce the highest alignment. 

#### To reproduce content of this table

## Manual Analysis
To understand whether attention analysis can explain why LLMs incorrectly generate code, we manually analyzed 211 incorrect generations by CodeGen and GPT-4. We found that 27% of errors can be explained by attention analysis. Specifically, there are five types of explainable errors.

The 211 incorrect generations by CodeGen and GPT-4 can be accessed at:

The explainable errors can be accessed at:

## User Study
To understand which explanation method programmers prefer, we conducted a user study with 22 students (18 males and 4 females) from the CS department of an R1 university. These participants have an average of 5.62 years of programming experience and have some basic understanding of model attention in machine learning. We randomly selected 8 task descriptions from our dataset. For each task, we leveraged CodeGen-2.7B, the best-performing open-source model, to calculate its model attention. We selected one attention calculation method from each category (self-attention-based, perturbation-based, and gradient-based): CODING_last, INPUTxGradient_coding, and SHAP.

In each user study, participants first read the task description to understand the programming task and then read the code generated by CodeGen. For each attention method, we render a highlighted version of the task description, where the top 10 attended tokens are highlighted based on the attention scores computed by this method. We chose to render the top 10 important keywords since it is close to the average number of important words (7) labeled in our dataset. Participants were then asked to rate each attention method by indicating their agreement with the following three statements on a 7-point Likert scale (1—completely disagree, 7—completely agree).

Overall, participants preferred the perturbation-based method over the gradient-based and self-attention-based methods. However, after seeing the attention-based explanations, participants still felt a lack of trust in LLMs and wished to see richer explanations, such as reference code and fine-grained attention mapping between text and code.



