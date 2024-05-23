
# Do Large Language Models Pay Similar Attention Like Human Programmers When Generating Code?
Large Language Models (LLMs) have recently been widely used for code generation. Due to the complexity and opacity of LLMs, little is known about how these models generate code. We made the first attempt to bridge this knowledge gap by investigating whether LLMs attend to the same parts of a task description as human programmers during code generation. An analysis of six LLMs, including GPT-4, on two popular code generation benchmarks revealed a consistent misalignment between LLMs' and programmers' attention. We manually analyzed 211 incorrect code snippets and found five attention patterns that can be used to explain many code generation errors. Finally, a user study showed that model attention computed by a perturbation-based method is often favored by human programmers. Our findings highlight the need for human-aligned LLMs for better interpretability and programmer trust. Data and codes to replicate our work are included in this repository.

[^fn1]: https://arxiv.org/abs/2306.01220

  <img src="[https://github.com/BonanKou/Empirical_Study_On_Attention_Alignment_CodeLLMs/blob/main/image/example_misalignment-1.png](https://github.com/BonanKou/Attention-Alignment-Empirical-Study/blob/main/pics/Image_20240523091222.png)" alt="drawing" width="400"/>
  
## Data
The labeled HumanEval and MBPP datasets are available in dataset folder. For details of the `dataset`, please check our paper.

## Experiment Setup
Our study includes 12 attention calculation methods from four categories: six self-attention-based, four gradient-based, and two perturbation-based methods. This repository contains attention calculated by all 12 methods on six models as pickled files in the `experiment_result` folder. Readers can also run the experiment from scratch with notebooks in `perturbation_based`, `gradient_based`, `self_attention_based` folders.

#### Prerequisites
To run the gradient-based methods, you will need to install the `ecco` library, which requires a Python version of 3.8.19.

Small modifications are needed in order to run SHAP on CodeLLMs. To install our modified ver. of SHAP.

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



