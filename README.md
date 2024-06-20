
# Do Large Language Models Pay Similar Attention Like Human Programmers When Generating Code?
Large Language Models (LLMs) have recently been widely used for code generation. Due to the complexity and opacity of LLMs, little is known about how these models generate code. We made the first attempt to bridge this knowledge gap by investigating whether LLMs attend to the same parts of a task description as human programmers during code generation. An analysis of six LLMs, including GPT-4, on two popular code generation benchmarks revealed a consistent misalignment between LLMs' and programmers' attention. We manually analyzed 211 incorrect code snippets and found five attention patterns that can be used to explain many code generation errors. Finally, a user study showed that model attention computed by a perturbation-based method is often favored by human programmers. Our findings highlight the need for human-aligned LLMs for better interpretability and programmer trust. Data and codes to replicate our work are included in this repository.

[^fn1]: https://arxiv.org/abs/2306.01220
  
## Data
The labeled HumanEval and MBPP datasets are available in dataset folder. For details of the `dataset`, please check our paper.

## Experiment Setup
Our study includes 12 attention calculation methods from four categories: six self-attention-based, four gradient-based, and two perturbation-based methods. This repository contains attention calculated by all 12 methods on six models as pickled files in the `experiment_result` folder. Readers can also run the experiment from scratch with notebooks in `perturbation_based`, `gradient_based`, `self_attention_based` folders.

To run the gradient-based methods, you must install the `ecco` library, which requires a Python version of 3.8.19.

## Experiment Results
Our study reveals several important findings about LLMs for code. First, there is a consistent misalignment between model and programmer attention. Secondly, after averaging all five models (except for GPT-4), BERT_masking produces the best attention alignment to human programmers among all methods. Gradient-based methods are generally better than self-attention-based methods. For gradient-based and self-attention-based methods, attention distributions in the coding stage produce higher alignment. For self-attention-based methods, attention distributions in the last layer produce the highest alignment. 

  <img src="https://github.com/BonanKou/Attention-Alignment-Empirical-Study/blob/main/pics/Image_20240523091222.png" alt="drawing" width="400"/>

## Manual Analysis
To understand whether attention analysis can explain why LLMs incorrectly generate code, we manually analyzed 211 incorrect generations by CodeGen and GPT-4. We found that 27% of errors can be explained by attention analysis. Specifically, there are five types of explainable errors.

The 211 incorrect generations by CodeGen and GPT-4 can be accessed at:

The explainable errors can be accessed at:

`other_data/explainable_errors.txt`



