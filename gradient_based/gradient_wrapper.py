import pickle
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import logging
import transformers
import ecco
import torch
from pandas.core.computation.check import NUMEXPR_INSTALLED
import torch
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# def save_checkpoints(model, saliency_reading, saliency_coding, inputxgradient_reading, inputxgradient_coding):
#     data_dict = {
#         "saliency_reading": saliency_reading,
#         "saliency_coding": saliency_coding,
#         "inputxgradient_reading": inputxgradient_reading,
#         "inputxgradient_coding": inputxgradient_coding
#     }
    
#     # 遍历字典，并将每个变量序列化到文件
#     for var_name, data in data_dict.items():
#         # 文件路径包含模型名称
#         pickle.dump(data, open(f"/home/bonan/fse2024/data/{var_name}_{model}.pkl", "wb"))
            
#     print("All checkpoints saved successfully.")

class gradient_based_method():
    def __init__(self, model_name, device, output_folder="."):
        self.output_folder = output_folder
        self.model_name = model_name
        self.token_counts = pickle.load(open("prompt_token_count.pkl", "rb"))

        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

        if model_name == "codegen":
            self.tokenizer = AutoTokenizer.from_pretrained('Salesforce/codegen-2B-mono')
            model_config = {
                'embedding': "transformer.wte.weight",
                'type': 'causal',
                'activations': ['gelu_new'], #This is a regex
                'token_prefix': 'Ġ',
                'partial_token_prefix': '',
            }
            self.lm = ecco.from_pretrained("Salesforce/codegen-2B-mono",model_config = model_config, verbose=False)
        elif model_name == "codeparrot":
            self.tokenizer = AutoTokenizer.from_pretrained("codeparrot/codeparrot")
            model_config = {
                'embedding': "transformer.wte.weight",
                'type': 'causal',
                'token_prefix': 'Ġ',
                'partial_token_prefix': '',
                'activations': ['gelu_new']
            }
            self.lm = ecco.from_pretrained("codeparrot/codeparrot",model_config = model_config, verbose=False)
        elif model_name == "gptj":
            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")
            model_config = {
                'embedding': "transformer.wte.weight",
                'type': 'causal',
                'activations': ['gelu_new'], #This is a regex
                'token_prefix': 'Ġ',
                'partial_token_prefix': '',
            }
            self.lm = ecco.from_pretrained("EleutherAI/gpt-j-6b",model_config = model_config, verbose=False)
        elif model_name == "incoder":
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/incoder-1B")
            model_config = {
                'embedding': "model.embed_tokens.weight",
                'type': 'causal',
                'activations': ['gelu'], #This is a regex
                'token_prefix': 'Ġ',
                'partial_token_prefix': '',
            }
            self.lm = ecco.from_pretrained("facebook/incoder-1B",model_config = model_config, verbose=False)
        elif model_name == "polycoder":
            def rename_attribute(obj, old_name, new_name):
                obj._modules[new_name] = obj._modules[old_name]
            self.tokenizer = AutoTokenizer.from_pretrained("NinedayWang/PolyCoder-2.7B")
            model_config = {
                'embedding': "gpt_neox.embed_in.weight",
                'type': 'causal',
                'activations': ['mlp\.c_proj'], #This is a regex
                'token_prefix': 'Ġ',
                'partial_token_prefix': '',
            }
            self.lm = ecco.from_pretrained("NinedayWang/PolyCoder-2.7B",model_config = model_config, verbose=False)
            # rename_attribute(self.lm.model, 'newname', 'lm_head')
            rename_attribute(self.lm.model, 'embed_out', 'lm_head')
    def perfrom_gradient_based(self, dataset):
        saliency_reading_filename = f"{self.output_folder}/saliency_reading_{self.model_name}.pkl"
        saliency_coding_filename = f"{self.output_folder}/saliency_coding_{self.model_name}.pkl"
        inputxgradient_reading_filename = f"{self.output_folder}/inputxgradient_reading_{self.model_name}.pkl"
        inputxgradient_coding_filename = f"{self.output_folder}/inputxgradient_coding_{self.model_name}.pkl"

        saliency_reading = dict()
        saliency_coding = dict()
        inputxgradient_reading = dict()
        inputxgradient_coding = dict()

        if os.path.exists(saliency_reading_filename):
            saliency_reading = pickle.load(open(saliency_reading_filename, "rb"))

        if os.path.exists(saliency_coding_filename):
            saliency_coding = pickle.load(open(saliency_coding_filename, "rb"))

        if os.path.exists(saliency_reading_filename):
            inputxgradient_reading = pickle.load(open(inputxgradient_reading_filename, "rb"))

        if os.path.exists(inputxgradient_coding_filename):
            inputxgradient_coding = pickle.load(open(inputxgradient_coding_filename, "rb"))

        for problem in tqdm(dataset, desc="Processing dataset"):
            if problem not in saliency_coding:
                # Determine ground truth
                delta = len(self.tokenizer(self.token_counts[problem])["input_ids"]) + 20

                # print("Prompt", problem)
                # print("需要生成", delta)

                # Determine whether it is humaneval or mbpp, if mbpp, add function header.
                if problem.find("def") != -1:
                    start_range_true_prompt = len(self.tokenizer(problem[:problem.find("def")])['input_ids'])
                    end_range_true_prompt = len(self.tokenizer(problem[:problem.find(">>>")])['input_ids'])

                    output = self.lm.generate(problem, generate=delta, attribution=['saliency', "grad_x_input"])
                    saliency_score = output.attribution['saliency']
                    inputxgradient_score = output.attribution['grad_x_input']
                    model_tokens = [i.replace("Ġ", " ").replace("Ċ", "\n").strip() for i in output.tokens[0][start_range_true_prompt:end_range_true_prompt] if i.find(">>>") == -1]
                    
                    each_saliency_reading = []
                    each_saliency_coding = []
                    each_inputxgradient_reading = []
                    each_inputxgradient_coding = []

                    for i, j in zip(model_tokens, saliency_score[0][start_range_true_prompt:start_range_true_prompt+len(model_tokens)]):
                        if i != "" and i.find("endoftext") == -1:
                            each_saliency_reading.append((i, j))
                    for i, j in zip(model_tokens, saliency_score[-1][start_range_true_prompt:start_range_true_prompt+len(model_tokens)]):
                        if i != "" and i.find("endoftext") == -1:
                            each_saliency_coding.append((i, j))
                    for i, j in zip(model_tokens, inputxgradient_score[0][start_range_true_prompt:start_range_true_prompt+len(model_tokens)]):
                        if i != "" and i.find("endoftext") == -1:
                            each_inputxgradient_reading.append((i, j))
                    for i, j in zip(model_tokens, inputxgradient_score[-1][start_range_true_prompt:start_range_true_prompt+len(model_tokens)]):
                        if i != "" and i.find("endoftext") == -1:
                            each_inputxgradient_coding.append((i, j))

                    saliency_reading[problem] = each_saliency_reading
                    saliency_coding[problem] = each_saliency_coding
                    inputxgradient_reading[problem] = each_inputxgradient_reading
                    inputxgradient_coding[problem] = each_inputxgradient_coding

                else:
                    # print("MBPP来的")
                    original_token_count = len(self.tokenizer(problem)['input_ids'])
                    output = self.lm.generate(problem + "\ndef function(", generate=delta, attribution=['saliency', "grad_x_input"])
                    saliency_score = output.attribution['saliency']
                    inputxgradient_score = output.attribution['grad_x_input']
                    model_tokens = [i.replace("Ġ", " ").replace("Ċ", "\n").strip() for i in output.tokens[0][:original_token_count]]

                    each_saliency_reading = []
                    each_saliency_coding = []
                    each_inputxgradient_reading = []
                    each_inputxgradient_coding = []

                    for i, j in zip(model_tokens, saliency_score[0][:original_token_count]):
                        if i != "" and i.find("endoftext") == -1:
                            each_saliency_reading.append((i, j))
                    for i, j in zip(model_tokens, saliency_score[-1][:original_token_count]):
                        if i != "" and i.find("endoftext") == -1:
                            each_saliency_coding.append((i, j))
                    for i, j in zip(model_tokens, inputxgradient_score[0][:original_token_count]):
                        if i != "" and i.find("endoftext") == -1:
                            each_inputxgradient_reading.append((i, j))
                    for i, j in zip(model_tokens, inputxgradient_score[-1][:original_token_count]):
                        if i != "" and i.find("endoftext") == -1:
                            each_inputxgradient_coding.append((i, j))

                saliency_reading[problem] = each_saliency_reading
                saliency_coding[problem] = each_saliency_coding
                inputxgradient_reading[problem] = each_inputxgradient_reading
                inputxgradient_coding[problem] = each_inputxgradient_coding

                # print(each_saliency_reading)
                # print(each_saliency_coding)
                # print(each_inputxgradient_reading)
                # print(each_inputxgradient_coding)
                # input()

                pickle.dump(saliency_reading, open(saliency_reading_filename, "wb"))
                pickle.dump(saliency_coding, open(saliency_coding_filename, "wb"))
                pickle.dump(inputxgradient_reading, open(inputxgradient_reading_filename, "wb"))
                pickle.dump(inputxgradient_coding, open(inputxgradient_coding_filename, "wb"))

                torch.cuda.empty_cache()


# def determine_generation_length(tokenizer):
#     with open("/home/bonan/fse2024/bert_perturbation_code/perturbed_prompts.pkl", "rb") as file:
#         perturbed_prompts = pickle.load(file)

#     # Load MBPP and HumanEval datasets
#     mbpp = load_dataset('mbpp')
#     humaneval = load_dataset('openai_humaneval')
#     humaneval_prompt = humaneval["test"]["prompt"]
#     humaneval_code = humaneval["test"]["canonical_solution"]
#     mbpp_prompt = mbpp["train"]["text"] + mbpp["test"]["text"] + mbpp["validation"]["text"] + mbpp["prompt"]["text"]
#     mbpp_code = mbpp["train"]["code"] + mbpp["test"]["code"] + mbpp["validation"]["code"] + mbpp["prompt"]["code"]
#     all_prompts = humaneval_prompt + mbpp_prompt
#     all_code = humaneval_code + mbpp_code
#     code_book = dict()
#     for index, i in enumerate(all_prompts):
#         code_book[i] = all_code[index]

#     # Create a function to retrieve solutions based on the prompt text
#     def get_solution_from_datasets(prompt_text, codebook):
#         return codebook[prompt_text]

#     token_counts = {}
#     for prompt_text in perturbed_prompts:
#         solution = get_solution_from_datasets(prompt_text["original_prompt"], code_book)
#         if solution:
#             tokens = len(tokenizer(solution, return_tensors="pt").input_ids[0]) + 20
#             token_counts[prompt_text["original_prompt"]] = tokens
#     print("Calculated output length.")
#     return token_counts