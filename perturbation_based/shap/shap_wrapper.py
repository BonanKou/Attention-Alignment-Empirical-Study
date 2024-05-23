from nltk.tokenize import word_tokenize
from transformers import pipeline, BertTokenizer
import torch
import string
import re
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import json
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelWithLMHead, T5ForConditionalGeneration, RobertaForCausalLM, RobertaConfig
import pickle
from transformers import GPTJForCausalLM
# from utils import *
import openai
import concurrent.futures
from nltk.translate.bleu_score import sentence_bleu
import shap
import numpy as np
from transformers import logging
import pickle
from datasets import load_dataset
from transformers import AutoTokenizer

logging.set_verbosity_error()

class shap_perturbation:
    def __init__(self, model_name, device, output_folder = "/home/bonan/fse2024/data"):
        self.output_folder = output_folder
        self.device = device
        self.model_name = model_name
        if model_name == "codegen":
            self.tokenizer = AutoTokenizer.from_pretrained('Salesforce/codegen-2B-mono')
            self.model = AutoModelForCausalLM.from_pretrained('Salesforce/codegen-2B-mono')
        elif model_name == "codeparrot":
            self.tokenizer = AutoTokenizer.from_pretrained("codeparrot/codeparrot")
            self.model = AutoModelForCausalLM.from_pretrained("codeparrot/codeparrot")
        elif model_name == "gptj":
            self.model = GPTJForCausalLM.from_pretrained(
                "EleutherAI/gpt-j-6B",
                revision="float16",
                torch_dtype=torch.float16,
            )
            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")
        elif model_name == "incoder":
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/incoder-1B")
            self.model = AutoModelForCausalLM.from_pretrained("facebook/incoder-1B")
        elif model_name == "polycoder":
            self.tokenizer = AutoTokenizer.from_pretrained("NinedayWang/PolyCoder-2.7B")
            self.model = AutoModelForCausalLM.from_pretrained("NinedayWang/PolyCoder-2.7B")
        elif model_name == "gpt4":
            self.tokenizer = None
            self.model = None
        else:
            print(f"Unknown model name: {model_name}")
        self.determine_generation_length()


    def determine_generation_length(self):
        # Load MBPP and HumanEval datasets
        mbpp = load_dataset('mbpp')
        humaneval = load_dataset('openai_humaneval')
        humaneval_prompt = humaneval["test"]["prompt"]
        humaneval_code = humaneval["test"]["canonical_solution"]
        mbpp_prompt = mbpp["train"]["text"] + mbpp["test"]["text"] + mbpp["validation"]["text"] + mbpp["prompt"]["text"]
        mbpp_code = mbpp["train"]["code"] + mbpp["test"]["code"] + mbpp["validation"]["code"] + mbpp["prompt"]["code"]
        all_prompts = humaneval_prompt + mbpp_prompt
        all_code = humaneval_code + mbpp_code
        code_book = dict()
        for index, i in enumerate(all_prompts):
            code_book[i] = all_code[index]

        # Create a function to retrieve solutions based on the prompt text
        def get_solution_from_datasets(prompt_text, codebook):
            return codebook[prompt_text]

        self.token_counts = {}
        for prompt_text in all_prompts:
            solution = get_solution_from_datasets(prompt_text, code_book)
            if solution:
                tokens = len(self.tokenizer(solution, return_tensors="pt").input_ids[0]) + 20
                self.token_counts[prompt_text] = tokens
        print("Calculated output length.")

    def compare(self, original_code, perturbated_code):
        return 1 - sentence_bleu([original_code.split()], perturbated_code.split())

    def generate_code(self, text, original_text):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        length_input = len(input_ids[0])

        delta = 50

        for i in self.token_counts:
            if original_text == i or original_text.replace("\n", "").replace(" ", "") == i.replace("\n", "").replace(" ", ""):
                delta = self.token_counts[i]
                # print(f"For {original_text},\nComparison against {i},\ndelta is {delta}")
        
        generated_ids = self.model.generate(input_ids, 
                                            max_length=length_input + delta, 
                                            output_attentions = False, 
                                            return_dict_in_generate=False)
        code = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return code

    def calcualte_each_shap(self, original_text):
        def generate_with_caching(perturbed_text):
            if perturbed_text in cache:
                return cache[perturbed_text]
            else:
                if original_text.find("def") != -1:
                    cache[perturbed_text] = self.generate_code(perturbed_text, original_text)
                else:
                    cache[perturbed_text] = self.generate_code(perturbed_text + "\ndef function():", original_text)
            return cache[perturbed_text]
    
        def model_function(binary_vectors, perturbed_texts, original_output):
            explanations = []
            for binary_vector in binary_vectors:
                # Find the corresponding perturbed text
                perturbed_text = perturbed_texts[np.argmax(np.all(binary_matrix == binary_vector, axis=1))]
                # print("Generating for:", perturbed_text)
                perturbed_output = generate_with_caching(perturbed_text)  # Get LLM prediction for perturbed text
                # print("Output:", perturbed_output)
                score = self.compare(original_output, perturbed_output)  # Compare perturbed output to original output
                explanations.append(score)
            return np.array(explanations)
        
        def generate_perturbed_texts(tokens):
            perturbed_texts = []
            for idx in range(len(tokens)):
                perturbed_tokens = tokens.copy()
                perturbed_tokens[idx] = "[MASK]"  # Replace selected tokens with a placeholder
                perturbed_text = " ".join(perturbed_tokens)
                perturbed_texts.append(perturbed_text)
            return perturbed_texts
        
        if original_text.find("def") != -1:
            original_output = self.generate_code(original_text, original_text)
        else:
            original_output = self.generate_code(original_text + "\ndef function():", original_text)

        cache = {}
        
        if original_text.find(">>>") != -1:
            tokens = word_tokenize(original_text[original_text.find("def"):original_text.find(">>>")])
        else:
            tokens = word_tokenize(original_text)

        background_dataset = generate_perturbed_texts(tokens)

        binary_matrix = []

        for perturbed_text in background_dataset:
            row = [1 if token not in perturbed_text.split() or token == "[MASK]" else 0 for token in tokens]
            binary_matrix.append(row)
        binary_matrix = np.array(binary_matrix)

        explainer = shap.KernelExplainer(lambda x: model_function(x, background_dataset, original_output), binary_matrix)

        try:
            shap_values = explainer.shap_values(np.ones((1, len(tokens))), nsamples=50)  # Explaining with all tokens "present"
            importances = []
            raw_shap_values = shap_values[0]
            abs_shap_values = list(map(abs, raw_shap_values))
            for token, shap_value in zip(tokens, abs_shap_values):
                importances.append((token, shap_value))
        except:
            print("Error in SHAP")
            importances = []
            for token in tokens:
                importances.append((token, 1))

        return importances

    def perform_shap_perturbation(self, dataset):
        self.model = self.model.to(self.device)

        file_name = f"{self.output_folder}/shap_perturbation_{self.model_name}.pkl"

        if os.path.exists(file_name):
            with open(file_name, 'rb') as file:
                result = pickle.load(file)
            print(f"Data loaded from {file_name}")
        else:
            result = dict()
            print(f"File {file_name} does not exist. Created an empty dictionary.")

        for i in tqdm(dataset):
            if i not in result:
                importances = self.calcualte_each_shap(i)
                print(importances)
                result[i] = importances
                pickle.dump(result, open(file_name, "wb"))