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
from utils import *
import openai
import concurrent.futures
# Import necessary modules
import pickle
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import logging

logging.set_verbosity_error()

class bert_perturbation:
    def __init__(self, model_name, device, output_folder="."):
        self.original_dataset = pickle.load(open("dataset.pkl", "rb"))
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
        
        if model_name != "gpt4":
            self.token_counts = pickle.load(open("prompt_token_count.pkl", "rb"))

    def not_gpt(self):
        return self.model_name != "gpt4"
    
    def perform_bert_perturbation(self, dataset):

        if self.not_gpt():
            self.model = self.model.to(self.device)

        # Generate the file name
        file_name = f"{self.output_folder}/bert_perturbation_{self.model_name}.pkl"

        if os.path.exists(file_name):
            with open(file_name, 'rb') as file:
                result = pickle.load(file)
            print(f"Data loaded from {file_name}")
        else:
            result = dict()
            print(f"File {file_name} does not exist. Created an empty dictionary.")
        
        gpt4_generations_path = f"{self.output_folder}/gpt4_generations.pkl"
        if not self.not_gpt() and os.path.exists(gpt4_generations_path):
            gpt4_generations = pickle.load(open(gpt4_generations_path, "rb"))
            print("Logs of GPT4 generations for each perturbed result loaded from existing file.")
        elif not self.not_gpt():
            print("Logs of GPT4 generations for each perturbed result created from scratch.")
            gpt4_generations = {}

        for i in tqdm(dataset):
            if i["original_prompt"] not in result or i["original_prompt"] == "Write a function that matches a word containing 'z', not at the start or end of the word.":
                if self.not_gpt():
                    with torch.no_grad():
                        delta = len(self.tokenizer(self.token_counts[i["original_prompt"]])["input_ids"]) + 20
                        print(delta)

                        # for raw_prompt in self.token_counts:
                        #     if raw_prompt == i["original_prompt"] or raw_prompt.replace("\n", "").replace(" ", "") == i["original_prompt"].replace("\n", "").replace(" ", ""):
                        #         delta = self.token_counts[raw_prompt]

                        def generate_code(text):
                            input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
                            length_input = len(input_ids[0])
                            generated_ids = self.model.generate(input_ids, max_length=length_input + delta, output_attentions = False, return_dict_in_generate=False)
                            code = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                            return code
                        
                        original_prompt = i["original_prompt"]
                        
                        if original_prompt.find(">>>") != -1:
                            perturbations = [original_prompt[:original_prompt.find("def")] + j + original_prompt[original_prompt.find(">>>"):] for j in i["new_prompts"]]
                        else:
                            original_prompt = i["original_prompt"] + "\ndef function():"
                            perturbations = [j+"\ndef function():" for j in i["new_prompts"]]

                        original_code = generate_code(original_prompt)
                        tokens = [j[0] for j in i["filled_tokens"]]
                        importances = []

                        # print(original_prompt)
                        # print(len(tokens))
                        # print(delta)

                        for index, j in enumerate(tokens):
                            perturbated_code = generate_code(perturbations[index])
                            importance = code_similarity(original_prompt, perturbations[index], original_code, perturbated_code)
                            importances.append((j, importance))

                        result[i["original_prompt"]] = importances

                        pickle.dump(result, open(f"{self.output_folder}/bert_perturbation_{self.model_name}.pkl", "wb"))
                        
                        torch.cuda.empty_cache()
                else:
                    input_price_per_1000_tokens = 0.01
                    output_price_per_1000_tokens = 0.03
                    gpt4_generations_for_this_prompt = []

                    def llm(prompt):
                        client = openai.OpenAI(
                            api_key= "",
                        )
                        chat_completion = client.chat.completions.create(
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are a helpful assistant that write Python code in a very concise manner. Keep your answers short.",
                                },
                                {
                                    "role": "user",
                                    "content": prompt,
                                }
                            ],
                            model="gpt-4-1106-preview",
                        )

                        input_tokens = chat_completion.usage.prompt_tokens
                        output_tokens = chat_completion.usage.completion_tokens

                        cost_for_this_prompt = input_tokens * input_price_per_1000_tokens / 1000 + output_tokens * output_price_per_1000_tokens / 1000

                        return chat_completion.choices[0].message.content, cost_for_this_prompt

                    def gpt4_code_similarity(original_code, perturbated_code):
                        return 1 - sentence_bleu([original_code.split()], perturbated_code.split())

                    def compute_importance(index, perturbation, token, original_prompt, original_code):
                        perturbated_code, cost = llm(perturbation)
                        # print("雄关漫道", perturbated_code, cost)
                        importance = gpt4_code_similarity(original_code, perturbated_code)
                        gpt4_generations_for_this_prompt.append((token, perturbated_code))
                        return (token, importance), cost

                    original_prompt = i["original_prompt"]
                    
                    if original_prompt.find(">>>") != -1:
                        perturbations = [original_prompt[:original_prompt.find("def")] + j + original_prompt[original_prompt.find(">>>"):] for j in i["new_prompts"]]
                    else:
                        original_prompt = i["original_prompt"] + "\ndef function():"
                        perturbations = [j+"\ndef function():" for j in i["new_prompts"]]

                    original_code, gpt4_cost_for_this_prompt = llm(original_prompt)
                    tokens = [j[0] for j in i["filled_tokens"]]
                    importances = []

                    tqdm_description = i["original_prompt"]

                    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
                        futures = [
                            executor.submit(compute_importance, idx, perturbations[idx], tokens[idx], original_prompt, original_code)
                            for idx in range(len(tokens))
                        ]
                        
                        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                            importances.append(future.result()[0])
                            gpt4_cost_for_this_prompt += future.result()[1]

                    result[i["original_prompt"]] = importances

                    print(f"Perturbed {len(importances)} times. Price: {gpt4_cost_for_this_prompt}")
                    print(f"Calculate results: {importances}")
                    print(f"GPT4 generation for each perturbation: {gpt4_generations_for_this_prompt}")

                    gpt4_generations[i["original_prompt"]] = gpt4_generations_for_this_prompt

                    pickle.dump(result, open(f"{self.output_folder}/bert_perturbation_{self.model_name}.pkl", "wb"))
                    pickle.dump(gpt4_generations, open(f"gpt4_generations.pkl", "wb"))