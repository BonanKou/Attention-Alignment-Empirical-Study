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

from transformers import logging

logging.set_verbosity_error()

def code_similarity(original_prompt, perturbated_prompt, original_code, perturbated_code):
    original_code = original_code[len(original_prompt):]
    perturbated_code = perturbated_code[len(perturbated_prompt):]
    return 1 - sentence_bleu([original_code.split()], perturbated_code.split())

def map_token_to_string(tokens, s):
    result = []
    start_index = 0
    for i in range(len(tokens)):
        token = tokens[i]
        while s[start_index:start_index + len(token)] != token:
            tmp = s[start_index:start_index + len(token)]
            start_index += 1
        result.append((start_index, start_index + len(token)))
    return result


def extract_description(text, offset, res):
    _, description_start_index = re.search('\"\"\"', text).span()
    description_start_index += 1
    if re.search('\"\"\"', text[description_start_index:]):
        description_end_index, _ = re.search('\"\"\"', text[description_start_index:]).span()
        description_end_index = description_end_index + description_start_index
    else:
        return res
    if re.search('>>>', text[description_start_index:description_end_index]):
        description_end_index, _ = re.search('>>>', text[description_start_index:description_end_index]).span()
        description_end_index = description_start_index + description_end_index
    res.append((description_start_index + offset, description_end_index + offset))
    if re.search('\"\"\"', text[description_end_index + offset + 3:]):
        return extract_description(text[description_end_index + offset + 3:], description_end_index + offset + 3, res)
    else:
        return res


def check_if_punctuation(s):
    if len(s) == 1:
        return s in string.punctuation
    else:
        for v in set(s):
            if v not in string.punctuation:
                return False
        return True

class perturbationTokenImportance:
    def __init__(self, model='bert-base-uncased', batch_size=8, top_k=1, device="cuda:3"):
        self.unmasker = pipeline('fill-mask', model=model, batch_size=batch_size, top_k=top_k,
                                 device=torch.device(device))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def get_prediction(self, text):
        self.masked_text = []
        self.masked_tokens = []
        self.original_text = text
        tmp_text = self.original_text
        tokens = word_tokenize(tmp_text.replace('\n', ' '))
        tokens = [v for v in tokens if not check_if_punctuation(v)]
        token_indexes = map_token_to_string(tokens, self.original_text)
        assert len(token_indexes) == len(tokens)
        masked_text = []
        masked_tokens = []
        for i in range(len(tokens)):
            token = tokens[i]
            masked_tokens.append(token)
            start_index, end_index = token_indexes[i]
            masked_text.append(self.original_text[:start_index] + '[MASK]' + self.original_text[end_index:])
        #             print(masked_text[-1])
        self.masked_text += [masked_text]
        self.masked_tokens += [masked_tokens]

    def unmasking(self):
        filled_prompts = []
        self.filled_tokens = None
        for i in range(len(self.masked_text)):
            masked_text = self.masked_text[i]
            results = self.unmasker(masked_text)
            filled_token = [v[0]['token_str'] for v in results]
            if not self.filled_tokens:
                self.filled_tokens = list(zip(self.masked_tokens[i], filled_token))
            else:
                self.filled_tokens += list(zip(self.masked_tokens[i], filled_token))
            filled_description = [masked_text[j].replace('[MASK]', filled_token[j]) for j in range(len(masked_text))]
            filled_prompts = filled_description
        return filled_prompts

    def get_perturbated_prompts(self, input_prompt):
        self.get_prediction(input_prompt)
        new_prompts = self.unmasking()
        return {'original_prompt': input_prompt, 'new_prompts': new_prompts, 'filled_tokens': self.filled_tokens}

# This function returns the index of end of the generated function. For example, this function returns the index 
# of the last "s" in "pass" given the following code:
# def function():
#   pass
def find_full_function_body_end(code):
    # Split the code into lines
    lines = code.splitlines()
    
    # Find the line where the first function definition starts
    start_line = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('def'):
            start_line = i
            break
    
    if start_line == -1:
        return -1  # No function definition found

    # Determine the indentation level of the function definition
    indentation = len(lines[start_line]) - len(lines[start_line].lstrip())

    # Find where the function body ends
    for i in range(start_line + 1, len(lines)):
        line = lines[i]
        # Check if the line has lesser indentation or is completely empty
        if len(line.strip()) == 0:  # Consider empty lines as part of the function
            continue
        if len(line) - len(line.lstrip()) <= indentation:
            # Return the index of the end of the last line of the function body
            return sum(len(l) + 1 for l in lines[:i]) - 1
    
    # If function goes till the end of the file
    return len(code)