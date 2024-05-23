import pickle
from transformers import AutoTokenizer
from transformers import logging
import transformers
import torch
from pandas.core.computation.check import NUMEXPR_INSTALLED
import torch
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTJForCausalLM


class self_attention_based_method():
    def __init__(self, model_name, device, output_folder="."):
        self.output_folder = output_folder
        self.model_name = model_name
        self.token_counts = pickle.load(open("prompt_token_count.pkl", "rb"))
        self.device = device

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

        self.model.to(self.device)

    def find_full_function_body_end(self, code):
        lines = code.splitlines()
        
        start_line = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('def'):
                start_line = i
                break
        
        if start_line == -1:
            return -1

        indentation = len(lines[start_line]) - len(lines[start_line].lstrip())

        for i in range(start_line + 1, len(lines)):
            line = lines[i]
            if len(line.strip()) == 0: 
                continue
            if len(line) - len(line.lstrip()) <= indentation:
                return sum(len(l) + 1 for l in lines[:i]) - 1
            
        return len(code)

    def perfrom_self_attention_based(self, dataset):
        reading_first_filename = f"{self.output_folder}/reading_first_{self.model_name}.pkl"
        coding_first_filename = f"{self.output_folder}/coding_first_{self.model_name}.pkl"
        reading_last_filename = f"{self.output_folder}/reading_last_{self.model_name}.pkl"
        coding_last_filename = f"{self.output_folder}/coding_last_{self.model_name}.pkl"
        reading_all_filename = f"{self.output_folder}/reading_all_{self.model_name}.pkl"
        coding_all_filename = f"{self.output_folder}/coding_all_{self.model_name}.pkl"

        reading_first = dict()
        coding_first = dict()
        reading_last = dict()
        coding_last = dict()
        reading_all = dict()
        coding_all = dict()

        if os.path.exists(reading_first_filename):
            reading_first = pickle.load(open(reading_first_filename, "rb"))

        if os.path.exists(coding_first_filename):
            coding_first = pickle.load(open(coding_first_filename, "rb"))

        if os.path.exists(reading_last_filename):
            reading_last = pickle.load(open(reading_last_filename, "rb"))

        if os.path.exists(coding_last_filename):
            coding_last = pickle.load(open(coding_last_filename, "rb"))

        if os.path.exists(reading_all_filename):
            reading_all = pickle.load(open(reading_all_filename, "rb"))

        if os.path.exists(coding_all_filename):
            coding_all = pickle.load(open(coding_all_filename, "rb"))

        for problem in tqdm(dataset, desc="Processing dataset"):
            if problem not in reading_first:
                delta = len(self.tokenizer(self.token_counts[problem])["input_ids"]) + 20

                if problem.find("def") == -1:
                    # MBPP
                    with torch.no_grad():
                        # Original Prompt
                        original_token_ids = self.tokenizer(problem, return_tensors="pt")['input_ids'][0]
                        original_tokens_list = self.tokenizer.convert_ids_to_tokens(original_token_ids)
                        clean_tokens = [token.replace('Ġ', ' ').replace("Ċ", "\n").strip() for token in original_tokens_list]

                        # Prompt needs to generate the code
                        text = problem + "\ndef function():"
                        token_ids = self.tokenizer(text, return_tensors="pt")['input_ids'][0]  # Assuming you are using the first example if batched
                        tokens_list = self.tokenizer.convert_ids_to_tokens(token_ids)

                        # Generate code
                        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
                        length_input = len(input_ids[0])
                        generated_ids = self.model.generate(input_ids, max_length=length_input + delta, output_attentions = False, return_dict_in_generate=False)
                        code = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

                        input_2 = self.tokenizer.encode(code, return_tensors="pt")
                        input_id_list = input_2[0].tolist()

                        temp = self.tokenizer(code, return_tensors='pt').to(self.device)
                        output_2 = self.model(temp["input_ids"], output_attentions = True)
                        attention = output_2.attentions[0][0]

                        first_attentions = attention[0]
                        last_attentions = attention[-1]
                        all_attentions = torch.sum(attention, dim=0)

                        def get_token_attention(attention_list, flag):
                            result = []
                            if flag == "reading":
                                for i_index, i in enumerate(clean_tokens):
                                    if i != "" and problem.find(i) != -1:
                                        result.append((i, attention_list[len(clean_tokens)][i_index].item()))
                            else:
                                for i_index, i in enumerate(clean_tokens):
                                    if i != "" and problem.find(i) != -1:
                                        code_end_ids = self.tokenizer(code[:self.find_full_function_body_end(code)], return_tensors="pt")['input_ids'][0]
                                        code_end_lists = self.tokenizer.convert_ids_to_tokens(code_end_ids)
                                        code_end_index = len(code_end_lists)
                                        result.append((i, attention_list[code_end_index-1][i_index].item()))
                            return result

                        reading_first[problem] = get_token_attention(first_attentions, "reading")
                        coding_first[problem] = get_token_attention(first_attentions, "coding")
                        reading_last[problem] = get_token_attention(last_attentions, "reading")
                        coding_last[problem] = get_token_attention(last_attentions, "coding")
                        reading_all[problem] = get_token_attention(all_attentions, "reading")
                        coding_all[problem] = get_token_attention(all_attentions, "coding")

                        del attention, output_2,first_attentions, last_attentions, all_attentions

                        torch.cuda.empty_cache()
                else:
                    # HumanEval
                    with torch.no_grad():
                        start_range_true_prompt = len(self.tokenizer(problem[:problem.find("def")])['input_ids'])
                        end_range_true_prompt = len(self.tokenizer(problem[:problem.find(">>>")])['input_ids'])
 
                        token_ids = self.tokenizer(problem, return_tensors="pt")['input_ids'][0]  # Assuming you are using the first example if batched
                        tokens_list = self.tokenizer.convert_ids_to_tokens(token_ids)
                        clean_tokens = [token.replace('Ġ', ' ').replace("Ċ", "\n").strip() for token in tokens_list]

                        input_ids = self.tokenizer(problem, return_tensors="pt").input_ids.to(self.device)
                        length_input = len(input_ids[0])

                        generated_ids = self.model.generate(input_ids, max_length=length_input + delta, output_attentions = False, return_dict_in_generate=False)
                        code = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

                        temp = self.tokenizer(code, return_tensors='pt').to(self.device)
                        output_2 = self.model(temp["input_ids"], output_attentions = True)
                        attention = output_2.attentions[0][0]

                        first_attentions = attention[0]
                        last_attentions = attention[-1]
                        all_attentions = torch.sum(attention, dim=0)

                        def get_token_attention_humaneval(attention_list, flag):
                            result = []
                            if flag == "reading":
                                for i_index, i in enumerate(clean_tokens):
                                    if i != "" and problem.find(i) != -1 and i_index >= start_range_true_prompt and i_index < end_range_true_prompt and i.find(">>>") == -1:
                                        result.append((i, attention_list[len(clean_tokens)][i_index].item()))
                            else:
                                for i_index, i in enumerate(clean_tokens):
                                    if i != "" and problem.find(i) != -1 and i_index >= start_range_true_prompt and i_index < end_range_true_prompt and i.find(">>>") == -1:
                                        code_end_ids = self.tokenizer(code[:self.find_full_function_body_end(code)], return_tensors="pt")['input_ids'][0]
                                        code_end_lists = self.tokenizer.convert_ids_to_tokens(code_end_ids)
                                        code_end_index = len(code_end_lists)
                                        result.append((i, attention_list[code_end_index-1][i_index].item()))
                            return result

                        reading_first[problem] = get_token_attention_humaneval(first_attentions, "reading")
                        coding_first[problem] = get_token_attention_humaneval(first_attentions, "coding")
                        reading_last[problem] = get_token_attention_humaneval(last_attentions, "reading")
                        coding_last[problem] = get_token_attention_humaneval(last_attentions, "coding")
                        reading_all[problem] = get_token_attention_humaneval(all_attentions, "reading")
                        coding_all[problem] = get_token_attention_humaneval(all_attentions, "coding")

                        del attention, output_2,first_attentions, last_attentions, all_attentions
                        torch.cuda.empty_cache()      

                pickle.dump(reading_first, open(reading_first_filename, "wb"))
                pickle.dump(coding_first, open(coding_first_filename, "wb"))
                pickle.dump(reading_last, open(reading_last_filename, "wb"))
                pickle.dump(coding_last, open(coding_last_filename, "wb"))
                pickle.dump(reading_all, open(reading_all_filename, "wb"))
                pickle.dump(coding_all, open(coding_all_filename, "wb"))

                torch.cuda.empty_cache()
