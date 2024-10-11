from dadapy import data
from dadapy.data import Data
import torch
import numpy as np
import pandas as pd
from transformers import LlamaForCausalLM, LlamaTokenizer, \
    pipeline,AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset,concatenate_datasets
from datetime import datetime
import sys
import os
from huggingface_hub import login
from accelerate import load_checkpoint_and_dispatch
import datasets
import evaluate
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import argparse
import json

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def convert_to_numpy(hs):
    return np.array([item.to(torch.float32).cpu().detach().numpy() for item in hs])

def extract_hidden_states(sequence, model, tokenizer, max_length):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  with torch.no_grad():  # Disable gradient computation  
      inputs = tokenizer(sequence.strip() , add_special_tokens = False, \
                         return_tensors = "pt", max_length = max_length, \
                             truncation=True).to(device)
      outputs = model(**inputs, labels = inputs['input_ids'].clone(), output_hidden_states=True)
      hidden_states, loss = outputs.hidden_states, outputs.loss
  return convert_to_numpy(hidden_states), loss.to(torch.float32).cpu().detach().numpy()

def get_path(model_name):
    path_dict ={"Llama-3-8B": "meta-llama/Meta-Llama-3-8B", \
               "Mistral-7B" :"mistralai/Mistral-7B-v0.1", \
            "Pythia-6.9B"  :"EleutherAI/pythia-6.9b",\
        "Pythia-6.9B-Deduped"  :"EleutherAI/pythia-6.9b-deduped"}
    return path_dict[model_name]


def parse_arguments():
    parser = argparse.ArgumentParser()   
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    args = parser.parse_args()
    print("input args:\n", json.dumps(vars(args), indent=4, separators=(",", ":")))
    return args

args = parse_arguments()


login_token = '' # Fill in login_token here
login(token=login_token)
args = parse_arguments()
model_name = args.model_name
path = get_path(model_name)
input_dir = args.input_dir

print(f"Model = {model_name}, path = {path}")

tokenizer = AutoTokenizer.from_pretrained(path)   
model = AutoModelForCausalLM.from_pretrained(path,device_map="auto", torch_dtype=torch.bfloat16)
ds = load_dataset("NeelNanda/pile-10k")['train']
max_length = 1024
batch_sz = 16
sequences = ds['text'][0 : 10000]

lengths = np.array([len(tokenizer(sequence.strip())['input_ids']) for sequence in tqdm(sequences)])
filtered_indices =  np.where(lengths >= max_length)[0]

def compute_distances(hs):
    return  np.array([euclidean_distances(embeddings[0]) for embeddings in hs])

def compute_cosine_similarities(hs):
    return  np.array([cosine_similarity(embeddings[0]) for embeddings in hs])

filtered_sequences = [sequences[idx] for idx in filtered_indices]
for batch_start in tqdm(range(0, len(filtered_indices), batch_sz)):
    batch_sequences = filtered_sequences[batch_start: batch_start + batch_sz]
    intermediate_reps = [extract_hidden_states(sequence, model, tokenizer, max_length = max_length) \
                         for sequence in batch_sequences]
    hidden_states = np.array([item[0] for item in intermediate_reps])
    
    losses = np.array([item[1] for item in intermediate_reps])
    distances = Parallel(n_jobs=-1)(delayed(compute_distances)(hs) for hs in hidden_states)
    csn_sim = Parallel(n_jobs=-1)(delayed(compute_cosine_similarities)(hs) for hs in hidden_states)
    print(" ")
    output_folder = f"{input_dir}/{model_name}"
    batch_num = batch_start//batch_sz
    np.save(f"{output_folder}/distances/{batch_num}.npy", distances)
    np.save(f"{output_folder}/cosine_similarities/{batch_num}.npy", csn_sim)
    np.save(f"{output_folder}/losses/{batch_num}.npy", losses)
