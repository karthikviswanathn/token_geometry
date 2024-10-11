import numpy as np
import glob
from natsort import natsorted
import tqdm 
from tqdm import tqdm
from scipy.spatial.distance import cdist
from dadapy import data
from dadapy.data import Data
import argparse
import json
import os
from sklearn.metrics.pairwise import euclidean_distances
from datasets import load_dataset
from joblib import Parallel, delayed

def parse_arguments():
    parser = argparse.ArgumentParser()   
    parser.add_argument("--input_dir", type=str, default=None)
    args = parser.parse_args()
    print("input args:\n", json.dumps(vars(args), indent=4, separators=(",", ":")))
    return args

def compute_ids_for_all_layers(reps):
    all_ids = []
    for old_rep in reps[1:]: # Not calculating stuff for embedding layer
        _, indices = np.unique(old_rep, axis = 0, return_index=True)
        rep = old_rep[indices, :][:, indices]
        _data = data.Data(distances = rep, maxk = 300)
        all_ids.append(_data.return_id_scaling_gride(range_max = 256))
    return np.array(all_ids)

def compute_ids(input_file):
    reps = np.load(input_file)
    all_ids = []
    for rep in tqdm(reps):
        all_ids.append(compute_ids_for_all_layers(rep))
    return np.array(all_ids)

def read_and_write_gride(input_dir, n_batches = 71):
    ids = Parallel(n_jobs=-1)(delayed(compute_ids)(f"{input_dir}/distances/{idx}.npy")\
                              for idx in range(n_batches)) 
    np.save(f"{input_dir}/summaries/gride.npy", np.concatenate(ids, axis = 0))


def compute_csn(input_file):
    reps = np.load(input_file)
    all_csn = []
    for rep in tqdm(reps): # Iterating through prompts in this batch
        all_csn.append([np.mean(rep, axis = (-1, -2)), np.std(rep, axis = (-1, -2))])
    return np.array(all_csn)

def read_and_write_csn(input_dir, n_batches = 71):
    csn = Parallel(n_jobs=-1)(delayed(compute_csn)(f"{input_dir}/cosine_similarities/{idx}.npy")\
                              for idx in range(n_batches)) 
    np.save(f"{input_dir}/summaries/csn_sim.npy", np.concatenate(csn, axis = 0))
    
def compute_neighborhood_overlap(rep, next_rep, maxk = 32):
    rep = np.argsort(rep, axis = 1)
    next_rep = np.argsort(next_rep, axis = 1)
    
    n = len(rep)
    # Initialize the answer array to hold the number of overlaps at each knn level
    ans = np.zeros((n, maxk))
    
    for idx in range(n): # Iterating through tokens
        
        # For each index, create a boolean array for quick lookups
        rep_flags = np.zeros(np.max(rep) + 1, dtype=bool)
        next_rep_flags = np.zeros(np.max(next_rep) + 1, dtype=bool)
        
        # Iterate over knn and count the number of intersections
        for knn in range(1, maxk):
            rep_flags[rep[idx, knn]] = True
            next_rep_flags[next_rep[idx, knn]] = True
            ans[idx, knn] = np.sum(rep_flags & next_rep_flags)/knn 
    
    return ans.mean(axis=0)

def compute_nos_for_all_layers(reps):
    all_nos = []
    for rep, next_rep in zip(reps[1:-1], reps[2:]): # Not calculating stuff for embedding layer
        all_nos.append(compute_neighborhood_overlap(rep, next_rep))
    return np.array(all_nos)

def compute_nos(input_file):
    reps = np.load(input_file)
    all_nos = []
    for rep in tqdm(reps):
        all_nos.append(compute_nos_for_all_layers(rep))
    return np.array(all_nos)

def read_and_write_no(input_dir, n_batches = 71):
    mnos = Parallel(n_jobs=-1)(delayed(compute_nos)(f"{input_dir}/distances/{idx}.npy")\
                              for idx in range(n_batches)) 
    np.save(f"{input_dir}/summaries/mnos.npy", np.concatenate(mnos, axis = 0))
    

args = parse_arguments()
input_dir = args.input_dir
read_and_write_gride(input_dir)
read_and_write_csn(input_dir)
read_and_write_no(input_dir)
