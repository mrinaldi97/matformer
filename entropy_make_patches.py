"""
Script to perform exploratory analysis on an entropy model as well as 
creating a dataset of patches together with their occurrences count
with the possibility of varying the amount of "smoothing".

The input is a Matformer dataset, the output are a raw txt with all the 
patches and a CSV dictionary with the statistics for each patch.

"""
import argparse, torch, sys, re, os
from matformer.models import PL_ModelWrapper
from matformer.tokenizers import MatformerTokenizer
from matformer.transformer_blocks import EntropyModel
from matformer.matformer_dataset import MatformerDataset
from itertools import islice
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split
import json
import numpy as np
import matplotlib.pyplot as plt

# PARSE ARGUMENTS
p = argparse.ArgumentParser("Entropy model dataset creator")
p.add_argument('--model', required=True)
p.add_argument('--dataset', required=True)
p.add_argument('--output', default='entropy_model')
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--max_batches', type=int, default=None) 
p.add_argument('--smoothing', type=float, default=0.0)
p.add_argument('--hard_limit',type=int,default=None)
args = p.parse_args()

output_dir = os.path.join(args.output, f"smoothing_{args.smoothing}")
os.makedirs(output_dir, exist_ok=True)

# LOAD MODEL, TOKENIZER AND DATASET
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, cfg = PL_ModelWrapper.load_from_checkpoint(
    checkpoint_path=args.model,
    ModelClass=EntropyModel,
    map_location=device,
    inference_fix=True,
    tokenizer='bytes'
)
print("Loaded model on", device)
print("Config:", cfg)
model = model.to(device).to(torch.bfloat16).eval()
for module in model.modules():
    if hasattr(module, "alibi_slopes") and module.alibi_slopes is not None:
        module.alibi_slopes = module.alibi_slopes.to(dtype=torch.float32)

tokenizer=MatformerTokenizer(cfg,'bytes','unpadding')
n_bytes=cfg.max_position_embeddings

ds=MatformerDataset(args.dataset,'bytes',n_bytes=n_bytes,return_type='text',byte_tokenizer=tokenizer)


# COMPUTE ENTROPY

chunkset = set()
chunkdict = {}
maxlen = 0
minlen = float('inf')
totlen = 0
n_chunks = 0
avglen = 0
hard_limit_violations=0
lengths = [] 

# Batching input
ds_iterator = iter(ds)
num_batches = len(ds) // args.batch_size
if args.max_batches is None:
    args.max_batches = num_batches
    
for _ in tqdm(range(min(num_batches, args.max_batches)), desc="Batches"):
   batch = list(islice(ds_iterator, args.batch_size)) 
   if not batch:
       break
   
   cutting_points,_,_=model.model.monotonicity_breakpoints(prompt=batch,smoothing=args.smoothing)
   all_chunks,part_hard_limit_violations = model.model.cut_text(batch, cutting_points, hard_limit=args.hard_limit)
   hard_limit_violations+=part_hard_limit_violations
   for chunks in all_chunks:
       for chunk in chunks:
           l = len(chunk)
           chunkdict[chunk] = chunkdict.get(chunk, 0) + 1
           maxlen = max(maxlen, l)
           minlen = min(minlen, l)
           totlen += l
           n_chunks += 1
           lengths.append(l)

avglen = totlen / n_chunks

chunkset=set(chunkdict.keys())
print(f"Number of chunks: {n_chunks}\n Total bytes processed: {totlen} \n Maximum length: {maxlen} \n Minimum length: {minlen}\n Average length: {avglen}\n Unique chunks {len(chunkset)} \n Hard Limit violations: {hard_limit_violations}")
print("Salvo il set completo...")

with open(os.path.join(output_dir, f"{args.output}-s{args.smoothing}_set.jsonl"), "w") as f:
    for chunk in chunkset:
        f.write(f"{json.dumps(chunk)}\n")

print("Salvo il dizionario ordinato con le occorrezne")

sorted_chunkdict = sorted(chunkdict.items(), key=lambda item: item[1], reverse=True)
with open(os.path.join(output_dir, f"{args.output}-s{args.smoothing}_stat.txt"), "w") as f: 
    for elem in sorted_chunkdict: 
        f.write(f"{json.dumps(elem)}\n")

print("Creo train 80%, test split 20%...")
train,test=train_test_split(list(chunkset),train_size=0.8,shuffle=True)
print(f"Creati train: {len(train)}, test: {len(test)}")
print("Li salvo...")
with open(os.path.join(output_dir, f"{args.output}-s{args.smoothing}_train_set.jsonl"), "w") as f:
    for chunk in train:
        f.write(f"{json.dumps(chunk)}\n")
print("Train salvato")
with open(os.path.join(output_dir, f"{args.output}-s{args.smoothing}_test_set.jsonl"), "w") as f:
    for chunk in test:
        f.write(f"{json.dumps(chunk)}\n")
print("Test salvato. Calcolo le statistiche avanzate...")
lengths_np = np.array(lengths)
below_avg = np.sum(lengths_np < avglen)
above_avg = np.sum(lengths_np > avglen)
equal_avg = np.sum(lengths_np == avglen)

stats_text = [
    f"Number of chunks: {n_chunks}",
    f"Total bytes processed: {totlen}",
    f"Maximum length: {maxlen}",
    f"Minimum length: {minlen}",
    f"Average length: {avglen:.2f}",
    f"Unique chunks: {len(chunkset)}",
    f"Hard Limit violations: {hard_limit_violations}",
    f"Patches below average length: {below_avg}",
    f"Patches above average length: {above_avg}",
    f"Patches exactly average length: {equal_avg}",
    f"Length standard deviation: {np.std(lengths_np):.2f}",
    f"Median length: {np.median(lengths_np):.2f}"
]

with open(os.path.join(output_dir, f"{args.output}-s{args.smoothing}_extra_stats.txt"), "w") as f:
    for line in stats_text:
        f.write(line + "\n")

print("Faccio i grafici...")

# Histogram of patch lengths
plt.figure(figsize=(8, 5))
plt.hist(lengths_np, bins=50, color='skyblue', edgecolor='black')
plt.title("Distribution of Patch Lengths")
plt.xlabel("Patch length")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "patch_length_distribution.png"))
plt.close()

# Top 50 most frequent patches occurrence plot
top_n = 50
top_patches = sorted_chunkdict[:top_n]
labels, counts = zip(*top_patches)
plt.figure(figsize=(10, 6))
plt.bar(range(top_n), counts, color='salmon')
plt.title(f"Top {top_n} Most Frequent Patches")
plt.xlabel("Patch rank")
plt.ylabel("Occurrences")
plt.xticks(range(top_n), range(1, top_n+1))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "top_patches_frequency.png"))
plt.close()

print("Pulizia...")
