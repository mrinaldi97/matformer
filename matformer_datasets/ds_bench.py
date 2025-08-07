import argparse
import time
import random
from tqdm import tqdm
from datasets import Dataset,load_dataset
from tqdm import tqdm
from torch_atlas_ds import AtlasDataset, AtlasDatasetWriter  

def create_dataset(dataset_path):
    try:
        ds = AtlasDataset(dataset_path)
    except Exception:
        ds = load_dataset("json", data_files=dataset_path, split="train")
    return ds

def test_read_performance(ds, test_type, num_elements):
    print(f"Starting {test_type} read test...")
    if test_type == "random":
        pointers = list(range(num_elements))
        random.shuffle(pointers)
    else:
        pointers = range(num_elements)

    start_time = time.perf_counter()
    for i in tqdm(pointers, desc=f"{test_type.capitalize()} Read"):
        _ = ds[i]
    end_time = time.perf_counter()
    print(f"{test_type.capitalize()} read completed in {end_time - start_time:.4f} seconds.\n")
def atlas2huggingface(atlas_path):
    def _atlasGenerator():
        ds = AtlasDataset(atlas_path)
        for row in tqdm(ds):
            yield row
    hf_ds = Dataset.from_generator(_atlasGenerator)
    return hf_ds
#hf_ds=atlas2huggingface('liberliber1024')
def main():
    parser = argparse.ArgumentParser(description="Benchmark script for dataset read performance.")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset file.")
    parser.add_argument("--disable-sequential", action="store_true", help="Disable the sequential read test.")
    parser.add_argument("--disable-random", action="store_true", help="Disable the random read test.")
    args = parser.parse_args()

    ds = create_dataset(args.dataset_path)
    num_elements = len(ds)

    if not args.disable_sequential:
        test_read_performance(ds, "sequential", num_elements)
    
    if not args.disable_random:
        test_read_performance(ds, "random", num_elements)

if __name__ == "__main__":
    main()







