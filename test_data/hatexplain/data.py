import json
import os
from collections import Counter

def process_hatexplain(folder, data_path, split_path):
    with open(folder+data_path, 'r') as f:
        full_data = json.load(f)
    
    with open(folder+split_path, 'r') as f:
        splits = json.load(f)

    processed_splits = {
        'train': [],
        'test': [],
        'val': []
    }

    for split_name, post_ids in splits.items():
        for post_id in post_ids:
            if post_id in full_data:
                entry = full_data[post_id].copy()
                
                entry['full_text'] = " ".join(entry['post_tokens'])
                
                # Keep the original post_id for reference
                entry['post_id'] = post_id
                
                # take the max of annotator label
                labels = [label_converter(ann['label']) for ann in entry['annotators']]
                entry['final_label'] = Counter(labels).most_common(1)[0][0]
                
                del entry['annotators']
                
                processed_splits[split_name].append(entry)

    for split_name, data in processed_splits.items():
        output_filename = f"{split_name}.json"
        with open(folder+output_filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Saved {len(data)} samples to {output_filename}")

# hatespeech (0), normal (1) or offensive (2)
def label_converter(label):
  label_dict = {
    'hatespeech':0,
    'normal':1,
    "offensive":2
  }
  return label_dict[label]

if __name__ == "__main__":
    process_hatexplain('test_data/hatexplain/', 'dataset.json', 'post_id_divisions.json')