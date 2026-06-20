import sys
def expand(template,N):
             expanded=[]
             for item in template:
                 parts=item.split('.')
                 is_N=False
                 for p in parts:
                     if p=='N':
                         is_N=True
                 if is_N:
                     for i in range(N):
                         expanded.append(item.replace("N",str(i)))
                 else:
                     expanded.append(item)
             return expanded

def collapse(_list):
             import re
             keyset=set()
             for item in _list:
                 parts=item.split('.')
                 new_parts=[]
                 for p in parts:
                     new_parts.append(re.sub(r'\d+', 'N', p))
                 keyset.add(".".join(new_parts))
             return list(keyset)

def get_compact_state_dict(ckpt_path):
         import torch
         sys.path.append("/home/jgili/matformer")
         from matformer.matformer_registry import registry
         from matformer.transformer_blocks import BERTModel, Autoregressive_Model
         ckpt=torch.load(ckpt_path,map_location='cpu',weights_only=False)
         if 'state_dict' in ckpt.keys():
              state_dict=ckpt['state_dict']
         else:
              state_dict=ckpt
         return collapse(list(state_dict.keys()))

if __name__ == "__main__":
    import sys
    import csv
    data = get_compact_state_dict(sys.argv[1])
    print(data)
    with open('state_dict_albertina_M6.csv', 'w', newline='') as csvfile:
        fieldnames = ['name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{'name': key} for key in data])
