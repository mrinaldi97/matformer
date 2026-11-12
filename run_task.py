import argparse
import os
import sys
import torch
from matformer.matformer_registry import registry
from matformer.matformer_wrapper import MatformerModelWrapper, load_model
from matformer.matformer_tokenizers import MatformerTokenizer
from matformer.matformer_configs import MatformerConfigs
def main():
    p = argparse.ArgumentParser(
        description=(
            "Run any Matformer task on a checkpoint. "
            "Pass one or more YAMLs via --cfg; they merge left-to-right. "
            "Each file may contain any combination of recognised sections."
        )
    )
    p.add_argument("--model",       required=True,  help="Path to model checkpoint")
    p.add_argument("--tokenizer",   default=None,   help="HF tokenizer id; omit to use checkpoint default")
    p.add_argument("--cfg",         action="append", required=True, metavar="YAML")
    p.add_argument("--override",   action="append", default=None, metavar="FIELD=VALUE",
                    help="In-line override applied after the YAMLs (ex. plugin.store.filename=...)")
    p.add_argument("--num_devices", type=int, default=1)
    args = p.parse_args()

    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_arg = (MatformerTokenizer.from_huggingface(args.tokenizer)
                      if args.tokenizer else "default")

    configs = MatformerConfigs.create_configs(args.cfg, overrides=args.override)

    model_dict = load_model(
        path=args.model,
        map_location=device, dtype=torch.bfloat16, inference=True, configs=configs
    )
    model=model_dict['model']
    configs=model_dict['configs']
    print(configs.model)
    print(configs.tokenizer)
    print(configs.data.test)

    for task_name, task_config in configs.task:
        task=task_config.instantiate(model=model)
        result=task.run()
        if result is not None:
            print(result)

if __name__ == "__main__":
    main()