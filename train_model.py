"""`python train_model.py --cfg exp.yaml …` — the repository's command line for a training.

It forwards, nothing more. The implementation lives in the framework, where the server's job queue
and any script can reach it the same way: see matformer/training/launch.py.
"""
from matformer.training.launch import main

if __name__ == "__main__":
    raise SystemExit(main())
