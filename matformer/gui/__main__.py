# matformer/gui/__main__.py
"""`python -m matformer.gui`. The guard matters: a training is spawned as a fresh process, and the
child re-imports whatever the parent's __main__ is — an unguarded call here would launch a GUI
inside every job."""
from matformer.gui.main import main

if __name__ == "__main__":
    main()
