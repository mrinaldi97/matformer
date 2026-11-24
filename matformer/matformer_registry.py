#matformer/matformer_registry.py
# MATFORMER_ASSETS: the shared folder for datasets, checkpoints, configs, logs and plugins.
# Declared by configuration.yaml in the repo root (or the MATFORMER_ASSETS env var); the folder
# tree is created here if it is configured but missing. Everything is optional: no declaration,
# no effect. Helpers and $ASSETS path expansion live in matformer/utils/matformer_assets.py.
from matformer.utils import matformer_assets
matformer_assets.bootstrap()

from matformer.utils.matformer_registry_implementation import MatformerRegistry
registry = MatformerRegistry()
registry.populate_registry()

# The Matformer server rides on the registry: wherever the framework is imported, it is up.
# Quietly skips when disabled (MATFORMER_SERVER=0) or when another process already holds the port.
from matformer.server import start_server
start_server()
