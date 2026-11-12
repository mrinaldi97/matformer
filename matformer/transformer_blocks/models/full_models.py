from .language_models import Autoregressive_Model, BERTModel
from matformer.matformer_registry import registry
@registry.register("model","language_model","default")
class LanguageModel(Autoregressive_Model,BERTModel):
   pass
