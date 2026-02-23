import argparse
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
from matformer.matformer_tokenizers import ByteLevelTokenizer, MatformerTokenizer
from matformer.extra.muon import Muon
from matformer.model_config import ModelConfig
from matformer.masked_models import Maskerator
from matformer.initialization import init_transformer_weights_

# import matformer.transformer_blocks
from matformer.tensors_dataclasses import PaddedTensor, UnpaddedTensor, NormalTensor

# from matformer.debug_methods import train_debug_print
from torch.optim import AdamW, Adam
from transformers import get_scheduler
import math
import torch.distributed as dist
import numpy as np
from dataclasses import replace

# from copy import deepcopy Provo a rimuovere
from transformers import AutoTokenizer
from matformer.matformer_registry import registry
from matformer.cached_stuff import CachedStuff
from copy import deepcopy
from matformer.matformer_module import MatformerModule

import sys


class PL_ModelWrapper(MatformerModule):
    def __init__(
        self,
        ModelClass,
        config,
        tokenizer,
        device,
        batch_size=None,
        train_config=None,
        inference=False,
        load_mode="full",
        init_weights=True,
        **model_kwargs,
    ):
        super().__init__()
        self.config = config
        self.train_config = train_config
        self.save_hyperparameters()
        self.load_mode = load_mode

        # Initialize cache and set registry
        self.cache = CachedStuff()
        self.cache.registry = registry

        self.model = ModelClass(
            config, tokenizer=tokenizer, device=device, cache=self.cache, **model_kwargs
        )
        self.nested = None
        if self.config.loss_type == "fused":
            self.cross_entropy_loss = self.cache.registry.create(
                "loss",
                "cross_entropy_loss_fused",
                *[],
                **{"ignore_index": config.pad_token_id},
            )
        else:  
            self.cross_entropy_loss = self.cache.registry.create(
                "loss",
                "cross_entropy_loss",
                *[],
                **{"ignore_index": config.pad_token_id},
            )
        ### ATTENZIONE: Bug sistemato al volo perchè bloccava l'addestramento della testa di classificazione, ora però manca l'iniziallizzazione dei pesi del classificatore, da sistemare presto (molto facile)
        # E rendere anche più eleganti questi parametri
        if init_weights==False:
            self._restored_from_ckpt=True
        if not getattr(self, "_restored_from_ckpt", False):
            self.model.apply(init_transformer_weights_)

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        # Maskerator setup
        self.maskerator = Maskerator(
            mask_token=self.config.mask_token_id,
            substitution_rate=self.config.masked_substitution_rate,
            pad_token_id=self.config.pad_token_id,
            cloze_prob=self.config.cloze_probability,
            random_prob=self.config.random_probability,
            same_prob=self.config.same_probability,
            vocab_size=self.config.vocab_size,
        )

    def forward(self, _input, *args, **kwargs):       
        if isinstance(_input, torch.Tensor):
            _input = NormalTensor(tensor=_input)
        output = self.model(_input.to(self.device), *args, **kwargs)
        # print(f"Model output - mean: {output.mean().item():.4f}, std: {output.std().item():.4f}, min: {output.min().item():.4f}, max: {output.max().item():.4f}", file=log_file)
        return output

    def on_load_checkpoint(self, checkpoint):
        self._restored_from_ckpt = True
        if self.load_mode in ["weights_only", "weights_and_optimizer"]:
            checkpoint["lr_schedulers"] = []
            checkpoint["epoch"] = 0
            checkpoint["global_step"] = 0
            checkpoint.pop("loops", None)
            checkpoint.pop("MatformerDataModule", None)
            checkpoint.pop("callbacks", None)
        if self.load_mode == "weights_only":
            checkpoint["optimizer_states"] = []
        if self.load_mode == "publication":
            new_checkpoint = checkpoint["state_dict"]
            new_checkpoint = checkpoint["hyper_parameters"]

    def training_step(self, batch, batch_idx=None):
        if "labels" in batch:
            return self._classification_step(batch)
        else:
            return self._pretraining_step(batch)

    def _classification_step(self, batch):
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        logits = self(input_ids)
        is_token_level = len(logits.shape) == 3  # (B, S, C) vs (B, C)

        loss = self._compute_loss(logits, labels, is_token_level)
        acc = self._compute_accuracy(logits, labels, is_token_level)

        batch_size = input_ids.shape[0]
        self.log(
            "train/classification_loss", loss, prog_bar=True, batch_size=batch_size
        )
        self.log(
            "train/classification_accuracy",
            acc,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )

        try:
            self.log(
                "lr",
                self.lr_schedulers().get_last_lr()[0],
                prog_bar=True,
                on_step=True,
                batch_size=batch_size,
            )
        except:
            pass

        return loss

    def _compute_loss(self, logits, labels, is_token_level):
        """Compute classification loss using registry."""
        loss_config = self.train_config.get("loss", {})
        loss_type = loss_config.get("type", "cross_entropy")

        # Reshape for token-level
        if is_token_level:
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)

        loss_kwargs = {k: v for k, v in loss_config.items() if k != "type"}
        loss_kwargs["ignore_index"] = -100 if is_token_level else -100

        if not hasattr(self, "_loss_fn") or self._loss_type != loss_type:
            self._loss_fn = self.cache.registry.create(
                "loss_fn", loss_type, **loss_kwargs
            )
            self._loss_type = loss_type

        loss = self._loss_fn(logits, labels)

        return loss

    def _compute_accuracy(self, logits, labels, is_token_level):
        """Compute classification accuracy."""
        with torch.no_grad():
            preds = logits.argmax(dim=-1)

            if is_token_level:
                mask = labels != -100
                acc = ((preds == labels) & mask).sum().float() / mask.sum().clamp(min=1)
            else:
                acc = (preds == labels).float().mean()

        return acc

    def _pretraining_step(self, batch, batch_idx=None):
        try:
            input_sequence = batch[
                "sequence"
            ]  # Arriva la sequenza già tokenizzata dal MatformerDataModule
            # Un modo sporco per testare cosa succede se arrivano tutte sequenze lunghe
            test_memory = False
            if test_memory:
                if isinstance(input_sequence, UnpaddedTensor):
                    max_len = self.config.max_position_embeddings
                    batch_size = input_sequence.batch_size
                    total_tokens = max_len * batch_size
                    input_sequence = replace(
                        input_sequence,
                        tensor=torch.randint(
                            0,
                            self.config.vocab_size,
                            (total_tokens,),
                            device=input_sequence.tensor.device,
                        ),
                        cu_seqlens=torch.arange(
                            0,
                            total_tokens + 1,
                            max_len,
                            dtype=torch.int32,
                            device=input_sequence.tensor.device,
                        ),
                        max_seq_len=max_len,
                        indices=torch.arange(
                            total_tokens, device=input_sequence.tensor.device
                        ),
                        original_seq_len=max_len,
                    )
            batch["sequence"] = input_sequence
            if batch["worker_has_finished"]:
                zero_loss = (
                    sum(p.sum() for p in self.parameters()) * 0.0
                )  # Questa roba è da riguardare attentamente!!!
                return zero_loss
            masked = True if self.config.training_objective == "masked" else False
            if self.config.training_objective == "crazy":
                self.crazy_previous_state = not getattr(
                    self, "crazy_previous_state", False
                )
                masked = self.crazy_previous_state
                for m in self.model.modules():
                    if hasattr(m, "is_causal"):
                        m.is_causal = not masked
                    if hasattr(m, "attn_kernel") and hasattr(
                        m.attn_kernel, "is_causal"
                    ):
                        m.attn_kernel.is_causal = not masked

            # input_sequence=sequence
            if masked:
                # If masking rate is variable and variable rate is per document, we need to be sure that the tensor has batch dimension
                # if isinstance(sequence,UnpaddedTensor):
                #    repad=True
                #    sequence=sequence.pad()
                # else:
                #    repad=False
                # masked_tokens,cloze_mask,masking_ratio=self.maskerator(sequence.tensor)
                # input_sequence=replace(sequence,tensor=masked_tokens,cloze_mask=cloze_mask)
                # if repad:
                #    input_sequence=input_sequence.unpad()
                input_sequence, masking_ratio = self.maskerator(input_sequence)
                original_sequence = batch["sequence"]
            if self.config.loss_type == "fused":
                model_return_type = "hidden"
                flattening_dimension = self.config.hidden_size
                loss_kwargs = {"lm_head_weight": self.model.lm_head.module.inner.weight}
                if hasattr(self.model.lm_head.module.inner, "bias"):
                    loss_kwargs["lm_head_bias"] = (
                        self.model.lm_head.module.inner.bias
                    )  # TODO: Better way to access inner attributes of wrapped modules
            else:  # Normal loss
                model_return_type = "logits"
                flattening_dimension = self.config.vocab_size
                loss_kwargs = {}

            ### Input al modello ###
            model_output = self(
                input_sequence, return_type=model_return_type
            )  # Return type can be 'logits' or 'hidden' (required for fused loss)
            is_unpadded = isinstance(model_output, UnpaddedTensor)

            if is_unpadded:
                model_output_flat = model_output.tensor
                targets_flat = original_sequence.tensor
                # If already unpadded, all tokens are valid
                base_mask = torch.ones_like(targets_flat, dtype=torch.bool)
                cloze_mask_flat = input_sequence.cloze_mask if masked else None
            else:
                # (B, S, H) -> 2D (B*S, H)
                model_output_flat = model_output.tensor.view(
                    -1, model_output.tensor.size(-1)
                )
                # (B, S) -> 1D (B*S)
                targets_flat = original_sequence.tensor.view(-1)
                base_mask = targets_flat != self.config.pad_token_id
                cloze_mask_flat = input_sequence.cloze_mask.view(-1) if masked else None

            # 2. Setting the training objective
            if masked:
                mask = cloze_mask_flat & base_mask
                inputs = model_output_flat[mask]
                targets = targets_flat[mask]
            else:  # Autoregressive
                mask = base_mask[1:]
                inputs = model_output_flat[:-1][mask]
                targets = targets_flat[1:][mask]
            # 4. Getting the loss
            loss = self.cross_entropy_loss(inputs, targets, **loss_kwargs)
            self.log("train/loss", loss, prog_bar=True, batch_size=self.batch_size)
            if "aux_losses" in self.cache.storage:
                aux_losses = self.cache.storage["aux_losses"]
                if aux_losses:
                    total_aux_loss = torch.stack(aux_losses).sum()
                    aux_weight = 0.1
                    loss += aux_weight * total_aux_loss
                    self.log(
                        "train/aux_memory_loss", total_aux_loss.item(), on_step=True
                    )
                    self.log("train/total_loss", loss, batch_size=self.batch_size)
                self.cache.storage["aux_losses"] = []

            if self.config.training_objective == "crazy":
                self.log(
                    f"train/loss_masked_{str(masked)}",
                    loss,
                    prog_bar=True,
                    batch_size=self.batch_size,
                )
            try:
                current_lr = self.lr_schedulers().get_last_lr()[0]
                self.log(
                    "lr",
                    current_lr,
                    prog_bar=True,
                    on_step=True,
                    on_epoch=False,
                    batch_size=self.batch_size,
                )
            except:
                pass
            if masked:  # Logging also the accuracy
                preds = model_output_flat[mask].argmax(dim=-1)
                targets = targets_flat[mask]
                acc = (preds == targets).float().mean()
                self.log(
                    "train/accuracy",
                    acc,
                    prog_bar=True,
                    on_step=True,
                    on_epoch=True,
                    batch_size=self.batch_size,
                )
                # self.log("train/masking_rate",masking_ratio,prog_bar=False,on_step=True,on_epoch=False,batch_size=self.batch_size)

            # TODO: this part has to be revised and cleaned
            additional_metrics = False
            if additional_metrics:
                if self.global_step % 100 == 0:
                    with torch.no_grad():
                        if self.cache.additional_logs:
                            keys = list(self.cache.additional_logs.keys())
                            vals = [
                                (
                                    v
                                    if torch.is_tensor(v)
                                    else torch.tensor(v, device=self.device)
                                )
                                for v in self.cache.additional_logs.values()
                            ]
                            vals_cpu = torch.stack(vals).detach().cpu().float().tolist()
                            log_dict = dict(zip(keys, vals_cpu))
                            self.log_dict(
                                log_dict, on_step=True, batch_size=self.batch_size
                            )
                            self.cache.additional_logs.clear()
                        # 'inputs' is already flattened and filtered for the loss (works for GPT/BERT/Unpadded)
                        # We sample the first 1024 tokens to keep the compute cost negligible
                        sample_size = min(inputs.size(0), 1024)
                        diag_data = inputs[:sample_size]

                        if self.config.loss_type != "fused":
                            # 1. Logit Statistics (Saturation check)
                            self.log(
                                "health/logits_max",
                                diag_data.max(),
                                on_step=True,
                                batch_size=self.batch_size,
                            )
                            self.log(
                                "health/logits_std",
                                diag_data.std(),
                                on_step=True,
                                batch_size=self.batch_size,
                            )

                            # 2. Entropy (Confidence check)
                            probs = torch.softmax(diag_data, dim=-1)
                            entropy = -torch.sum(
                                probs * torch.log(probs + 1e-9), dim=-1
                            ).mean()
                            self.log(
                                "health/entropy",
                                entropy,
                                on_step=True,
                                batch_size=self.batch_size,
                            )
                        else:
                            # If fused, inputs are actually hidden states
                            self.log(
                                "health/hidden_max",
                                diag_data.max(),
                                on_step=True,
                                batch_size=self.batch_size,
                            )
                            self.log(
                                "health/hidden_std",
                                diag_data.std(),
                                on_step=True,
                                batch_size=self.batch_size,
                            )

                        # 3. Gating Health (if the model has gates)
                        for name, p in self.named_parameters():
                            if "gate" in name and p.numel() == 1:
                                self.log(
                                    f"gates/{name}_val",
                                    p.item(),
                                    on_step=True,
                                    batch_size=self.batch_size,
                                )
            return loss
            """
            This exception must be only for extreme cases. To avoid breaking a large training if only a minor issue occurs,
            ex a particularly long batch, the batch is skipped. The event will be logged. Be sure that this number is limited
            to a very small amount of steps.
            """
        except Exception as e:
            print("------------------  ALARM ------------------")
            if not any(
                s in str(e).lower()
                for s in ("out of memory", "cuda out of memory", "cublas")
            ):
                print("CAUGHT EXCEPTION: ")
                print(e)
                print(
                    "Training will try to continue but this may be unreliable. Careful check what is happening"
                )
                self.log(
                    "train/EXCEPTIONS",
                    1,
                    on_step=True,
                    on_epoch=False,
                    batch_size=self.batch_size,
                )
            else:
                print("OUT OF MEMORY ERROR")
                print(
                    "Training is continuing, but be sure that this happens very seldom or reduce the batch size."
                )
                self.log(
                    "train/skipped_oom",
                    1,
                    on_step=True,
                    on_epoch=False,
                    batch_size=self.batch_size,
                )
            print(
                "All the gradients from the current batch and all the previous accumulated gradients were discarded."
            )
            try:
                opts = self.optimizers()
                for o in opts if isinstance(opts, list) else (opts,):
                    o.zero_grad(set_to_none=True)
            except Exception:
                pass

            if hasattr(self.cache, "storage"):
                self.cache.storage.clear()

            import gc

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            return sum(p.sum() for p in self.parameters()) * 0.0

    def on_before_optimizer_step(self, optimizer):
        additional_metrics = False
        if additional_metrics:
            if self.global_step % 100 == 0:
                grad_norm_sq = 0.0
                weight_norm_sq = 0.0

                for p in self.parameters():
                    if p.grad is not None:
                        # Use .item() to move scalars to CPU immediately, preventing GPU sync overhead later
                        grad_norm_sq += p.grad.detach().norm(2).item() ** 2
                        weight_norm_sq += p.detach().norm(2).item() ** 2

                total_grad_norm = grad_norm_sq**0.5
                total_weight_norm = weight_norm_sq**0.5

                self.log(
                    "health/total_grad_norm",
                    total_grad_norm,
                    on_step=True,
                    batch_size=self.batch_size,
                )

                # 4. Update-to-Weight Ratio (The "Karpathy Metric")
                try:
                    lr = optimizer.param_groups[0]["lr"]
                    if total_weight_norm > 1e-8:
                        ratio = (lr * total_grad_norm) / total_weight_norm
                        self.log(
                            "health/update_weight_ratio",
                            ratio,
                            on_step=True,
                            batch_size=self.batch_size,
                        )
                except Exception:
                    pass

    def configure_optimizers(self):
        if (
            getattr(self.train_config, "no_decay_for_embedding", False)
            and self.train_config["optimizer"] != "muon"
        ):
            raise ValueError(
                "no_decay_for_embedding for optimizers different than Muon not implemented yet! (Altough it's easy). Please remove it "
            )

        if self.train_config["optimizer"] == "muonclip":
            from muon import MuonClip, MuonConfig

            base_lr = self.train_config["lr"]

            from types import SimpleNamespace

            model_config = SimpleNamespace(
                num_key_value_heads=self.config.num_attention_heads,
                num_attention_heads=self.config.num_attention_heads,
                head_dim=self.config.hidden_size // self.config.num_attention_heads,
            )

            muon_lr = (
                base_lr
                * 0.2
                * math.sqrt(
                    max(
                        p.shape[0]
                        for p in self.parameters()
                        if p.ndim >= 2 and p.requires_grad
                    )
                )
            )

            muon_config = MuonConfig(
                lr=base_lr,  ###To check!!!
                muon_beta=getattr(self.train_config, "muon_momentum", 0.95),
                muon_decay=getattr(self.train_config, "weight_decay", 0.01),
                ns_steps=getattr(self.train_config, "muon_ns_steps", 5),
                adam_betas=getattr(self.train_config, "betas", (0.9, 0.95)),
                adam_decay=getattr(self.train_config, "weight_decay", 0.01),
                adam_eps=getattr(self.train_config, "eps", 1e-10),
                enable_clipping=True,
                clipping_layers_mapping={
                    "q_proj": "packed_proj",
                    "k_proj": "packed_proj",
                },
                clipping_threshold=getattr(self.train_config, "clip_threshold", 50.0),
                clipping_alpha=getattr(self.train_config, "clip_alpha", 0.5),
                log_max_logits=False,
                cans_ortho=False,
                estimate_lower_bound=False,
            )

            optimizer = MuonClip(self, model_config, muon_config)

        elif self.train_config["optimizer"] == "muonflash":
            muon_params = []
            adamw_params = []
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if "conv" in name:
                    adamw_params.append(param)
                    # print(f"{name} (Convolutional) in AdamW (ndim={param.ndim})")
                elif "lm_head" in name or "embed_tokens" in name or param.ndim < 2:
                    adamw_params.append(param)
                    # print(f"{name} in AdamW (ndim={param.ndim})")
                else:
                    muon_params.append(param)
                    # print(f"{name} in Muon")

            base_lr = self.train_config["lr"]

            from flash_muon import Muon as FlashMuon

            muon_lr = (
                base_lr * 0.2 * math.sqrt(max(muon_params[0].shape[:2]))
                if muon_params
                else base_lr
            )
            optimizer = [
                FlashMuon(
                    muon_params,
                    lr=muon_lr,
                    momentum=getattr(self.train_config, "muon_momentum", 0.95),
                    rank=0,
                    world_size=1,
                ),
                torch.optim.AdamW(
                    adamw_params,
                    lr=base_lr,
                    betas=getattr(self.train_config, "betas", (0.9, 0.95)),
                    weight_decay=getattr(self.train_config, "weight_decay", 0.01),
                ),
            ]

        elif self.train_config["optimizer"] == "muon":
            muon_params = []
            adamw_params = []
            no_decay_params = []
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if "conv" in name:
                    adamw_params.append(param)
                    # print(f"{name} (Convolutional) in AdamW (ndim={param.ndim})")
                elif "lm_head" in name or "embed_tokens" in name or param.ndim < 2:
                    adamw_params.append(param)
                    # print(f"{name} in AdamW (ndim={param.ndim})")
                else:
                    muon_params.append(param)
                    # print(f"{name} in Muon")
                if getattr(self.train_config, "no_decay_for_embedding", False) and (
                    "lm_head" in name or "embed_tokens" in name
                ):
                    no_decay_params.append(param)
                    # print(f"{name} has weight decay disabled.")

            base_lr = self.train_config["lr"]

            optimizer = Muon(
                lr=base_lr,
                wd=getattr(self.train_config, "weight_decay", 0.01),
                muon_params=muon_params,
                momentum=getattr(self.train_config, "muon_momentum", 0.95),
                nesterov=getattr(self.train_config, "muon_nesterov", True),
                ns_steps=getattr(self.train_config, "muon_ns_steps", 5),
                adamw_params=adamw_params,
                adamw_betas=getattr(self.train_config, "betas", (0.9, 0.95)),
                adamw_eps=getattr(self.train_config, "eps", 1e-10),
                no_decay_params=no_decay_params,
            )

        elif self.train_config["optimizer"] == "adamw":
            # Filter only trainable parameters
            trainable_params = [p for p in self.parameters() if p.requires_grad]

            if len(trainable_params) == 0:
                raise ValueError("No trainable parameters found!")

            optimizer = AdamW(
                trainable_params,
                lr=self.train_config["lr"],
                weight_decay=self.train_config.get("weight_decay", 0.01),
            )

            print(
                f"Optimizer initialized with {sum(p.numel() for p in trainable_params):,} trainable params"
            )
        elif self.train_config["optimizer"] == "adam":
            optimizer = Adam(
                self.parameters(),
                lr=self.train_config["lr"],
                weight_decay=getattr(self.train_config, "weight_decay", 0.01),
            )

        # === Scheduler ===
        if not getattr(self.train_config, "lr_scheduling", False):
            return optimizer

        total_steps = getattr(self.train_config, "total_steps")

        self.total_training_steps = total_steps

        def create_scheduler(opt):
            if getattr(self.train_config, "scheduler") == "custom":
                warmup = int(
                    getattr(self.train_config, "warmup_steps", 0.05) * total_steps
                )
                hold = int(getattr(self.train_config, "hold_steps", 0.10) * total_steps)
                target = getattr(self.train_config, "final_lr", 0.0)
                base_lr = opt.param_groups[0]["lr"]
                factor = target / base_lr if base_lr > 0 else 0.0

                def lr_schedule(step):
                    if step < warmup:
                        return step / max(1, warmup)
                    if step < warmup + hold:
                        return 1.0
                    prog = (step - warmup - hold) / max(1, total_steps - warmup - hold)
                    return factor + (1 - factor) * 0.5 * (1 + math.cos(math.pi * prog))

                return torch.optim.lr_scheduler.LambdaLR(
                    opt, lr_schedule, last_epoch=-1
                )

            elif getattr(self.train_config, "scheduler") == "cosine_decay":
                from transformers import get_cosine_schedule_with_warmup

                warmup = int(
                    getattr(self.train_config, "warmup_steps", 0.05) * total_steps
                )
                return get_cosine_schedule_with_warmup(
                    optimizer=opt,
                    num_warmup_steps=warmup,
                    num_training_steps=total_steps,
                )

            elif getattr(self.train_config, "scheduler") == "linear_decay":
                from transformers import get_linear_schedule_with_warmup

                warmup = int(
                    getattr(self.train_config, "warmup_steps", 0.05) * total_steps
                )
                return get_linear_schedule_with_warmup(
                    optimizer=opt,
                    num_warmup_steps=warmup,
                    num_training_steps=total_steps,
                )

            else:
                warmup = int(
                    getattr(self.train_config, "warmup_steps", 0.05) * total_steps
                )
                return get_scheduler(
                    name=getattr(self.train_config, "scheduler", "linear"),
                    optimizer=opt,
                    num_warmup_steps=warmup,
                    num_training_steps=total_steps,
                )

        if isinstance(optimizer, list):
            scheduler = [create_scheduler(opt) for opt in optimizer]
        else:
            scheduler = create_scheduler(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    @staticmethod
    def load_from_checkpoint(
        checkpoint_path,
        ModelClass,
        config=None,
        train_config=None,
        map_location=None,
        tokenizer=None,
        overrides=None,
        varlen_strategy="padding",
        external_mapping=None,
        **model_kwargs,
    ):
        checkpoint = torch.load(
            checkpoint_path, map_location=map_location, weights_only=False
        )

        if config is None:
            if (
                "hyper_parameters" in checkpoint
                and "config" in checkpoint["hyper_parameters"]
            ):
                config = checkpoint["hyper_parameters"]["config"]
            else:
                raise ValueError(
                    "Config not found in checkpoint and not provided. Please provide a config."
                )

        
        #tokenizer = (
        #    AutoTokenizer.from_pretrained(tokenizer) 
        #    if tokenizer != 'bytes' else 'bytes'
        #)
        
        if overrides is not None:
            for k, v in overrides.items():
                setattr(config, k, v)

        
        if type(tokenizer) != MatformerTokenizer:
          tokenizer = MatformerTokenizer(
              config=config,
              tokenizer_type='huggingface',
              tokenizer=tokenizer,
              tokenizer_name=tokenizer,
              varlen_strategy=varlen_strategy,
          )
        
        if tokenizer is not None:
          assert isinstance(tokenizer,MatformerTokenizer)
        model = PL_ModelWrapper(
            ModelClass=ModelClass,
            config=config,
            train_config=train_config,
            tokenizer=tokenizer,
            device=map_location,
            init_weights=False,
            **model_kwargs,
        )
        # model.load_state_dict(checkpoint['state_dict'])
        model.load_stable_state_dict(
            checkpoint["state_dict"], strict=False, external_mapping=external_mapping
        )
        # print("Found this config:")
        # print(config)
        return model, config
