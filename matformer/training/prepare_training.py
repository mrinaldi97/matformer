#matformer/training/prepare_training.py
from matformer.training.utils import detect_device,get_model_class
from matformer.training.ckpt_manager import CheckpointDirectoryManager
import math, torch
from datetime import datetime
from lightning.fabric import Fabric
from lightning.pytorch.loggers import WandbLogger
from lightning.fabric.strategies import DDPStrategy
from matformer.models import PL_ModelWrapper
from matformer.data.matformer_data_module import MatformerDataModule
from matformer.training.optimizers import configure_optimizers
from matformer.training.lr_scheduler import configure_lr_schedulers
from matformer.training.logger import MatformerLogger, WandbAdapter, TQDMBar

def decide_checkpoint(save_dir, model_name, version,
                      checkpoint=None, start_from_scratch=False,
                      start_from_step=None, print_info=True, fabric=None):
    ckpt_manager = CheckpointDirectoryManager(save_dir, model_name, version)
    if checkpoint:
        # A specific path is given                               
        load_path = checkpoint
        ckpt_manager.start_from_scratch()  #Advances the version automatically                       
    elif start_from_scratch:
        load_path, step = ckpt_manager.start_from_scratch()        
    elif start_from_step:
        load_path, step = ckpt_manager.restart_from_step(start_from_step)
    else:
        load_path, step = ckpt_manager.resume() #Starts from scratch if ckpt not existing, if not resume from last version
    if fabric and print_info:
        fabric.print("======== STARTING TRAINING ========")
        if checkpoint is None:
            if load_path is None:
                fabric.print("Starting training from scratch")
            else:
                fabric.print(f"Restarting training from {load_path} at step {step}")               
        else:
            fabric.print(f"Starting training from explicit path: {load_path}")
        fabric.print("===================================")    
    return load_path,ckpt_manager

def count_training_steps(datamodule,batch_size=1,accumulate_grad_batches=1,num_nodes=1,max_steps=None):
    """
    Return the training steps count for each epoch
    Expects the datamodule to return the *per rank* length (already divided by rank size)
    Will divide the training steps by the number of nodes 
    """
    if hasattr(datamodule, '__len__'):    
        per_rank_data_len=len(datamodule)
        if per_rank_data_len and per_rank_data_len > 0:
            num_batches = math.ceil(per_rank_data_len / batch_size)
            total_steps = (num_batches // accumulate_grad_batches) // num_nodes # Dividi anche per i nodi
            if max_steps:
                total_steps = max_steps
            return total_steps,num_batches
        else:
            return None, None
    else:
        return None, None



def instantiate_training(model_config, train_config, data_config, tokenizer_config, raw_config,
                         device='auto', precision='bf16-mixed', num_nodes=1, device_count=1,
                         checkpoint=None, start_from_scratch=False,
                         what_to_reset=None, max_steps=None, version=1, _compile=False):
    
    # 1. Choose the device
    if device == 'auto':
        accelerator, device_string=detect_device()
    elif device=='cuda':
        assert torch.cuda.is_available()
        accelerator='gpu'
        device_string='cuda'
    else:
        accelerator=device_string=device 
           
    # 2. Launch Fabric
    fabric = Fabric(
        accelerator=accelerator, devices=device_count, num_nodes=num_nodes,
        precision=precision,
        strategy=DDPStrategy(gradient_as_bucket_view=False, find_unused_parameters=False),
        loggers=WandbLogger(
            name=f"{raw_config.get('wandb_run_name','matformer')}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            project=raw_config.get('wandb_project', 'matformer'),
            config=raw_config
        )
    )
    fabric.launch() # From now on, everything belongs to all ranks
    fabric.seed_everything(train_config.get('seed', 27))
    
    # 3. Instantiate model
    ModelClass = get_model_class(raw_config['model_class'])
    model = PL_ModelWrapper(
        ModelClass,
        config=model_config,
        tokenizer=None,
        train_config=train_config,
        device=device_string,
        batch_size=data_config['batch_size'],
        reset=what_to_reset
    )    
    if _compile:
        model=compile_model(model)

    # 4. Configure datamodule
    datamodule = MatformerDataModule(
        mdat_path=data_config['data_root'],
        iteration_modality=data_config['wanted_from_strategy'], 
        pad_token_id=model_config.pad_token_id,
        varlen_strategy=tokenizer_config['varlen_strategy'],
        mdat_strategy=data_config['mdat_strategy'],
        mdat_view=data_config['mdat_view'],
        with_meta=False,
        max_seq_len=model_config.max_position_embeddings,
        batch_size=data_config['batch_size'],
        num_devices=device_count
    )   
    datamodule.setup() # It's done here to correctly count the total training steps
    # 5. Compute the number of training steps
    total_steps,total_batches=count_training_steps(
        datamodule,
        batch_size=data_config['batch_size'],
        accumulate_grad_batches=train_config.get('accumulate_grad_batches', 1),
        num_nodes=num_nodes,
        max_steps=max_steps
    )
    train_config["total_steps"]=total_steps
    train_config["num_batches"] = total_batches
    model.train_config=train_config
    if total_steps is None:
        fabric.print("The Datamodule is not returning the length. Thus, LR scheduling is disabled")
        train_config["lr_scheduling"] = False      
        
    # 6. Configure optimizer(s)
    optimizers=configure_optimizers(model,train_config) #(!) According to torch doc, it's important to initialize scheduler before optimizer to avoid losing states
    # 7. Configure scheduler(s)
    schedulers=configure_lr_schedulers(train_config, optimizers)


            
    # 7. Restore states
    load_path, ckpt_manager = decide_checkpoint(
        save_dir=raw_config.get('save_dir', './checkpoints'),
        model_name=train_config.get('checkpoint_name','matformer_model'),
        version=version,
        checkpoint=checkpoint,
        start_from_scratch=start_from_scratch, fabric=fabric
    )
    if load_path is not None:
        checkpoint_result = model.load_checkpoint(
                ckpt_path=load_path,
                map_location=device_string,
                what_to_reset=what_to_reset,
                optimizer=optimizers,
                scheduler=schedulers,
                datamodule=datamodule
            )
    # 8. Setup logger
    step_ref = [checkpoint_result.get("step", 0) if load_path else 0]
    matformer_logger = MatformerLogger()
    matformer_logger.add(WandbAdapter(fabric.loggers[0]))
    if total_steps:
        matformer_logger.add(TQDMBar(total=total_steps, initial=step_ref[0]))
    model.setup_logging(matformer_logger, step_ref)    
    # 9. Fabric setup
    train_dataloader = fabric.setup_dataloaders(datamodule.train_dataloader())
    model, *optimizers = fabric.setup(model, *optimizers)
        
    return fabric, {
        "model":        model,
        "optimizer":    optimizers,
        "lr_scheduler": schedulers,
        "datamodule":   datamodule,
        "dataloader":   train_dataloader,
        "ckpt_manager": ckpt_manager,
        "train_config": train_config,
        "logger": matformer_logger,
        "step_ref": step_ref,
        "epoch":        checkpoint_result.get("epoch", 0) if load_path else 0,
        "step":         checkpoint_result.get("step",  0) if load_path else 0
    }
    

    
    
