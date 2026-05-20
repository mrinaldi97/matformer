#matformer/training/training_loop.py
from matformer.models import PL_ModelWrapper
from matformer.training.training_hooks import TrainingHooks
def training_loop(fabric, training_dict):
    model        = training_dict['model']
    optimizers   = training_dict['optimizer']
    schedulers   = training_dict['lr_scheduler']
    datamodule   = training_dict['datamodule']
    dataloader   = training_dict['dataloader']
    train_config = training_dict['train_config']
    ckpt_manager = training_dict['ckpt_manager']
    state = {"epoch": training_dict['epoch'], "step": training_dict['step']}
    hooks = training_dict.get('hooks', TrainingHooks())
    val_every = train_config.get("validate_every_n_steps", -1)
    step_ref    = training_dict['step_ref']
    matformer_logger  = training_dict['logger']
    for epoch in range(train_config.get("max_epochs",1)):
        for batch_idx, batch in enumerate(dataloader):
            is_accumulating = (batch_idx + 1) % train_config.get("accumulate_grad_batches",1) != 0

            with fabric.no_backward_sync(model, enabled=is_accumulating):
                loss = model.training_step(batch, batch_idx)
                fabric.backward(loss / train_config.get("accumulate_grad_batches",1))

            if not is_accumulating:
                fabric.clip_gradients(model, optimizers[0], max_norm=train_config.get("gradient_clip_val", 1.0))
                if hooks.on_before_optimizer: hooks.on_before_optimizer(model, optimizers)
                for optimizer in optimizers: optimizer.step()
                if hooks.on_after_optimizer:  hooks.on_after_optimizer(model, optimizers)
                if schedulers:
                    for i,scheduler in enumerate(schedulers): scheduler.step()
                    # Log the LR
                    matformer_logger.log(f"lr{i}", scheduler.get_last_lr()[0],step=state["step"],prog_bar=True, on_step=True, on_epoch=False)
                for optimizer in optimizers: optimizer.zero_grad()
                state["step"] += 1
                step_ref[0] = state["step"]
                matformer_logger.advance()
                if hooks.on_step_end: hooks.on_step_end(model, state["step"], loss)
                for every_n, fn in hooks.periodic.items():
                    if state["step"] % every_n == 0: fn(model, state["step"])
                if hooks.on_validation and val_every > 0 and state["step"] % val_every == 0:
                    hooks.on_validation(model, state["step"])
                if train_config.get("save_every_n_steps", -1) > 0 and state["step"] % train_config.get("save_every_n_steps", -1) == 0:
                    PL_ModelWrapper.save_checkpoint(
                            model=model, optimizer=optimizers, scheduler=schedulers,
                            step=state["step"], epoch=state["epoch"],
                            datamodule=datamodule, ckpt_manager=ckpt_manager, fabric=fabric
                        )

        state["epoch"] += 1
        if hooks.on_epoch_end: hooks.on_epoch_end(model, state["epoch"])
    # training finished


def launch_training_loop(fabric, training_dict, save_at_end=True):
    # (before training hooks)
    training_loop(fabric, training_dict)
    if save_at_end:
        PL_ModelWrapper.save_checkpoint(
                model=training_dict['model'], optimizer=training_dict['optimizer'],
                scheduler=training_dict['lr_scheduler'], step=training_dict['step'],
                epoch=training_dict['epoch'], datamodule=training_dict['datamodule'],
                ckpt_manager=training_dict['ckpt_manager'], fabric=fabric
            )
    # (after training hooks)
            
