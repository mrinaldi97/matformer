   def training_step_curriculum(self, batch, batch_idx):
        self.saveit = False
        self.log("train/curriculum_step", self.curriculumStep)
        
        max_len = self.until_convergence if self.until_convergence else (
            self.encoder_config.max_position_embeddings if self.curriculumStep == 0 else self.curriculumStep
        )
        self.log("train/max_seq_len", max_len)
        
        # generate random data up to max_len to avoid catastrophic forgetting
        input_ids, sequence_lengths = generate_random_batch(
            max_len=max_len,
            pad_token=self.encoder_config.pad_token_id,
            batch_size=self.train_config['batch_size'],
            device=self.device
        )
        
        char_logits, seqlen_logits = self(input_ids, sequence_lengths)
        
        # seq len loss
        seqlen_targets = (sequence_lengths.tensor.squeeze(-1) - 1).long()
        seqlen_loss = nn.functional.cross_entropy(seqlen_logits.tensor, seqlen_targets)
        self.log("train/seq_len", seqlen_loss)
        
        # char reconstruction loss
        char_targets = input_ids.tensor.view(-1).long()
        char_logits_flat = char_logits.tensor.view(-1, char_logits.tensor.size(-1))
        char_loss = nn.functional.cross_entropy(char_logits_flat, char_targets, ignore_index=self.encoder_config.pad_token_id)
        self.log("train/char_loss", char_loss)
        
        # EDITED BY LLM: global curriculum max steps early exit
        if self.global_step > self.max_curriculum_steps:
            print("Curriculum max steps reached. Exiting.")
            sys.exit(0)
        
        # EDITED BY LLM: periodic save
        if self.global_step % self.train_config.get("save_every_n_steps", 5000) == 0:
            self.saveit = True
        
        # curriculum progression thresholds
        if seqlen_loss < 0.02 and self.curriculumStep == 0:
            self.patience = self.patience + 1 if self.patience <= self.curriculum_patience else 0
            if self.patience == 0:
                self.curriculumStep = 2
        
        if self.curriculumStep >= self.curriculumTarget:
            sys.exit(0)
        
        if char_loss < 0.04:
            threshold_patience = self.curriculum_patience if self.curriculumStep < 27 else self.curriculum_patience * 4
            self.patience += 1
            if self.patience > threshold_patience:
                self.curriculumStep += 5 if self.curriculumStep < 27 else 1
                self.saveit = True
                self.patience = 0
        
        # compute combined loss
        loss = (1.4 * seqlen_loss + 0.7 * char_loss) if self.curriculumStep == 0 else (2 * seqlen_loss + char_loss)
        self.log("train/loss", loss)
        self.log("train/seqlen_loss",seqlen_loss)
        self.log("train/char_loss",char_loss)
        print(f"Step {batch_idx}: loss={loss.item():.4f}, seqlen={seqlen_loss.item():.4f}, char={char_loss.item():.4f} Curriculum: {self.curriculumStep}")
        
        # EDITED BY LLM: until-convergence behavior
        if self.until_convergence:
            if not self.converged:
                self.patience = self.patience + 1 if char_loss.item() < self.target_loss else 0
                if self.patience >= self.curriculum_patience:
                    print(f"Converged at length {self.until_convergence}, entering post-convergence phase.")
                    self.converged = True
            else:
                self.post_convergence_counter += 1
                if self.post_convergence_counter >= self.post_convergence_steps:
                    ckpt_path = f"convergence-{self.until_convergence}.ckpt"
                    print(f"Saving converged model: {ckpt_path}")
                    self.trainer.save_checkpoint(ckpt_path)
                    sys.exit(0)
        
        return loss
