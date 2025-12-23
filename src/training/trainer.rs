/// Training loop for TinyRecursiveModel
use candle_core::{Result, Tensor, Device, DType};
use candle_nn::{VarMap, VarBuilder, AdamW, ParamsAdamW, Optimizer, loss, ops};
use std::path::Path;

use crate::{TinyRecursiveModel, TRMConfig};
use crate::data::BatchDataLoader;
use crate::models::InnerCarry;
use super::scheduler::CosineScheduler;
use super::ema::{EMA, EMAConfig};
use super::checkpoint::{Checkpoint, CheckpointMetadata};

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of training epochs
    pub num_epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Learning rate (initial)
    pub learning_rate: f64,
    /// Minimum learning rate
    pub lr_min: f64,
    /// Warmup steps
    pub warmup_steps: usize,
    /// Total training steps (for scheduler)
    pub total_steps: usize,
    /// Weight decay
    pub weight_decay: f64,
    /// Gradient clipping value
    pub grad_clip: Option<f64>,
    /// EMA decay
    pub ema_decay: f64,
    /// Save checkpoint every N steps
    pub save_every: usize,
    /// Evaluation every N steps
    pub eval_every: usize,
    /// Checkpoint directory
    pub checkpoint_dir: String,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            num_epochs: 10,
            batch_size: 32,
            learning_rate: 3e-4,
            lr_min: 3e-5,
            warmup_steps: 1000,
            total_steps: 100000,
            weight_decay: 0.1,
            grad_clip: Some(1.0),
            ema_decay: 0.9999,
            save_every: 1000,
            eval_every: 500,
            checkpoint_dir: "checkpoints".to_string(),
        }
    }
}

/// Trainer for TinyRecursiveModel
pub struct Trainer {
    model: TinyRecursiveModel,
    model_config: TRMConfig,
    varmap: VarMap,
    optimizer: AdamW,
    scheduler: CosineScheduler,
    ema: Option<EMA>,
    config: TrainingConfig,
    device: Device,
    step: usize,
}

impl Trainer {
    /// Create new trainer
    pub fn new(
        model_config: TRMConfig,
        training_config: TrainingConfig,
        device: Device,
    ) -> Result<Self> {
        // Create model with F64 to avoid dtype issues
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &device);
        let model = TinyRecursiveModel::new(model_config.clone(), vb)
            .map_err(|e| candle_core::Error::Msg(format!("Model init failed: {:?}", e)))?;

        // Create optimizer using candle's built-in AdamW
        let optimizer_params = ParamsAdamW {
            lr: training_config.learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: training_config.weight_decay,
        };
        let optimizer = AdamW::new(varmap.all_vars(), optimizer_params)?;

        // Create scheduler
        let scheduler = CosineScheduler::new(super::scheduler::CosineSchedulerConfig {
            lr_init: training_config.learning_rate,
            lr_min: training_config.lr_min,
            warmup_steps: training_config.warmup_steps,
            total_steps: training_config.total_steps,
        });

        // Create EMA
        let ema_config = EMAConfig {
            decay: training_config.ema_decay,
        };
        let ema = Some(EMA::new(ema_config));

        Ok(Self {
            model,
            model_config,
            varmap,
            optimizer,
            scheduler,
            ema,
            config: training_config,
            device,
            step: 0,
        })
    }

    /// Compute loss for a batch
    fn compute_loss(
        &self,
        logits: &Tensor,
        targets: &Tensor,
    ) -> Result<Tensor> {
        // For opcode classification:
        // logits shape: [batch, seq_len, num_classes]
        // targets shape: [batch, 1] or [batch]

        // If targets are for sequence modeling (seq_len > 1), use all positions
        // If targets are classification labels (seq_len = 1 or single label), use pooled representation

        let batch_size = logits.dim(0)?;
        let seq_len = logits.dim(1)?;
        let num_classes = logits.dim(2)?;

        let target_shape = targets.dims();

        // Check if this is classification (single label per example) or sequence modeling
        if target_shape.len() == 2 && target_shape[1] == 1 {
            // Classification task: targets shape [batch, 1]
            // Pool logits across sequence (mean pooling)
            let logits_pooled = logits.mean(1)?; // [batch, num_classes]

            // Flatten targets to [batch]
            let targets_flat = targets.flatten_all()?;

            // Compute log_softmax
            let log_probs = ops::log_softmax(&logits_pooled, candle_core::D::Minus1)?;

            // Gather log probs at target indices and compute negative log likelihood
            let mut loss_sum = 0.0f64;
            for i in 0..batch_size {
                let target_idx = targets_flat.get(i)?.to_scalar::<u32>()? as usize;
                let log_prob = log_probs.get(i)?.get(target_idx)?.to_scalar::<f64>()?;
                loss_sum -= log_prob;
            }

            let loss_val = loss_sum / batch_size as f64;
            Tensor::from_slice(&[loss_val], 1, &self.device)?.to_dtype(DType::F64)?.squeeze(0)
        } else {
            // Sequence modeling task: targets shape [batch, seq_len]
            let logits_flat = logits.reshape((batch_size * seq_len, num_classes))?;
            let targets_flat = targets.flatten_all()?;

            // Compute log_softmax
            let log_probs = ops::log_softmax(&logits_flat, candle_core::D::Minus1)?;

            // Gather log probs at target indices
            let mut loss_sum = 0.0f64;
            for i in 0..(batch_size * seq_len) {
                let target_idx = targets_flat.get(i)?.to_scalar::<u32>()? as usize;
                let log_prob = log_probs.get(i)?.get(target_idx)?.to_scalar::<f64>()?;
                loss_sum -= log_prob;
            }

            let loss_val = loss_sum / (batch_size * seq_len) as f64;
            Tensor::from_slice(&[loss_val], 1, &self.device)?.to_dtype(DType::F64)?.squeeze(0)
        }
    }

    /// Training step
    pub fn train_step(
        &mut self,
        input_ids: &Tensor,
        target_ids: &Tensor,
    ) -> Result<f32> {
        // Get batch size and sequence length
        let batch_size = input_ids.dim(0)?;
        let seq_len = input_ids.dim(1)?;

        log::debug!("Input dtype: {:?}, Target dtype: {:?}", input_ids.dtype(), target_ids.dtype());

        // Create initial carry with F64
        let carry = InnerCarry::empty(
            batch_size,
            seq_len,
            self.model_config.hidden_size,
            DType::F64,
            &self.device,
        )?;

        // Forward pass
        log::debug!("Running forward pass...");
        let (_new_carry, logits) = self.model.forward(&carry, input_ids)
            .map_err(|e| candle_core::Error::Msg(format!("Forward pass failed: {:?}", e)))?;

        log::debug!("Logits shape: {:?}, dtype: {:?}", logits.dims(), logits.dtype());

        // Compute loss
        log::debug!("Computing loss...");
        let loss = self.compute_loss(&logits, target_ids)
            .map_err(|e| candle_core::Error::Msg(format!("Loss computation failed: {:?}", e)))?;
        let loss_val = loss.to_scalar::<f64>()? as f32;

        // Update learning rate before optimizer step
        let lr = self.scheduler.get_lr();
        self.optimizer.set_learning_rate(lr);

        // Backward pass + parameter update (all in one!)
        // This is THE KEY: backward_step() computes gradients AND updates parameters in-place
        self.optimizer.backward_step(&loss)?;

        // Scheduler step
        self.scheduler.step();

        // EMA update
        if let Some(ref mut ema) = self.ema {
            let vars = self.varmap.all_vars();
            let params: Vec<Tensor> = vars.iter().map(|v| v.as_tensor().clone()).collect();
            ema.update(&params)?;
        }

        self.step += 1;

        Ok(loss_val)
    }

    /// Save checkpoint
    pub fn save_checkpoint<P: AsRef<Path>>(&self, path: P, loss: Option<f64>) -> Result<()> {
        use std::collections::HashMap;

        std::fs::create_dir_all(&self.config.checkpoint_dir)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create checkpoint dir: {}", e)))?;

        let metadata = CheckpointMetadata {
            step: self.step,
            lr: self.scheduler.get_lr(),
            loss,
            config: None,
        };

        // Extract tensors from varmap
        let mut tensors = HashMap::new();
        for (name, var) in self.varmap.data().lock().unwrap().iter() {
            tensors.insert(name.clone(), var.as_tensor().clone());
        }

        let checkpoint = Checkpoint::new(tensors, metadata);

        checkpoint.save(path.as_ref())
            .map_err(|e| candle_core::Error::Msg(format!("Failed to save checkpoint: {}", e)))
    }

    /// Train for one epoch
    pub fn train_epoch(&mut self, dataloader: &mut impl BatchDataLoader) -> Result<f32> {
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        dataloader.reset();

        println!("Starting epoch loop...");
        while let Some((input_ids, target_ids)) = dataloader.next_batch(&self.device)? {
            println!("Processing batch {}...", num_batches + 1);
            let loss = self.train_step(&input_ids, &target_ids)?;
            total_loss += loss;
            num_batches += 1;

            println!("Batch {} complete, loss: {:.4}", num_batches, loss);

            if self.step % 100 == 0 {
                log::info!(
                    "Step {}: loss={:.4}, lr={:.6}",
                    self.step,
                    loss,
                    self.scheduler.get_lr()
                );
            }

            // Save checkpoint
            if self.step % self.config.save_every == 0 {
                let checkpoint_path = format!(
                    "{}/checkpoint_step_{}.safetensors",
                    self.config.checkpoint_dir,
                    self.step
                );
                log::info!("Saving checkpoint to {}", checkpoint_path);
                self.save_checkpoint(&checkpoint_path, Some(loss as f64))?;
            }
        }

        let avg_loss = total_loss / num_batches as f32;
        Ok(avg_loss)
    }

    /// Full training loop
    pub fn train(&mut self, dataloader: &mut impl BatchDataLoader) -> Result<()> {
        log::info!("Starting training for {} epochs", self.config.num_epochs);
        log::info!("Total batches per epoch: {}", dataloader.num_batches());

        for epoch in 0..self.config.num_epochs {
            log::info!("=== Epoch {}/{} ===", epoch + 1, self.config.num_epochs);

            let avg_loss = self.train_epoch(dataloader)?;

            log::info!(
                "Epoch {} complete: avg_loss={:.4}, step={}",
                epoch + 1,
                avg_loss,
                self.step
            );

            // Save epoch checkpoint
            let checkpoint_path = format!(
                "{}/checkpoint_epoch_{}.safetensors",
                self.config.checkpoint_dir,
                epoch + 1
            );
            self.save_checkpoint(&checkpoint_path, Some(avg_loss as f64))?;
        }

        log::info!("Training complete!");

        // Save final model
        let final_path = format!("{}/final_model.safetensors", self.config.checkpoint_dir);
        log::info!("Saving final model to {}", final_path);
        self.varmap.save(&final_path)?;

        Ok(())
    }
}
