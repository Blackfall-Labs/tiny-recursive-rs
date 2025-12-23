/// Sudoku parity training - validate Rust TRM against Python TinyRecursiveModels
use candle_core::Device;
use tiny_recursive_rs::{TRMConfig, data::{NumpyDataset, NumpyDataLoader}};
use tiny_recursive_rs::training::{Trainer, TrainingConfig};

fn main() -> anyhow::Result<()> {
    // Initialize logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    log::info!("=== TinyRecursiveModel - Sudoku Parity Training ===");
    log::info!("Goal: Match Python TRM performance (75-87% accuracy)");

    // Device setup - CPU training (GPU OOM with recursive architecture)
    let device = Device::Cpu;
    log::info!("Using device: {:?}", device);

    // Data loading - using smaller 100K dataset for faster validation
    let data_path = "E:\\repos\\TinyRecursiveModels\\data\\sudoku-extreme-100-aug-1000\\train\\";

    log::info!("Loading Sudoku dataset from: {}", data_path);
    let dataset = NumpyDataset::from_directory(data_path)?;

    log::info!("Dataset loaded:");
    log::info!("  - Total examples: {}", dataset.len());
    log::info!("  - Vocab size: {}", dataset.vocab_size());
    log::info!("  - Sequence length: {}", dataset.seq_len());
    log::info!("  - Description: {}", dataset.metadata().description);

    // Create data loader
    // Python uses global_batch_size=768, but we use smaller batch for CPU training
    // CPU handles batch_size=8 well (GPU OOMs even with batch_size=2)
    let batch_size = 8;  // Stable batch size for CPU training
    let mut dataloader = NumpyDataLoader::new(dataset, batch_size, true);

    log::info!("Data loader created:");
    log::info!("  - Batch size: {}", batch_size);
    log::info!("  - Num batches: {}", dataloader.num_batches());

    // Model configuration - EXACT match to Python TinyRecursiveModels for Sudoku
    let model_config = TRMConfig {
        vocab_size: dataloader.dataset().vocab_size(),   // 11 (PAD + digits 0-9)
        num_outputs: dataloader.dataset().vocab_size(),  // 11
        hidden_size: 512,      // Match Python
        h_cycles: 3,           // Match Python
        l_cycles: 6,           // 6 for Sudoku specifically (4 for ARC-AGI)
        l_layers: 2,           // Match Python
        num_heads: 8,          // Match Python
        expansion: 4.0,        // Match Python
        pos_encodings: "rope".to_string(),
        mlp_t: false,
        halt_max_steps: 10,
        dropout: 0.0,          // Python uses 0.0 for puzzle tasks
    };

    log::info!("Model configuration (Python-matched): {:#?}", model_config);

    // Calculate approximate parameter count
    let embed_params = model_config.vocab_size * model_config.hidden_size;
    let layer_params = model_config.hidden_size * model_config.hidden_size * 4 * model_config.l_layers;
    let head_params = model_config.hidden_size * model_config.num_outputs;
    let total_params = embed_params + layer_params + head_params;
    log::info!("Approximate parameters: ~{:.2}M", total_params as f64 / 1_000_000.0);

    // Training configuration - EXACT match to Python
    let num_batches = dataloader.num_batches();
    let num_epochs = 10;  // Start with 10 epochs for validation (Python uses 100k steps)
    let total_steps = num_batches * num_epochs;

    let training_config = TrainingConfig {
        num_epochs,
        batch_size,
        learning_rate: 1e-4,     // Match Python exactly
        lr_min: 1e-4,            // Python uses lr_min_ratio=1.0 (no decay)
        warmup_steps: 2000,      // Match Python
        total_steps,
        weight_decay: 0.1,       // Match Python
        grad_clip: Some(1.0),    // Conservative clipping
        ema_decay: 0.999,        // Match Python ema_rate
        save_every: 5000,
        eval_every: 1000,
        checkpoint_dir: "checkpoints_sudoku".to_string(),
    };

    log::info!("Training configuration (Python-matched):");
    log::info!("  - Epochs: {}", training_config.num_epochs);
    log::info!("  - Batch size: {}", training_config.batch_size);
    log::info!("  - Learning rate: {:.6}", training_config.learning_rate);
    log::info!("  - LR min: {:.6}", training_config.lr_min);
    log::info!("  - Warmup steps: {}", training_config.warmup_steps);
    log::info!("  - Total steps: {}", training_config.total_steps);
    log::info!("  - Weight decay: {}", training_config.weight_decay);
    log::info!("  - EMA decay: {}", training_config.ema_decay);

    // Expected performance metrics
    log::info!("\n=== Expected Performance ===");
    log::info!("Target accuracy: 75-87% (Python baseline)");
    log::info!("Initial loss: ~2.4 (ln(11) for random init)");
    log::info!("Training time: 8-12 hours on CPU (estimated)");
    log::info!("Checkpoint directory: {}", training_config.checkpoint_dir);

    // Create trainer
    log::info!("\nInitializing trainer...");
    let mut trainer = Trainer::new(model_config, training_config, device)?;

    // Train!
    log::info!("\nStarting training...");
    log::info!("This will take a while. Monitor loss convergence.");
    log::info!("Press Ctrl+C to stop (checkpoints are saved every 5000 steps)");

    trainer.train(&mut dataloader)?;

    log::info!("\n=== Training Complete ===");
    log::info!("Check final checkpoint for accuracy evaluation");
    log::info!("Compare training curve with Python results");

    Ok(())
}
