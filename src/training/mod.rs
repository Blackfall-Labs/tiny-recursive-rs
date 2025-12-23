/// Training infrastructure for TRM

// Note: Using candle-nn's built-in AdamW optimizer instead of custom implementation
// pub mod optimizer;
pub mod scheduler;
pub mod ema;
pub mod checkpoint;
pub mod trainer;

pub use scheduler::CosineScheduler;
pub use ema::EMA;
pub use checkpoint::Checkpoint;
pub use trainer::{Trainer, TrainingConfig};
