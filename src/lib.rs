//! Tiny Recursive Models - Rust implementation
//!
//! A Rust port of TinyRecursiveModels for fast, efficient recursive reasoning
//! with tiny parameter counts (~7M params).
//!
//! # Architecture
//!
//! The model uses a hierarchical recursive architecture with:
//! - **H-cycles**: High-level reasoning cycles for refinement
//! - **L-cycles**: Low-level update cycles for detailed processing
//! - **ACT**: Adaptive Computation Time for dynamic halting
//!
//! # Example
//!
//! ```ignore
//! use tiny_recursive::{TinyRecursiveModel, TRMConfig};
//!
//! let config = TRMConfig::default();
//! let model = TinyRecursiveModel::new(config)?;
//! let output = model.forward(input)?;
//! ```

pub mod config;
pub mod data;
pub mod layers;
pub mod models;
pub mod training;
pub mod utils;

// Re-export commonly used items
pub use config::TRMConfig;
pub use models::TinyRecursiveModel;

/// Library error types
#[derive(Debug, thiserror::Error)]
pub enum TRMError {
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Training error: {0}")]
    Training(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, TRMError>;
