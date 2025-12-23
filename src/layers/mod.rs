/// Neural network layer primitives
///
/// This module contains the building blocks for the TRM model:
/// - Attention mechanisms (multi-head, grouped-query)
/// - Positional encodings (RoPE, learned)
/// - Activations (SwiGLU, LinearSwish)
/// - Normalizations (RMS LayerNorm)
/// - Embeddings (with automatic dtype casting)

pub mod attention;
pub mod embeddings;
pub mod normalization;
pub mod activations;
pub mod positional;

pub use attention::Attention;
pub use embeddings::CastedEmbedding;
pub use normalization::RMSNorm;
pub use activations::{SwiGLU, LinearSwish};
pub use positional::RotaryEmbedding;
