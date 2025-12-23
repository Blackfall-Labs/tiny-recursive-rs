/// Utility functions for TRM
use candle_core::{Result, Tensor, Device};
use num_traits::Float;

/// Truncated normal initialization
pub fn trunc_normal_init<F: Float>(std: F, a: F, b: F) -> impl Fn() -> F {
    move || {
        // TODO: Implement truncated normal
        // For now, return 0
        F::zero()
    }
}

/// Calculate the number of parameters in a tensor
pub fn count_parameters(tensor: &Tensor) -> usize {
    tensor.dims().iter().product()
}

/// Create a causal mask for attention
pub fn create_causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    // TODO: Implement causal mask creation
    todo!("Implement causal mask")
}
