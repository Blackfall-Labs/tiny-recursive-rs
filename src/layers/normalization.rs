/// RMS Layer Normalization
///
/// Based on the Python implementation: `rms_norm` function in layers.py
use candle_core::{Result, Tensor, DType};

/// RMS Normalization function
///
/// Normalizes by root mean square with a small epsilon for numerical stability.
/// The computation is done in f32 for precision, then cast back to original dtype.
///
/// # Arguments
/// * `hidden_states` - Input tensor
/// * `variance_epsilon` - Small constant for numerical stability (typically 1e-6)
///
/// # Returns
/// Normalized tensor with same shape and dtype as input
pub fn rms_norm(hidden_states: &Tensor, variance_epsilon: f64) -> Result<Tensor> {
    let input_dtype = hidden_states.dtype();

    // Convert to f32 for precision
    let hidden_states = if input_dtype != DType::F32 {
        hidden_states.to_dtype(DType::F32)?
    } else {
        hidden_states.clone()
    };

    // Compute variance: mean of squares along last dimension
    let variance = hidden_states.sqr()?.mean_keepdim(candle_core::D::Minus1)?;

    // Normalize: x * rsqrt(variance + epsilon)
    let normalized = hidden_states.broadcast_div(
        &(variance + variance_epsilon)?.sqrt()?
    )?;

    // Convert back to original dtype
    if input_dtype != DType::F32 {
        normalized.to_dtype(input_dtype)
    } else {
        Ok(normalized)
    }
}

/// RMS Normalization layer with learnable scale
pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

impl RMSNorm {
    pub fn new(hidden_size: usize, eps: f64, vb: candle_nn::VarBuilder) -> Result<Self> {
        // Initialize weight to ones
        let weight = vb.get((hidden_size,), "weight")?;
        Ok(Self { weight, eps })
    }

    /// Create RMSNorm without learnable parameters (just the function)
    pub fn new_no_weight(hidden_size: usize, eps: f64, device: &candle_core::Device) -> Result<Self> {
        let weight = Tensor::ones((hidden_size,), DType::F32, device)?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let normalized = rms_norm(x, self.eps)?;
        // Apply learnable scale
        normalized.broadcast_mul(&self.weight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};

    #[test]
    fn test_rms_norm_basic() -> Result<()> {
        let device = Device::Cpu;

        // Create a simple tensor [1, 2, 3, 4]
        let x = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)?.reshape((1, 4))?;

        // Apply RMS norm
        let normalized = rms_norm(&x, 1e-6)?;

        // The RMS should be approximately 1.0 after normalization
        let rms = normalized.sqr()?.mean_all()?.to_scalar::<f32>()?;
        assert!((rms - 1.0).abs() < 0.1, "RMS should be close to 1.0, got {}", rms);

        Ok(())
    }

    #[test]
    fn test_rms_norm_preserves_shape() -> Result<()> {
        let device = Device::Cpu;

        // Test with multiple shapes
        let x = Tensor::randn(0f32, 1.0, (2, 8, 64), &device)?;
        let normalized = rms_norm(&x, 1e-6)?;

        assert_eq!(x.dims(), normalized.dims());

        Ok(())
    }
}
