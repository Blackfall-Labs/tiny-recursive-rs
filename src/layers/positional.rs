/// Rotary Positional Embeddings (RoPE)
///
/// Based on the Python implementation in layers.py
use candle_core::{Result, Tensor, Device, DType, D};

/// Rotates half the hidden dims of the input
///
/// Splits the tensor along the last dimension and rotates:
/// [x1, x2] -> [-x2, x1]
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let last_dim = x.dims().len() - 1;
    let dim_size = x.dim(last_dim)?;
    let half = dim_size / 2;

    // Split into two halves
    let x1 = x.narrow(last_dim, 0, half)?;
    let x2 = x.narrow(last_dim, half, half)?;

    // Concatenate [-x2, x1]
    Tensor::cat(&[&x2.neg()?, &x1], last_dim)
}

/// Apply rotary positional embeddings to query and key tensors
///
/// # Arguments
/// * `q` - Query tensor [batch, seq_len, num_heads, head_dim]
/// * `k` - Key tensor [batch, seq_len, num_heads, head_dim]
/// * `cos` - Cosine embeddings [seq_len, head_dim]
/// * `sin` - Sine embeddings [seq_len, head_dim]
///
/// # Returns
/// Tuple of (rotated_q, rotated_k) with same shapes as inputs
pub fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let orig_dtype = q.dtype();

    // Convert to same dtype as cos/sin for computation
    let q = if q.dtype() != cos.dtype() {
        q.to_dtype(cos.dtype())?
    } else {
        q.clone()
    };

    let k = if k.dtype() != cos.dtype() {
        k.to_dtype(cos.dtype())?
    } else {
        k.clone()
    };

    // Reshape cos/sin to broadcast: [seq_len, head_dim] -> [1, seq_len, 1, head_dim]
    let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

    // Apply rotation: q_embed = (q * cos) + (rotate_half(q) * sin)
    let q_rotated = rotate_half(&q)?;
    let q_embed = q.broadcast_mul(&cos)?.add(&q_rotated.broadcast_mul(&sin)?)?;

    let k_rotated = rotate_half(&k)?;
    let k_embed = k.broadcast_mul(&cos)?.add(&k_rotated.broadcast_mul(&sin)?)?;

    // Convert back to original dtype
    let q_embed = if q_embed.dtype() != orig_dtype {
        q_embed.to_dtype(orig_dtype)?
    } else {
        q_embed
    };

    let k_embed = if k_embed.dtype() != orig_dtype {
        k_embed.to_dtype(orig_dtype)?
    } else {
        k_embed
    };

    Ok((q_embed, k_embed))
}

/// Rotary Positional Embedding layer
///
/// Precomputes cos/sin embeddings for all positions up to max_position_embeddings
pub struct RotaryEmbedding {
    cos_cached: Tensor,
    sin_cached: Tensor,
}

impl RotaryEmbedding {
    /// Create new RoPE embeddings
    ///
    /// # Arguments
    /// * `dim` - Dimension of the embeddings (must be even)
    /// * `max_position_embeddings` - Maximum sequence length
    /// * `base` - Base for inverse frequencies (typically 10000.0)
    /// * `device` - Device to create tensors on
    pub fn new(
        dim: usize,
        max_position_embeddings: usize,
        base: f32,
        device: &Device,
    ) -> Result<Self> {
        // Compute inverse frequencies: 1.0 / (base^(i/dim)) for i in [0, 2, 4, ..., dim-2]
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| {
                let exponent = i as f32 / dim as f32;
                1.0 / base.powf(exponent)
            })
            .collect();

        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;

        // Create position indices: [0, 1, 2, ..., max_position_embeddings-1]
        let t: Vec<f32> = (0..max_position_embeddings).map(|i| i as f32).collect();
        let t = Tensor::new(t.as_slice(), device)?;

        // Compute outer product: freqs[i, j] = t[i] * inv_freq[j]
        // Shape: [max_position_embeddings, dim/2]
        let freqs = t.unsqueeze(1)?.broadcast_mul(&inv_freq.unsqueeze(0)?)?;

        // Concatenate freqs with itself to get full embedding
        // Shape: [max_position_embeddings, dim]
        let emb = Tensor::cat(&[&freqs, &freqs], 1)?;

        // Cache cos and sin
        let cos_cached = emb.cos()?;
        let sin_cached = emb.sin()?;

        Ok(Self {
            cos_cached,
            sin_cached,
        })
    }

    /// Get the cached cos/sin embeddings
    ///
    /// # Returns
    /// Tuple of (cos, sin) tensors with shape [max_position_embeddings, dim]
    pub fn forward(&self) -> Result<(Tensor, Tensor)> {
        Ok((self.cos_cached.clone(), self.sin_cached.clone()))
    }

    /// Get cos/sin embeddings for a specific sequence length
    ///
    /// # Arguments
    /// * `seq_len` - Length of the sequence (must be <= max_position_embeddings)
    ///
    /// # Returns
    /// Tuple of (cos, sin) tensors with shape [seq_len, dim]
    pub fn forward_with_len(&self, seq_len: usize) -> Result<(Tensor, Tensor)> {
        let cos = self.cos_cached.narrow(0, 0, seq_len)?;
        let sin = self.sin_cached.narrow(0, 0, seq_len)?;
        Ok((cos, sin))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotate_half() -> Result<()> {
        let device = Device::Cpu;

        // Simple test: [1, 2, 3, 4] -> [-3, -4, 1, 2]
        let x = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)?.reshape((1, 4))?;
        let rotated = rotate_half(&x)?;

        let expected = Tensor::new(&[-3.0f32, -4.0, 1.0, 2.0], &device)?.reshape((1, 4))?;

        let diff = rotated.sub(&expected)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-6, "rotate_half failed");

        Ok(())
    }

    #[test]
    fn test_rotary_embedding_shape() -> Result<()> {
        let device = Device::Cpu;

        let rope = RotaryEmbedding::new(64, 512, 10000.0, &device)?;
        let (cos, sin) = rope.forward()?;

        assert_eq!(cos.dims(), &[512, 64]);
        assert_eq!(sin.dims(), &[512, 64]);

        Ok(())
    }

    #[test]
    fn test_rotary_embedding_with_len() -> Result<()> {
        let device = Device::Cpu;

        let rope = RotaryEmbedding::new(64, 512, 10000.0, &device)?;
        let (cos, sin) = rope.forward_with_len(128)?;

        assert_eq!(cos.dims(), &[128, 64]);
        assert_eq!(sin.dims(), &[128, 64]);

        Ok(())
    }

    #[test]
    fn test_apply_rotary_pos_emb_shape() -> Result<()> {
        let device = Device::Cpu;

        // Create dummy q, k tensors: [batch, seq_len, num_heads, head_dim]
        let q = Tensor::randn(0f32, 1.0, (2, 16, 8, 64), &device)?;
        let k = Tensor::randn(0f32, 1.0, (2, 16, 8, 64), &device)?;

        // Create RoPE embeddings
        let rope = RotaryEmbedding::new(64, 512, 10000.0, &device)?;
        let (cos, sin) = rope.forward_with_len(16)?;

        // Apply RoPE
        let (q_embed, k_embed) = apply_rotary_pos_emb(&q, &k, &cos, &sin)?;

        // Shapes should be preserved
        assert_eq!(q_embed.dims(), q.dims());
        assert_eq!(k_embed.dims(), k.dims());

        Ok(())
    }
}
