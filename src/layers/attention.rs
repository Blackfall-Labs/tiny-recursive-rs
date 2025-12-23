/// Multi-head attention implementation
///
/// Based on the Python implementation in layers.py
use candle_core::{Result, Tensor, DType, Device};
use candle_nn::VarBuilder;
use super::activations::CastedLinear;
use super::positional::{apply_rotary_pos_emb, RotaryEmbedding};

/// Multi-head attention with optional grouped-query attention
///
/// Supports:
/// - Standard multi-head attention (num_heads == num_key_value_heads)
/// - Grouped-query attention (num_key_value_heads < num_heads)
/// - RoPE positional embeddings
/// - Optional causal masking
pub struct Attention {
    hidden_size: usize,
    head_dim: usize,
    output_size: usize,
    num_heads: usize,
    num_key_value_heads: usize,
    causal: bool,

    qkv_proj: CastedLinear,
    o_proj: CastedLinear,
}

impl Attention {
    /// Create new Attention layer
    ///
    /// # Arguments
    /// * `hidden_size` - Input/output dimension
    /// * `head_dim` - Dimension per attention head
    /// * `num_heads` - Number of query heads
    /// * `num_key_value_heads` - Number of key/value heads (for GQA)
    /// * `causal` - Whether to use causal masking
    /// * `vb` - VarBuilder for parameter initialization
    pub fn new(
        hidden_size: usize,
        head_dim: usize,
        num_heads: usize,
        num_key_value_heads: usize,
        causal: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let output_size = head_dim * num_heads;

        // QKV projection: projects to (num_heads + 2 * num_key_value_heads) * head_dim
        let qkv_size = (num_heads + 2 * num_key_value_heads) * head_dim;
        let qkv_proj = CastedLinear::new(
            hidden_size,
            qkv_size,
            false,
            vb.pp("qkv_proj"),
        )?;

        // Output projection
        let o_proj = CastedLinear::new(
            output_size,
            hidden_size,
            false,
            vb.pp("o_proj"),
        )?;

        Ok(Self {
            hidden_size,
            head_dim,
            output_size,
            num_heads,
            num_key_value_heads,
            causal,
            qkv_proj,
            o_proj,
        })
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor [batch, seq_len, hidden_size]
    /// * `cos_sin` - Optional RoPE embeddings (cos, sin) each [seq_len, head_dim]
    ///
    /// # Returns
    /// Output tensor [batch, seq_len, hidden_size]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        cos_sin: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;

        // Project to QKV
        let qkv = self.qkv_proj.forward(hidden_states)?;

        // Reshape and split into Q, K, V
        // qkv: [batch, seq_len, (num_heads + 2 * num_kv_heads) * head_dim]
        // -> [batch, seq_len, num_heads + 2 * num_kv_heads, head_dim]
        let qkv = qkv.reshape((
            batch_size,
            seq_len,
            self.num_heads + 2 * self.num_key_value_heads,
            self.head_dim,
        ))?;

        // Split Q, K, V
        let query = qkv.narrow(2, 0, self.num_heads)?; // [batch, seq_len, num_heads, head_dim]
        let key = qkv.narrow(2, self.num_heads, self.num_key_value_heads)?;
        let value = qkv.narrow(2, self.num_heads + self.num_key_value_heads, self.num_key_value_heads)?;

        // Apply RoPE if provided
        let (query, key) = if let Some((cos, sin)) = cos_sin {
            apply_rotary_pos_emb(&query, &key, cos, sin)?
        } else {
            (query, key)
        };

        // Reshape for attention: [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        let query = query.transpose(1, 2)?.contiguous()?;
        let key = key.transpose(1, 2)?.contiguous()?;
        let value = value.transpose(1, 2)?.contiguous()?;

        // Handle grouped-query attention by repeating key/value heads if needed
        let (key, value) = if self.num_key_value_heads < self.num_heads {
            let repeat_factor = self.num_heads / self.num_key_value_heads;
            (
                repeat_kv(&key, repeat_factor)?,
                repeat_kv(&value, repeat_factor)?,
            )
        } else {
            (key, value)
        };

        // Scaled dot-product attention
        let attn_output = scaled_dot_product_attention(
            &query,
            &key,
            &value,
            self.causal,
        )?;

        // Reshape back: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
        let attn_output = attn_output.transpose(1, 2)?;

        // Concatenate heads: [batch, seq_len, num_heads * head_dim]
        let attn_output = attn_output.reshape((batch_size, seq_len, self.output_size))?;

        // Output projection
        self.o_proj.forward(&attn_output)
    }
}

/// Repeat key/value heads for grouped-query attention
///
/// Repeats each head `n` times along the head dimension.
fn repeat_kv(x: &Tensor, n: usize) -> Result<Tensor> {
    if n == 1 {
        return Ok(x.clone());
    }

    let (batch, num_kv_heads, seq_len, head_dim) = x.dims4()?;

    // Expand: [batch, num_kv_heads, seq_len, head_dim]
    // -> [batch, num_kv_heads, n, seq_len, head_dim]
    let x = x.unsqueeze(2)?;
    let x = x.broadcast_as((batch, num_kv_heads, n, seq_len, head_dim))?;

    // Reshape: [batch, num_kv_heads * n, seq_len, head_dim]
    x.reshape((batch, num_kv_heads * n, seq_len, head_dim))
}

/// Scaled dot-product attention
///
/// attention = softmax(Q @ K^T / sqrt(d_k)) @ V
///
/// # Arguments
/// * `query` - [batch, num_heads, seq_len, head_dim]
/// * `key` - [batch, num_heads, seq_len, head_dim]
/// * `value` - [batch, num_heads, seq_len, head_dim]
/// * `causal` - Whether to apply causal masking
fn scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    causal: bool,
) -> Result<Tensor> {
    let (_batch, _num_heads, seq_len, head_dim) = query.dims4()?;
    let scale = 1.0 / (head_dim as f64).sqrt();

    // Q @ K^T: [batch, num_heads, seq_len, seq_len]
    let scores = query.matmul(&key.transpose(2, 3)?)?;
    let scores = (scores * scale)?;

    // Apply causal mask if needed
    let scores = if causal {
        let mask = create_causal_mask(seq_len, scores.device())?;
        scores.broadcast_add(&mask)?
    } else {
        scores
    };

    // Softmax over last dimension
    let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;

    // attn_weights @ V: [batch, num_heads, seq_len, head_dim]
    attn_weights.matmul(value)
}

/// Create causal attention mask
///
/// Returns a mask with 0s on/below diagonal and -inf above diagonal.
/// This masks out future positions in self-attention.
fn create_causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mut mask_data = vec![0.0f32; seq_len * seq_len];

    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask_data[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }

    Tensor::from_vec(mask_data, (seq_len, seq_len), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    #[test]
    fn test_attention_shape() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let attn = Attention::new(256, 32, 8, 8, false, vb)?;

        let x = Tensor::randn(0f32, 1.0, (2, 16, 256), &device)?;
        let out = attn.forward(&x, None)?;

        assert_eq!(out.dims(), &[2, 16, 256]);

        Ok(())
    }

    #[test]
    fn test_attention_with_rope() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let attn = Attention::new(256, 32, 8, 8, false, vb)?;

        let x = Tensor::randn(0f32, 1.0, (2, 16, 256), &device)?;

        // Create RoPE embeddings
        let rope = RotaryEmbedding::new(32, 512, 10000.0, &device)?;
        let (cos, sin) = rope.forward_with_len(16)?;

        let out = attn.forward(&x, Some((&cos, &sin)))?;

        assert_eq!(out.dims(), &[2, 16, 256]);

        Ok(())
    }

    #[test]
    fn test_grouped_query_attention() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // 8 query heads, 2 key/value heads
        let attn = Attention::new(256, 32, 8, 2, false, vb)?;

        let x = Tensor::randn(0f32, 1.0, (2, 16, 256), &device)?;
        let out = attn.forward(&x, None)?;

        assert_eq!(out.dims(), &[2, 16, 256]);

        Ok(())
    }

    #[test]
    fn test_causal_mask() -> Result<()> {
        let device = Device::Cpu;
        let mask = create_causal_mask(4, &device)?;

        // Check shape
        assert_eq!(mask.dims(), &[4, 4]);

        // Check that lower triangle is 0 and upper triangle is -inf
        let mask_vec = mask.flatten_all()?.to_vec1::<f32>()?;

        // First row: [0, -inf, -inf, -inf]
        assert_eq!(mask_vec[0], 0.0);
        assert!(mask_vec[1].is_infinite() && mask_vec[1].is_sign_negative());

        // Second row: [0, 0, -inf, -inf]
        assert_eq!(mask_vec[4], 0.0);
        assert_eq!(mask_vec[5], 0.0);
        assert!(mask_vec[6].is_infinite() && mask_vec[6].is_sign_negative());

        Ok(())
    }

    #[test]
    fn test_repeat_kv() -> Result<()> {
        let device = Device::Cpu;

        let x = Tensor::randn(0f32, 1.0, (2, 2, 16, 32), &device)?;
        let repeated = repeat_kv(&x, 4)?;

        assert_eq!(repeated.dims(), &[2, 8, 16, 32]);

        Ok(())
    }
}
