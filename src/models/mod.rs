/// Tiny Recursive Model implementation
use candle_core::{Result, Tensor, Device, DType};
use candle_nn::{VarBuilder, Module};
use crate::config::TRMConfig;
use crate::layers::{Attention, SwiGLU, CastedEmbedding, RMSNorm, RotaryEmbedding};
use crate::layers::normalization::rms_norm;
use crate::layers::activations::CastedLinear;

pub mod loader;

/// State carry for recursive computation
///
/// Holds the high-level and low-level states that are refined
/// across recursive cycles.
#[derive(Debug, Clone)]
pub struct InnerCarry {
    /// High-level state: [batch, seq_len, hidden_size]
    pub z_h: Tensor,
    /// Low-level state: [batch, seq_len, hidden_size]
    pub z_l: Tensor,
}

impl InnerCarry {
    /// Create new carry with given states
    pub fn new(z_h: Tensor, z_l: Tensor) -> Self {
        Self { z_h, z_l }
    }

    /// Create empty carry (uninitialized tensors)
    pub fn empty(batch_size: usize, seq_len: usize, hidden_size: usize, dtype: DType, device: &Device) -> Result<Self> {
        let z_h = Tensor::zeros((batch_size, seq_len, hidden_size), dtype, device)?;
        let z_l = Tensor::zeros((batch_size, seq_len, hidden_size), dtype, device)?;
        Ok(Self { z_h, z_l })
    }
}

/// Transformer block for TRM
///
/// Each block consists of:
/// - Self-attention (optional, not used in MLP mode)
/// - Feed-forward network (SwiGLU)
/// - RMS normalization with residual connections (post-norm)
pub struct TransformerBlock {
    config: TRMConfig,
    self_attn: Option<Attention>,
    mlp: SwiGLU,
    norm_eps: f64,
}

impl TransformerBlock {
    /// Create new transformer block
    pub fn new(config: TRMConfig, vb: VarBuilder) -> Result<Self> {
        // Self-attention (only if not using MLP-T mode)
        let self_attn = if !config.mlp_t {
            Some(Attention::new(
                config.hidden_size,
                config.head_dim(),
                config.num_heads,
                config.num_heads, // num_key_value_heads = num_heads (no GQA by default)
                false, // not causal
                vb.pp("self_attn"),
            )?)
        } else {
            None
        };

        // Feed-forward network
        let mlp = SwiGLU::new(
            config.hidden_size,
            config.expansion,
            vb.pp("mlp"),
        )?;

        Ok(Self {
            config: config.clone(),
            self_attn,
            mlp,
            norm_eps: 1e-5,
        })
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor [batch, seq_len, hidden_size]
    /// * `cos_sin` - Optional RoPE embeddings
    ///
    /// # Returns
    /// Output tensor [batch, seq_len, hidden_size]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        cos_sin: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();

        // Self-attention sublayer (if not MLP mode)
        if let Some(ref attn) = self.self_attn {
            let attn_out = attn.forward(&hidden_states, cos_sin)?;
            hidden_states = rms_norm(&(hidden_states + attn_out)?, self.norm_eps)?;
        }

        // Feed-forward sublayer
        let mlp_out = self.mlp.forward(&hidden_states)?;
        hidden_states = rms_norm(&(hidden_states + mlp_out)?, self.norm_eps)?;

        Ok(hidden_states)
    }
}

/// Reasoning module: stack of transformer blocks with input injection
///
/// This is the L-level or H-level reasoning component.
pub struct ReasoningModule {
    layers: Vec<TransformerBlock>,
}

impl ReasoningModule {
    /// Create new reasoning module
    ///
    /// # Arguments
    /// * `num_layers` - Number of transformer blocks
    /// * `config` - Model configuration
    /// * `vb` - VarBuilder for parameter initialization
    pub fn new(num_layers: usize, config: TRMConfig, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 0..num_layers {
            layers.push(TransformerBlock::new(
                config.clone(),
                vb.pp(&format!("layer_{}", i)),
            )?);
        }

        Ok(Self { layers })
    }

    /// Forward pass with input injection
    ///
    /// # Arguments
    /// * `hidden_states` - Current state
    /// * `input_injection` - Input to inject (added to hidden_states)
    /// * `cos_sin` - Optional RoPE embeddings
    ///
    /// # Returns
    /// Updated state after processing through all layers
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        input_injection: &Tensor,
        cos_sin: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        // Add input injection
        let mut hidden_states = (hidden_states + input_injection)?;

        // Process through all layers
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, cos_sin)?;
        }

        Ok(hidden_states)
    }
}

/// Main Tiny Recursive Model
///
/// Implements the recursive reasoning architecture with H-cycles and L-cycles.
pub struct TinyRecursiveModel {
    config: TRMConfig,

    // I/O components
    embed_tokens: CastedEmbedding,
    lm_head: CastedLinear,
    embed_scale: f64,

    // Positional encodings
    rotary_emb: Option<RotaryEmbedding>,

    // Reasoning components
    l_level: ReasoningModule,

    // Initial states
    h_init: Tensor,
    l_init: Tensor,

    // Device
    device: Device,
}

impl TinyRecursiveModel {
    /// Create new TinyRecursiveModel
    pub fn new(config: TRMConfig, vb: VarBuilder) -> crate::Result<Self> {
        config.validate()?;

        let device = vb.device().clone();
        let dtype = vb.dtype();

        // Embedding scale: sqrt(hidden_size)
        let embed_scale = (config.hidden_size as f64).sqrt();

        // Token embeddings
        let embed_tokens = CastedEmbedding::new(
            config.vocab_size,
            config.hidden_size,
            vb.pp("embed_tokens"),
            dtype,
        )?;

        // Output head
        let lm_head = CastedLinear::new(
            config.hidden_size,
            config.num_outputs,
            false,
            vb.pp("lm_head"),
        )?;

        // Positional encodings
        let rotary_emb = if config.pos_encodings == "rope" {
            Some(RotaryEmbedding::new(
                config.head_dim(),
                2048, // max sequence length
                10000.0,
                &device,
            )?)
        } else {
            None
        };

        // L-level reasoning module
        let l_level = ReasoningModule::new(
            config.l_layers,
            config.clone(),
            vb.pp("l_level"),
        )?;

        // Initial states (learnable parameters)
        let h_init = vb.get(config.hidden_size, "h_init")?;
        let l_init = vb.get(config.hidden_size, "l_init")?;

        Ok(Self {
            config,
            embed_tokens,
            lm_head,
            embed_scale,
            rotary_emb,
            l_level,
            h_init,
            l_init,
            device,
        })
    }

    /// Create empty carry for a batch
    pub fn empty_carry(&self, batch_size: usize) -> Result<InnerCarry> {
        InnerCarry::empty(
            batch_size,
            self.config.vocab_size, // Using vocab_size as placeholder for seq_len
            self.config.hidden_size,
            DType::F32,
            &self.device,
        )
    }

    /// Reset carry to initial states where reset_flag is true
    ///
    /// # Arguments
    /// * `reset_flag` - Boolean tensor [batch_size] indicating which sequences to reset
    /// * `carry` - Current carry state
    pub fn reset_carry(&self, reset_flag: &Tensor, carry: &InnerCarry) -> Result<InnerCarry> {
        // Reshape reset_flag to [batch, 1, 1] for broadcasting
        let reset_flag = reset_flag.unsqueeze(1)?.unsqueeze(1)?;

        // Broadcast h_init and l_init to batch dimensions
        let batch_size = carry.z_h.dim(0)?;
        let seq_len = carry.z_h.dim(1)?;

        let h_init = self.h_init
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((batch_size, seq_len, self.config.hidden_size))?;

        let l_init = self.l_init
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((batch_size, seq_len, self.config.hidden_size))?;

        // Where reset_flag is true, use init states; otherwise use carry states
        let z_h = reset_flag.where_cond(&h_init, &carry.z_h)?;
        let z_l = reset_flag.where_cond(&l_init, &carry.z_l)?;

        Ok(InnerCarry::new(z_h, z_l))
    }

    /// Encode input tokens to embeddings
    fn input_embeddings(&self, input: &Tensor) -> Result<Tensor> {
        // Token embedding
        let embedding = self.embed_tokens.forward(input)?;

        // Scale by sqrt(hidden_size)
        embedding.affine(self.embed_scale, 0.0)
    }

    /// Forward pass with recursive reasoning
    ///
    /// # Arguments
    /// * `carry` - Current state (z_H, z_L)
    /// * `input` - Input token IDs [batch, seq_len]
    ///
    /// # Returns
    /// Tuple of (new_carry, logits)
    /// - new_carry: Updated state for next iteration
    /// - logits: Output logits [batch, seq_len, vocab_size]
    pub fn forward(&self, carry: &InnerCarry, input: &Tensor) -> Result<(InnerCarry, Tensor)> {
        let seq_len = input.dim(1)?;

        // Get RoPE embeddings if needed
        let cos_sin = if let Some(ref rope) = self.rotary_emb {
            let (cos, sin) = rope.forward_with_len(seq_len)?;
            Some((cos, sin))
        } else {
            None
        };

        // Input encoding
        let input_embeddings = self.input_embeddings(input)?;

        // Extract current states
        let mut z_h = carry.z_h.clone();
        let mut z_l = carry.z_l.clone();

        // Recursive forward iterations
        // Pattern from Python:
        // - (H_cycles - 1) iterations without gradients
        // - 1 final iteration with gradients
        //
        // Each H-cycle:
        //   - L_cycles iterations: z_L = L_level(z_L, z_H + input)
        //   - 1 iteration: z_H = L_level(z_H, z_L)

        // For inference in Rust, we don't need the gradient control,
        // so we just do all H_cycles normally
        for _h_step in 0..self.config.h_cycles {
            // L-cycles: refine z_L with z_H + input as injection
            for _l_step in 0..self.config.l_cycles {
                let injection = (&z_h + &input_embeddings)?;
                z_l = self.l_level.forward(
                    &z_l,
                    &injection,
                    cos_sin.as_ref().map(|(c, s)| (c.as_ref(), s.as_ref())),
                )?;
            }

            // Update z_H with z_L as injection
            z_h = self.l_level.forward(
                &z_h,
                &z_l,
                cos_sin.as_ref().map(|(c, s)| (c.as_ref(), s.as_ref())),
            )?;
        }

        // Output logits
        let logits = self.lm_head.forward(&z_h)?;

        // New carry (detached for stateful inference)
        let new_carry = InnerCarry::new(z_h.clone(), z_l.clone());

        Ok((new_carry, logits))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    #[test]
    fn test_inner_carry_creation() -> Result<()> {
        let device = Device::Cpu;

        let carry = InnerCarry::empty(2, 16, 256, DType::F32, &device)?;

        assert_eq!(carry.z_h.dims(), &[2, 16, 256]);
        assert_eq!(carry.z_l.dims(), &[2, 16, 256]);

        Ok(())
    }

    #[test]
    fn test_transformer_block() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let mut config = TRMConfig::default();
        config.hidden_size = 256;
        config.num_heads = 8;

        let block = TransformerBlock::new(config, vb)?;

        let x = Tensor::randn(0f32, 1.0, (2, 16, 256), &device)?;
        let out = block.forward(&x, None)?;

        assert_eq!(out.dims(), &[2, 16, 256]);

        Ok(())
    }

    #[test]
    fn test_reasoning_module() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let mut config = TRMConfig::default();
        config.hidden_size = 256;
        config.num_heads = 8;
        config.l_layers = 2;

        let module = ReasoningModule::new(2, config, vb)?;

        let hidden = Tensor::randn(0f32, 1.0, (2, 16, 256), &device)?;
        let injection = Tensor::randn(0f32, 1.0, (2, 16, 256), &device)?;

        let out = module.forward(&hidden, &injection, None)?;

        assert_eq!(out.dims(), &[2, 16, 256]);

        Ok(())
    }
}
