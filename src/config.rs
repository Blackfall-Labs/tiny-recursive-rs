/// Configuration for Tiny Recursive Model
///
/// Based on TinyRecursiveReasoningModel_ACTV1Config from the Python implementation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TRMConfig {
    /// Embedding/hidden dimension
    pub hidden_size: usize,

    /// Number of high-level reasoning cycles
    pub h_cycles: usize,

    /// Number of low-level update cycles per H-cycle
    pub l_cycles: usize,

    /// Number of layers in L-level (low-level) blocks
    pub l_layers: usize,

    /// Number of attention heads
    pub num_heads: usize,

    /// FFN expansion factor (hidden_size * expansion)
    pub expansion: f32,

    /// Positional encoding type: "rope", "learned", or "none"
    pub pos_encodings: String,

    /// Use MLP instead of transformer (smaller, faster)
    pub mlp_t: bool,

    /// Maximum steps for ACT halting
    pub halt_max_steps: usize,

    /// Dropout probability
    pub dropout: f32,

    /// Vocabulary size (for embeddings)
    pub vocab_size: usize,

    /// Number of output classes/tokens
    pub num_outputs: usize,
}

impl Default for TRMConfig {
    fn default() -> Self {
        Self {
            hidden_size: 256,
            h_cycles: 3,
            l_cycles: 6,
            l_layers: 2,
            num_heads: 8,
            expansion: 4.0,
            pos_encodings: "rope".to_string(),
            mlp_t: false,
            halt_max_steps: 10,
            dropout: 0.0,
            vocab_size: 50257, // GPT-2 vocab size as default
            num_outputs: 50257,
        }
    }
}

impl TRMConfig {
    /// Validate configuration
    pub fn validate(&self) -> crate::Result<()> {
        if self.hidden_size == 0 {
            return Err(crate::TRMError::Config(
                "hidden_size must be > 0".to_string(),
            ));
        }

        if self.hidden_size % self.num_heads != 0 {
            return Err(crate::TRMError::Config(
                "hidden_size must be divisible by num_heads".to_string(),
            ));
        }

        if self.h_cycles == 0 || self.l_cycles == 0 {
            return Err(crate::TRMError::Config(
                "h_cycles and l_cycles must be > 0".to_string(),
            ));
        }

        if !["rope", "learned", "none"].contains(&self.pos_encodings.as_str()) {
            return Err(crate::TRMError::Config(format!(
                "Invalid pos_encodings: {}. Must be 'rope', 'learned', or 'none'",
                self.pos_encodings
            )));
        }

        Ok(())
    }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// Get FFN hidden size
    pub fn ffn_hidden_size(&self) -> usize {
        (self.hidden_size as f32 * self.expansion) as usize
    }
}
