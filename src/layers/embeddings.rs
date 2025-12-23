/// Embedding layer with automatic dtype casting
use candle_core::{Module, Result, Tensor, DType};
use candle_nn::{Embedding, VarBuilder};

pub struct CastedEmbedding {
    embedding: Embedding,
    target_dtype: DType,
}

impl CastedEmbedding {
    pub fn new(vocab_size: usize, hidden_size: usize, vb: VarBuilder, target_dtype: DType) -> Result<Self> {
        let embedding = candle_nn::embedding(vocab_size, hidden_size, vb)?;
        Ok(Self {
            embedding,
            target_dtype,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let output = self.embedding.forward(input)?;
        if output.dtype() != self.target_dtype {
            output.to_dtype(self.target_dtype)
        } else {
            Ok(output)
        }
    }
}
