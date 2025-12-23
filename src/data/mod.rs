/// Data loading modules for TRM training
pub mod numpy_dataset;

pub use numpy_dataset::{NumpyDataset, NumpyDataLoader, DatasetMetadata};

use candle_core::{Result, Tensor, Device};

/// Generic data loader trait
pub trait BatchDataLoader {
    /// Get next batch of (input, target) tensors
    fn next_batch(&mut self, device: &Device) -> Result<Option<(Tensor, Tensor)>>;

    /// Reset loader for new epoch
    fn reset(&mut self);

    /// Get total number of batches
    fn num_batches(&self) -> usize;
}
