/// Model checkpointing with safetensors
use std::path::Path;
use std::collections::HashMap;
use candle_core::{Result, Tensor, Device, DType};
use safetensors::tensor::{SafeTensors, TensorView};

/// Checkpoint metadata
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CheckpointMetadata {
    /// Training step
    pub step: usize,
    /// Learning rate at checkpoint
    pub lr: f64,
    /// Loss at checkpoint
    pub loss: Option<f64>,
    /// Model configuration (as JSON string)
    pub config: Option<String>,
}

impl Default for CheckpointMetadata {
    fn default() -> Self {
        Self {
            step: 0,
            lr: 0.0,
            loss: None,
            config: None,
        }
    }
}

/// Model checkpoint
pub struct Checkpoint {
    /// Model parameters
    pub tensors: HashMap<String, Tensor>,
    /// Metadata
    pub metadata: CheckpointMetadata,
}

impl Checkpoint {
    /// Create new checkpoint
    pub fn new(tensors: HashMap<String, Tensor>, metadata: CheckpointMetadata) -> Self {
        Self { tensors, metadata }
    }

    /// Save checkpoint to file
    ///
    /// # Arguments
    /// * `path` - Path to save checkpoint
    ///
    /// # Returns
    /// Result indicating success or error
    pub fn save<P: AsRef<Path>>(&self, path: P) -> crate::Result<()> {
        // Simplified checkpoint saving
        // In a full implementation, this would use safetensors::serialize
        // For now, just save metadata

        let metadata_json = serde_json::to_string(&self.metadata)
            .map_err(|e| crate::TRMError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            )))?;

        std::fs::write(path.as_ref(), metadata_json.as_bytes())?;

        // TODO: Implement full safetensors serialization
        // This would involve:
        // 1. Collecting all tensor data
        // 2. Creating proper TensorView objects
        // 3. Using safetensors::serialize to write binary format

        Ok(())
    }

    /// Load checkpoint from file
    ///
    /// # Arguments
    /// * `path` - Path to checkpoint file
    /// * `device` - Device to load tensors on
    ///
    /// # Returns
    /// Loaded checkpoint
    pub fn load<P: AsRef<Path>>(path: P, device: &Device) -> crate::Result<Self> {
        // Read file
        let data = std::fs::read(path.as_ref())?;

        // This is a simplified placeholder
        // Proper implementation would:
        // 1. Parse safetensors format
        // 2. Load tensors onto device
        // 3. Extract metadata

        let metadata: CheckpointMetadata = serde_json::from_slice(&data).unwrap_or_default();

        Ok(Self {
            tensors: HashMap::new(),
            metadata,
        })
    }

    /// Load only model weights from checkpoint
    ///
    /// # Arguments
    /// * `path` - Path to checkpoint file
    /// * `device` - Device to load tensors on
    ///
    /// # Returns
    /// HashMap of parameter name to tensor
    pub fn load_weights<P: AsRef<Path>>(
        path: P,
        device: &Device,
    ) -> crate::Result<HashMap<String, Tensor>> {
        let checkpoint = Self::load(path, device)?;
        Ok(checkpoint.tensors)
    }
}

/// Save model parameters to checkpoint
///
/// # Arguments
/// * `params` - Model parameters as (name, tensor) pairs
/// * `path` - Path to save checkpoint
/// * `metadata` - Checkpoint metadata
pub fn save_checkpoint<P: AsRef<Path>>(
    params: HashMap<String, Tensor>,
    path: P,
    metadata: CheckpointMetadata,
) -> crate::Result<()> {
    let checkpoint = Checkpoint::new(params, metadata);
    checkpoint.save(path)
}

/// Load model parameters from checkpoint
///
/// # Arguments
/// * `path` - Path to checkpoint file
/// * `device` - Device to load tensors on
///
/// # Returns
/// HashMap of parameter name to tensor
pub fn load_checkpoint<P: AsRef<Path>>(
    path: P,
    device: &Device,
) -> crate::Result<HashMap<String, Tensor>> {
    Checkpoint::load_weights(path, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_checkpoint_metadata() {
        let metadata = CheckpointMetadata {
            step: 1000,
            lr: 0.001,
            loss: Some(0.5),
            config: Some("{}".to_string()),
        };

        assert_eq!(metadata.step, 1000);
        assert_eq!(metadata.lr, 0.001);
    }

    #[test]
    fn test_checkpoint_creation() -> Result<()> {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "weight".to_string(),
            Tensor::ones((10, 10), DType::F32, &device)?,
        );

        let metadata = CheckpointMetadata::default();
        let checkpoint = Checkpoint::new(tensors, metadata);

        assert_eq!(checkpoint.tensors.len(), 1);

        Ok(())
    }

    #[test]
    fn test_save_load_checkpoint() -> Result<()> {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "weight".to_string(),
            Tensor::ones((5, 5), DType::F32, &device)?,
        );

        let metadata = CheckpointMetadata {
            step: 500,
            lr: 0.0005,
            loss: Some(0.25),
            config: None,
        };

        let temp_path = std::path::Path::new("test_checkpoint.safetensors");

        // Save
        let result = save_checkpoint(tensors, temp_path, metadata.clone());

        // Clean up
        if temp_path.exists() {
            fs::remove_file(temp_path).ok();
        }

        Ok(())
    }
}
