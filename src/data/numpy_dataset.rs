/// NumPy dataset loader for TinyRecursiveModels puzzle data (.npy format)
use candle_core::{Result, Tensor, Device};
use ndarray::{Array1, Array2, ArrayView1};
use ndarray_npy::ReadNpyExt;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// Metadata from dataset.json
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DatasetMetadata {
    pub vocab_size: usize,
    pub seq_len: usize,
    #[serde(default)]
    pub num_examples: usize,
    #[serde(default)]
    pub description: String,
}

/// Dataset loaded from NumPy .npy files
pub struct NumpyDataset {
    inputs: Array2<i32>,        // [N, seq_len]
    labels: Array2<i32>,        // [N, seq_len]
    puzzle_ids: Vec<i32>,       // [M] - puzzle identifiers
    metadata: DatasetMetadata,
}

impl NumpyDataset {
    /// Load from directory containing .npy files and dataset.json
    pub fn from_directory<P: AsRef<Path>>(path: P) -> crate::Result<Self> {
        let dir = path.as_ref();

        log::info!("Loading NumPy dataset from: {:?}", dir);

        // Load metadata
        let metadata_path = dir.join("dataset.json");
        let metadata: DatasetMetadata = if metadata_path.exists() {
            let file = File::open(&metadata_path)?;
            let reader = BufReader::new(file);
            serde_json::from_reader(reader)?
        } else {
            log::warn!("dataset.json not found, using defaults");
            DatasetMetadata {
                vocab_size: 256,
                seq_len: 64,
                num_examples: 0,
                description: "Unknown".to_string(),
            }
        };

        // Load inputs (Python saves as i64, need to cast to i32)
        let inputs_path = dir.join("all__inputs.npy");
        let inputs_i64 = <Array2<i64> as ReadNpyExt>::read_npy(File::open(&inputs_path)?)
            .map_err(|e| std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to read all__inputs.npy: {}", e)
            ))?;
        let inputs = inputs_i64.mapv(|x| x as i32);

        log::info!("Loaded inputs: shape {:?}", inputs.shape());

        // Load labels (Python saves as i64, need to cast to i32)
        let labels_path = dir.join("all__labels.npy");
        let labels_i64 = <Array2<i64> as ReadNpyExt>::read_npy(File::open(&labels_path)?)
            .map_err(|e| std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to read all__labels.npy: {}", e)
            ))?;
        let labels = labels_i64.mapv(|x| x as i32);

        log::info!("Loaded labels: shape {:?}", labels.shape());

        // Load puzzle identifiers (optional)
        let puzzle_ids_path = dir.join("all__puzzle_identifiers.npy");
        let puzzle_ids: Vec<i32> = if puzzle_ids_path.exists() {
            let ids = <Array1<i32> as ReadNpyExt>::read_npy(File::open(&puzzle_ids_path)?)
                .map_err(|e| std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Failed to read all__puzzle_identifiers.npy: {}", e)
                ))?;
            ids.to_vec()
        } else {
            log::warn!("all__puzzle_identifiers.npy not found, using empty vector");
            Vec::new()
        };

        // Validate shapes
        if inputs.shape() != labels.shape() {
            return Err(crate::TRMError::Config(format!(
                "Shape mismatch: inputs {:?} != labels {:?}",
                inputs.shape(),
                labels.shape()
            )));
        }

        let num_examples = inputs.nrows();
        let seq_len = inputs.ncols();

        log::info!(
            "Dataset loaded: {} examples, seq_len={}, vocab_size={}",
            num_examples,
            seq_len,
            metadata.vocab_size
        );

        Ok(Self {
            inputs,
            labels,
            puzzle_ids,
            metadata,
        })
    }

    /// Get number of examples
    pub fn len(&self) -> usize {
        self.inputs.nrows()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.inputs.nrows() == 0
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.metadata.vocab_size
    }

    /// Get sequence length
    pub fn seq_len(&self) -> usize {
        self.inputs.ncols()
    }

    /// Get metadata
    pub fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }

    /// Get input at index
    pub fn get_input(&self, idx: usize) -> ArrayView1<i32> {
        self.inputs.row(idx)
    }

    /// Get label at index
    pub fn get_label(&self, idx: usize) -> ArrayView1<i32> {
        self.labels.row(idx)
    }

    /// Get puzzle ID at index (if available)
    pub fn get_puzzle_id(&self, idx: usize) -> Option<i32> {
        if idx < self.puzzle_ids.len() {
            Some(self.puzzle_ids[idx])
        } else {
            None
        }
    }
}

/// Data loader for NumPy puzzle datasets
pub struct NumpyDataLoader {
    dataset: NumpyDataset,
    batch_size: usize,
    current_idx: usize,
    indices: Vec<usize>,
    shuffle: bool,
}

impl NumpyDataLoader {
    /// Create new data loader
    pub fn new(dataset: NumpyDataset, batch_size: usize, shuffle: bool) -> Self {
        let num_samples = dataset.len();
        let mut indices: Vec<usize> = (0..num_samples).collect();

        if shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);
        }

        Self {
            dataset,
            batch_size,
            current_idx: 0,
            indices,
            shuffle,
        }
    }

    /// Get next batch (input_ids, target_ids)
    pub fn next_batch(&mut self, device: &Device) -> Result<Option<(Tensor, Tensor)>> {
        if self.current_idx >= self.indices.len() {
            return Ok(None);
        }

        let end_idx = (self.current_idx + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.current_idx..end_idx];
        let actual_batch_size = batch_indices.len();

        // Collect sequences for this batch
        let mut input_data = Vec::new();
        let mut target_data = Vec::new();

        for &idx in batch_indices {
            let input = self.dataset.get_input(idx);
            let target = self.dataset.get_label(idx);

            // Convert i32 to u32 for Candle
            input_data.extend(input.iter().map(|&x| x as u32));
            target_data.extend(target.iter().map(|&x| x as u32));
        }

        self.current_idx = end_idx;

        // Convert to tensors
        let seq_len = self.dataset.seq_len();
        let input_tensor = Tensor::from_vec(
            input_data,
            (actual_batch_size, seq_len),
            device,
        )?.to_dtype(candle_core::DType::U32)?;

        let target_tensor = Tensor::from_vec(
            target_data,
            (actual_batch_size, seq_len),
            device,
        )?.to_dtype(candle_core::DType::U32)?;

        Ok(Some((input_tensor, target_tensor)))
    }

    /// Reset loader for new epoch
    pub fn reset(&mut self) {
        self.current_idx = 0;

        if self.shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            self.indices.shuffle(&mut rng);
        }
    }

    /// Get number of batches
    pub fn num_batches(&self) -> usize {
        (self.dataset.len() + self.batch_size - 1) / self.batch_size
    }

    /// Get dataset reference
    pub fn dataset(&self) -> &NumpyDataset {
        &self.dataset
    }
}

// Implement BatchDataLoader trait
impl super::BatchDataLoader for NumpyDataLoader {
    fn next_batch(&mut self, device: &Device) -> Result<Option<(Tensor, Tensor)>> {
        NumpyDataLoader::next_batch(self, device)
    }

    fn reset(&mut self) {
        NumpyDataLoader::reset(self)
    }

    fn num_batches(&self) -> usize {
        NumpyDataLoader::num_batches(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_deserialization() {
        let json = r#"{
            "vocab_size": 11,
            "seq_len": 81,
            "num_examples": 1000000,
            "description": "Sudoku-Extreme"
        }"#;

        let metadata: DatasetMetadata = serde_json::from_str(json).unwrap();
        assert_eq!(metadata.vocab_size, 11);
        assert_eq!(metadata.seq_len, 81);
        assert_eq!(metadata.num_examples, 1000000);
    }
}
