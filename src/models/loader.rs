/// Weight loading from safetensors files
use std::path::Path;
use candle_core::{Device, DType};
use candle_nn::VarBuilder;
use crate::TRMConfig;
use super::TinyRecursiveModel;

/// Load model from safetensors file
///
/// # Arguments
/// * `config` - Model configuration
/// * `weights_path` - Path to safetensors file
/// * `device` - Device to load model on
///
/// # Returns
/// Loaded TinyRecursiveModel
pub fn load_model<P: AsRef<Path>>(
    config: TRMConfig,
    weights_path: P,
    device: &Device,
) -> crate::Result<TinyRecursiveModel> {
    // Load weights using Candle's built-in safetensors support
    let dtype = DType::F32;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[weights_path.as_ref()],
            dtype,
            device,
        )?
    };

    TinyRecursiveModel::new(config, vb)
}
