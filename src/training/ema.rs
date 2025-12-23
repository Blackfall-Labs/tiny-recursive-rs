/// Exponential Moving Average for model weights
///
/// Maintains a moving average of model parameters for improved stability
/// and generalization during training.
use candle_core::{Result, Tensor};
use std::collections::HashMap;

/// EMA configuration
#[derive(Debug, Clone)]
pub struct EMAConfig {
    /// Decay rate for exponential moving average
    /// EMA_weight = decay * EMA_weight + (1 - decay) * weight
    pub decay: f64,
}

impl Default for EMAConfig {
    fn default() -> Self {
        Self {
            decay: 0.9999, // Common value for model EMA
        }
    }
}

/// Exponential Moving Average
///
/// Maintains shadow copies of model parameters that are updated
/// with exponential moving average.
pub struct EMA {
    config: EMAConfig,
    shadow_params: HashMap<usize, Tensor>,
}

impl EMA {
    /// Create new EMA
    pub fn new(config: EMAConfig) -> Self {
        Self {
            config,
            shadow_params: HashMap::new(),
        }
    }

    /// Update EMA parameters
    ///
    /// # Arguments
    /// * `params` - Current model parameters
    ///
    /// # Returns
    /// Result indicating success or error
    pub fn update(&mut self, params: &[Tensor]) -> Result<()> {
        for (i, param) in params.iter().enumerate() {
            // Get or create shadow parameter
            let shadow = self.shadow_params.entry(i).or_insert_with(|| {
                // Initialize shadow to current parameter value
                param.clone()
            });

            // Update: shadow = decay * shadow + (1 - decay) * param
            *shadow = ((shadow.clone() * self.config.decay)?
                + (param * (1.0 - self.config.decay))?)?;
        }

        Ok(())
    }

    /// Get EMA parameters
    ///
    /// # Returns
    /// Vector of EMA'd parameters
    pub fn get_params(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for i in 0..self.shadow_params.len() {
            if let Some(shadow) = self.shadow_params.get(&i) {
                params.push(shadow.clone());
            }
        }
        params
    }

    /// Copy EMA parameters to model
    ///
    /// # Arguments
    /// * `params` - Model parameters to update
    pub fn copy_to(&self, params: &mut [Tensor]) -> Result<()> {
        for (i, param) in params.iter_mut().enumerate() {
            if let Some(shadow) = self.shadow_params.get(&i) {
                *param = shadow.clone();
            }
        }
        Ok(())
    }

    /// Copy model parameters to EMA
    ///
    /// # Arguments
    /// * `params` - Model parameters to copy from
    pub fn copy_from(&mut self, params: &[Tensor]) {
        for (i, param) in params.iter().enumerate() {
            self.shadow_params.insert(i, param.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_ema_creation() {
        let config = EMAConfig::default();
        let ema = EMA::new(config);

        assert_eq!(ema.shadow_params.len(), 0);
    }

    #[test]
    fn test_ema_update() -> Result<()> {
        let device = Device::Cpu;
        let param = Tensor::ones((10, 10), candle_core::DType::F32, &device)?;

        let config = EMAConfig { decay: 0.9 };
        let mut ema = EMA::new(config);

        // First update initializes shadow
        ema.update(&[param.clone()])?;

        // Shadow should be initialized to param value
        let shadow = &ema.shadow_params[&0];
        let diff = (shadow.clone() - param.clone())?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-6);

        Ok(())
    }

    #[test]
    fn test_ema_smoothing() -> Result<()> {
        let device = Device::Cpu;

        let config = EMAConfig { decay: 0.9 };
        let mut ema = EMA::new(config);

        // Start with ones
        let param1 = Tensor::ones((5, 5), candle_core::DType::F32, &device)?;
        ema.update(&[param1.clone()])?;

        // Update with zeros - EMA should be between 0 and 1
        let param2 = Tensor::zeros((5, 5), candle_core::DType::F32, &device)?;
        ema.update(&[param2.clone()])?;

        let shadow = &ema.shadow_params[&0];
        let mean_val = shadow.mean_all()?.to_scalar::<f32>()?;

        // Should be decay * 1 + (1 - decay) * 0 = 0.9
        assert!((mean_val - 0.9).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_copy_to() -> Result<()> {
        let device = Device::Cpu;

        let config = EMAConfig { decay: 0.95 };
        let mut ema = EMA::new(config);

        // Initialize EMA
        let param = Tensor::ones((5, 5), candle_core::DType::F32, &device)?;
        ema.update(&[param.clone()])?;

        // Update EMA
        let param2 = Tensor::zeros((5, 5), candle_core::DType::F32, &device)?;
        ema.update(&[param2.clone()])?;

        // Copy EMA back to params
        let mut params = vec![Tensor::ones((5, 5), candle_core::DType::F32, &device)?];
        ema.copy_to(&mut params)?;

        // Params should now match EMA shadow
        let expected = 0.95; // decay * 1 + (1 - decay) * 0
        let actual = params[0].mean_all()?.to_scalar::<f32>()?;
        assert!((actual - expected).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_copy_from() -> Result<()> {
        let device = Device::Cpu;

        let config = EMAConfig::default();
        let mut ema = EMA::new(config);

        // Create params
        let param = Tensor::full(2.0f32, (5, 5), &device)?;

        // Copy params to EMA
        ema.copy_from(&[param.clone()]);

        // Shadow should match param
        let shadow = &ema.shadow_params[&0];
        let diff = (shadow.clone() - param)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-6);

        Ok(())
    }
}
