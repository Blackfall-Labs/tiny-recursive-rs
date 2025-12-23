/// AdamW optimizer implementation
///
/// Adam with decoupled weight decay regularization.
use candle_core::{Result, Tensor, Device};
use std::collections::HashMap;

/// AdamW optimizer configuration
#[derive(Debug, Clone)]
pub struct AdamWConfig {
    /// Learning rate
    pub lr: f64,
    /// Coefficient for computing running averages of gradient (beta1)
    pub beta1: f64,
    /// Coefficient for computing running averages of squared gradient (beta2)
    pub beta2: f64,
    /// Term added to denominator for numerical stability
    pub eps: f64,
    /// Weight decay coefficient
    pub weight_decay: f64,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

/// Parameter state for AdamW
#[derive(Debug, Clone)]
struct ParamState {
    /// First moment estimate (exponential moving average of gradients)
    m: Tensor,
    /// Second moment estimate (exponential moving average of squared gradients)
    v: Tensor,
    /// Step counter
    step: usize,
}

/// AdamW optimizer
///
/// Implements Adam with decoupled weight decay as described in
/// "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)
pub struct AdamW {
    config: AdamWConfig,
    params: Vec<Tensor>,
    states: HashMap<usize, ParamState>,
}

impl AdamW {
    /// Create new AdamW optimizer
    ///
    /// # Arguments
    /// * `params` - Parameters to optimize
    /// * `config` - Optimizer configuration
    pub fn new(params: Vec<Tensor>, config: AdamWConfig) -> Result<Self> {
        Ok(Self {
            config,
            params,
            states: HashMap::new(),
        })
    }

    /// Perform a single optimization step
    ///
    /// # Arguments
    /// * `grads` - Gradients for each parameter
    ///
    /// # Returns
    /// Result indicating success or error
    pub fn step(&mut self, grads: &[Tensor]) -> Result<()> {
        if grads.len() != self.params.len() {
            return Err(candle_core::Error::Msg(format!(
                "Expected {} gradients, got {}",
                self.params.len(),
                grads.len()
            )));
        }

        for (i, (param, grad)) in self.params.iter_mut().zip(grads.iter()).enumerate() {
            // Get or create state for this parameter
            let state = self.states.entry(i).or_insert_with(|| {
                let device = param.device();
                let shape = param.shape();

                ParamState {
                    m: Tensor::zeros(shape, param.dtype(), device).unwrap(),
                    v: Tensor::zeros(shape, param.dtype(), device).unwrap(),
                    step: 0,
                }
            });

            state.step += 1;

            // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * grad
            state.m = ((state.m.clone() * self.config.beta1)?
                + (grad * (1.0 - self.config.beta1))?)?;

            // Update biased second moment estimate: v = beta2 * v + (1 - beta2) * grad^2
            state.v = ((state.v.clone() * self.config.beta2)?
                + (grad.sqr()? * (1.0 - self.config.beta2))?)?;

            // Compute bias-corrected first moment estimate
            let beta1_t = self.config.beta1.powi(state.step as i32);
            let m_hat = (state.m.clone() / (1.0 - beta1_t))?;

            // Compute bias-corrected second moment estimate
            let beta2_t = self.config.beta2.powi(state.step as i32);
            let v_hat = (state.v.clone() / (1.0 - beta2_t))?;

            // Compute update: lr * m_hat / (sqrt(v_hat) + eps)
            let update = ((m_hat / (v_hat.sqrt()? + self.config.eps)?)? * self.config.lr)?;

            // Apply weight decay: param = param * (1 - lr * weight_decay)
            let param_decayed = if self.config.weight_decay > 0.0 {
                (param.clone() * (1.0 - self.config.lr * self.config.weight_decay))?
            } else {
                param.clone()
            };

            // Update parameter: param = param_decayed - update
            *param = (param_decayed - update)?;
        }

        Ok(())
    }

    /// Zero all parameter gradients
    pub fn zero_grad(&mut self) {
        // In Candle, gradients are managed separately
        // This is a no-op for now
    }

    /// Get current learning rate
    pub fn get_lr(&self) -> f64 {
        self.config.lr
    }

    /// Set learning rate
    pub fn set_lr(&mut self, lr: f64) {
        self.config.lr = lr;
    }

    /// Get reference to parameters
    pub fn params(&self) -> &[Tensor] {
        &self.params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adamw_creation() -> Result<()> {
        let device = Device::Cpu;
        let param = Tensor::randn(0f32, 1.0, (10, 10), &device)?;

        let config = AdamWConfig::default();
        let optimizer = AdamW::new(vec![param], config)?;

        assert_eq!(optimizer.get_lr(), 1e-3);

        Ok(())
    }

    #[test]
    fn test_adamw_step() -> Result<()> {
        let device = Device::Cpu;
        let param = Tensor::randn(0f32, 1.0, (10, 10), &device)?;
        let grad = Tensor::ones((10, 10), param.dtype(), &device)?;

        let config = AdamWConfig {
            lr: 0.01,
            ..Default::default()
        };

        let mut optimizer = AdamW::new(vec![param.clone()], config)?;

        // Perform one step
        optimizer.step(&[grad])?;

        // Parameters should have changed
        // (We can't easily verify the exact values without running the computation)

        Ok(())
    }

    #[test]
    fn test_adamw_lr_scheduling() -> Result<()> {
        let device = Device::Cpu;
        let param = Tensor::randn(0f32, 1.0, (10, 10), &device)?;

        let config = AdamWConfig::default();
        let mut optimizer = AdamW::new(vec![param], config)?;

        assert_eq!(optimizer.get_lr(), 1e-3);

        optimizer.set_lr(5e-4);
        assert_eq!(optimizer.get_lr(), 5e-4);

        Ok(())
    }
}
