/// Activation functions for TRM
///
/// Based on the Python implementation in layers.py
use candle_core::{Result, Tensor, DType, Device, Module};
use candle_nn::{VarBuilder, Linear, linear, Init};

/// Helper function to find the smallest multiple of b that is >= a
fn find_multiple(a: usize, b: usize) -> usize {
    ((a + b - 1) / b) * b
}

/// Linear layer with automatic dtype casting
///
/// Casts weights and bias to input dtype before computation.
/// Uses truncated normal initialization (approximated here with normal).
pub struct CastedLinear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl CastedLinear {
    /// Create new CastedLinear layer
    ///
    /// # Arguments
    /// * `in_features` - Input dimension
    /// * `out_features` - Output dimension
    /// * `bias` - Whether to include bias
    /// * `vb` - VarBuilder for parameter initialization
    pub fn new(
        in_features: usize,
        out_features: usize,
        bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Use Kaiming Normal initialization for weights (like candle-nn's Linear)
        let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
        let weight = vb.get_with_hints((out_features, in_features), "weight", init_ws)?;

        let bias = if bias {
            // Use uniform initialization for bias (like candle-nn's Linear)
            let bound = 1. / (in_features as f64).sqrt();
            let init_bs = Init::Uniform { lo: -bound, up: bound };
            Some(vb.get_with_hints(out_features, "bias", init_bs)?)
        } else {
            None
        };

        Ok(Self { weight, bias })
    }

    /// Forward pass with automatic dtype casting
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input_dtype = input.dtype();

        // Cast weight to input dtype
        let weight = if self.weight.dtype() != input_dtype {
            self.weight.to_dtype(input_dtype)?
        } else {
            self.weight.clone()
        };

        // Perform linear transformation: input @ weight^T
        // weight is [out_features, in_features], so weight^T is [in_features, out_features]
        let weight_t = weight.t()?;
        let output = input.broadcast_matmul(&weight_t)?;

        // Add bias if present
        if let Some(ref b) = self.bias {
            let bias = if b.dtype() != input_dtype {
                b.to_dtype(input_dtype)?
            } else {
                b.clone()
            };
            output.broadcast_add(&bias)
        } else {
            Ok(output)
        }
    }
}

/// SwiGLU activation: Swish-Gated Linear Unit
///
/// A gated activation function that combines SiLU (Swish) with gating.
/// Formula: down_proj(silu(gate) * up)
pub struct SwiGLU {
    gate_up_proj: CastedLinear,
    down_proj: CastedLinear,
}

impl SwiGLU {
    /// Create new SwiGLU layer
    ///
    /// # Arguments
    /// * `hidden_size` - Input/output dimension
    /// * `expansion` - Expansion factor for intermediate dimension
    /// * `vb` - VarBuilder for parameter initialization
    pub fn new(hidden_size: usize, expansion: f32, vb: VarBuilder) -> Result<Self> {
        // Calculate intermediate size and round to multiple of 256
        let inter = find_multiple(((expansion * hidden_size as f32 * 2.0 / 3.0).round() as usize), 256);

        let gate_up_proj = CastedLinear::new(
            hidden_size,
            inter * 2,
            false,
            vb.pp("gate_up_proj"),
        )?;

        let down_proj = CastedLinear::new(
            inter,
            hidden_size,
            false,
            vb.pp("down_proj"),
        )?;

        Ok(Self {
            gate_up_proj,
            down_proj,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Project to 2x intermediate size
        let gate_up = self.gate_up_proj.forward(x)?;

        // Split into gate and up
        let last_dim = gate_up.dims().len() - 1;
        let inter_size = gate_up.dim(last_dim)? / 2;

        let gate = gate_up.narrow(last_dim, 0, inter_size)?;
        let up = gate_up.narrow(last_dim, inter_size, inter_size)?;

        // Apply SiLU to gate and multiply with up
        let gate_activated = candle_nn::ops::silu(&gate)?;
        let gated = gate_activated.mul(&up)?;

        // Project back down
        self.down_proj.forward(&gated)
    }
}

/// LinearSwish activation
///
/// Combines linear transformation with SiLU (Swish) activation.
/// Can apply in either order based on `reverse` flag.
pub struct LinearSwish {
    linear: CastedLinear,
    reverse: bool,
}

impl LinearSwish {
    /// Create new LinearSwish layer
    ///
    /// # Arguments
    /// * `hidden_size` - Input/output dimension
    /// * `reverse` - If true: SiLU(Linear(x)), if false: Linear(SiLU(x))
    /// * `vb` - VarBuilder for parameter initialization
    pub fn new(hidden_size: usize, reverse: bool, vb: VarBuilder) -> Result<Self> {
        let linear = CastedLinear::new(
            hidden_size,
            hidden_size,
            false,
            vb.pp("linear"),
        )?;

        Ok(Self { linear, reverse })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.reverse {
            // SiLU(Linear(x))
            let linear_out = self.linear.forward(x)?;
            candle_nn::ops::silu(&linear_out)
        } else {
            // Linear(SiLU(x))
            let silu_out = candle_nn::ops::silu(x)?;
            self.linear.forward(&silu_out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    #[test]
    fn test_find_multiple() {
        assert_eq!(find_multiple(100, 256), 256);
        assert_eq!(find_multiple(300, 256), 512);
        assert_eq!(find_multiple(256, 256), 256);
        assert_eq!(find_multiple(1, 256), 256);
    }

    #[test]
    fn test_casted_linear_shape() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let linear = CastedLinear::new(64, 128, true, vb)?;

        let x = Tensor::randn(0f32, 1.0, (2, 16, 64), &device)?;
        let out = linear.forward(&x)?;

        assert_eq!(out.dims(), &[2, 16, 128]);

        Ok(())
    }

    #[test]
    fn test_swiglu_shape() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let swiglu = SwiGLU::new(256, 4.0, vb)?;

        let x = Tensor::randn(0f32, 1.0, (2, 16, 256), &device)?;
        let out = swiglu.forward(&x)?;

        // Output should have same shape as input
        assert_eq!(out.dims(), x.dims());

        Ok(())
    }

    #[test]
    fn test_linear_swish_shape() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let lin_swish = LinearSwish::new(256, false, vb)?;

        let x = Tensor::randn(0f32, 1.0, (2, 16, 256), &device)?;
        let out = lin_swish.forward(&x)?;

        // Output should have same shape as input
        assert_eq!(out.dims(), x.dims());

        Ok(())
    }
}
