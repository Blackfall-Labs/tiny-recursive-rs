/// Cosine learning rate scheduler with warmup
use std::f64::consts::PI;

/// Cosine annealing learning rate scheduler configuration
#[derive(Debug, Clone)]
pub struct CosineSchedulerConfig {
    /// Initial learning rate
    pub lr_init: f64,
    /// Minimum learning rate (at end of schedule)
    pub lr_min: f64,
    /// Number of warmup steps
    pub warmup_steps: usize,
    /// Total number of training steps
    pub total_steps: usize,
}

impl Default for CosineSchedulerConfig {
    fn default() -> Self {
        Self {
            lr_init: 1e-3,
            lr_min: 1e-5,
            warmup_steps: 1000,
            total_steps: 100000,
        }
    }
}

/// Cosine learning rate scheduler
///
/// Implements cosine annealing with linear warmup:
/// - Linear warmup from 0 to lr_init over warmup_steps
/// - Cosine annealing from lr_init to lr_min over remaining steps
pub struct CosineScheduler {
    config: CosineSchedulerConfig,
    current_step: usize,
}

impl CosineScheduler {
    /// Create new cosine scheduler
    pub fn new(config: CosineSchedulerConfig) -> Self {
        Self {
            config,
            current_step: 0,
        }
    }

    /// Get learning rate for current step
    pub fn get_lr(&self) -> f64 {
        self.get_lr_at_step(self.current_step)
    }

    /// Get learning rate for a specific step
    pub fn get_lr_at_step(&self, step: usize) -> f64 {
        if step < self.config.warmup_steps {
            // Linear warmup: lr = lr_init * (step / warmup_steps)
            self.config.lr_init * (step as f64 / self.config.warmup_steps as f64)
        } else {
            // Cosine annealing
            let progress = (step - self.config.warmup_steps) as f64
                / (self.config.total_steps - self.config.warmup_steps) as f64;

            // Clamp progress to [0, 1]
            let progress = progress.min(1.0).max(0.0);

            // Cosine annealing formula:
            // lr = lr_min + (lr_init - lr_min) * 0.5 * (1 + cos(π * progress))
            let cosine_factor = 0.5 * (1.0 + (PI * progress).cos());
            self.config.lr_min + (self.config.lr_init - self.config.lr_min) * cosine_factor
        }
    }

    /// Step the scheduler (increment step counter)
    pub fn step(&mut self) {
        self.current_step += 1;
    }

    /// Get current step
    pub fn get_step(&self) -> usize {
        self.current_step
    }

    /// Reset scheduler to initial state
    pub fn reset(&mut self) {
        self.current_step = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warmup_phase() {
        let config = CosineSchedulerConfig {
            lr_init: 1.0,
            lr_min: 0.0,
            warmup_steps: 100,
            total_steps: 1000,
        };

        let scheduler = CosineScheduler::new(config);

        // At step 0, lr should be 0
        assert!((scheduler.get_lr_at_step(0) - 0.0).abs() < 1e-6);

        // At step 50 (halfway through warmup), lr should be 0.5
        assert!((scheduler.get_lr_at_step(50) - 0.5).abs() < 1e-6);

        // At step 100 (end of warmup), lr should be 1.0
        assert!((scheduler.get_lr_at_step(100) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_annealing() {
        let config = CosineSchedulerConfig {
            lr_init: 1.0,
            lr_min: 0.0,
            warmup_steps: 0,
            total_steps: 1000,
        };

        let scheduler = CosineScheduler::new(config);

        // At step 0, lr should be lr_init
        assert!((scheduler.get_lr_at_step(0) - 1.0).abs() < 1e-6);

        // At step 500 (halfway), lr should be ~0.5
        let lr_mid = scheduler.get_lr_at_step(500);
        assert!((lr_mid - 0.5).abs() < 0.1);

        // At step 1000 (end), lr should be lr_min
        assert!((scheduler.get_lr_at_step(1000) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_scheduler_stepping() {
        let config = CosineSchedulerConfig {
            lr_init: 1.0,
            lr_min: 0.1,
            warmup_steps: 10,
            total_steps: 100,
        };

        let mut scheduler = CosineScheduler::new(config);

        assert_eq!(scheduler.get_step(), 0);

        scheduler.step();
        assert_eq!(scheduler.get_step(), 1);

        scheduler.step();
        assert_eq!(scheduler.get_step(), 2);

        // LR should be increasing during warmup
        let lr1 = scheduler.get_lr_at_step(5);
        let lr2 = scheduler.get_lr_at_step(8);
        assert!(lr2 > lr1);
    }

    #[test]
    fn test_reset() {
        let config = CosineSchedulerConfig::default();
        let mut scheduler = CosineScheduler::new(config);

        scheduler.step();
        scheduler.step();
        assert_eq!(scheduler.get_step(), 2);

        scheduler.reset();
        assert_eq!(scheduler.get_step(), 0);
    }

    #[test]
    fn test_lr_never_exceeds_init() {
        let config = CosineSchedulerConfig {
            lr_init: 1.0,
            lr_min: 0.1,
            warmup_steps: 100,
            total_steps: 1000,
        };

        let scheduler = CosineScheduler::new(config.clone());

        // Test that LR never exceeds lr_init
        for step in 0..=config.total_steps {
            let lr = scheduler.get_lr_at_step(step);
            assert!(lr <= config.lr_init + 1e-6, "LR {} exceeds max {} at step {}", lr, config.lr_init, step);
        }

        // After warmup, LR should be >= lr_min
        for step in config.warmup_steps..=config.total_steps {
            let lr = scheduler.get_lr_at_step(step);
            assert!(lr >= config.lr_min - 1e-6, "LR {} is below min {} at step {}", lr, config.lr_min, step);
        }
    }
}
