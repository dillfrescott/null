use serde::{Deserialize, Serialize};
use rand::Rng;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLP {
    // Layer 1: 8 inputs -> 12 hidden neurons
    pub w1: Vec<Vec<f32>>,
    pub b1: Vec<f32>,
    // Layer 2: 12 hidden -> 8 hidden neurons
    pub w2: Vec<Vec<f32>>,
    pub b2: Vec<f32>,
    // Layer 3: 8 hidden -> 1 output neuron
    pub w3: Vec<f32>,
    pub b3: f32,
}

impl MLP {
    pub fn new_random() -> Self {
        let mut rng = rand::thread_rng();
        
        let init_weight_matrix = |rng: &mut rand::rngs::ThreadRng, rows: usize, cols: usize| -> Vec<Vec<f32>> {
            let limit = (6.0 / (rows + cols) as f32).sqrt(); // Xavier initialization
            (0..rows)
                .map(|_| (0..cols).map(|_| rng.gen_range(-limit..limit)).collect())
                .collect()
        };

        let init_weight_vector = |rng: &mut rand::rngs::ThreadRng, size: usize| -> Vec<f32> {
            let limit = (6.0 / (size + 1) as f32).sqrt();
            (0..size).map(|_| rng.gen_range(-limit..limit)).collect()
        };

        MLP {
            w1: init_weight_matrix(&mut rng, 12, 8),
            b1: vec![0.0; 12],
            w2: init_weight_matrix(&mut rng, 8, 12),
            b2: vec![0.0; 8],
            w3: init_weight_vector(&mut rng, 8),
            b3: 0.0,
        }
    }

    // Sigmoid function
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    // Derivative of sigmoid (using sigmoid output value y)
    fn d_sigmoid(y: f32) -> f32 {
        y * (1.0 - y)
    }

    // ReLU activation
    fn relu(x: f32) -> f32 {
        x.max(0.0)
    }

    // Derivative of ReLU
    fn d_relu(x: f32) -> f32 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }

    // Forward pass: returns (hidden1_raw, hidden1_act, hidden2_raw, hidden2_act, output_raw, output_act)
    fn forward(&self, x: &[f32; 8]) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, f32, f32) {
        // Layer 1
        let mut h1_raw = vec![0.0; 12];
        let mut h1_act = vec![0.0; 12];
        for i in 0..12 {
            let mut sum = self.b1[i];
            for j in 0..8 {
                sum += self.w1[i][j] * x[j];
            }
            h1_raw[i] = sum;
            h1_act[i] = Self::relu(sum);
        }

        // Layer 2
        let mut h2_raw = vec![0.0; 8];
        let mut h2_act = vec![0.0; 8];
        for i in 0..8 {
            let mut sum = self.b2[i];
            for j in 0..12 {
                sum += self.w2[i][j] * h1_act[j];
            }
            h2_raw[i] = sum;
            h2_act[i] = Self::relu(sum);
        }

        // Layer 3
        let mut out_raw = self.b3;
        for j in 0..8 {
            out_raw += self.w3[j] * h2_act[j];
        }
        let out_act = Self::sigmoid(out_raw);

        (h1_raw, h1_act, h2_raw, h2_act, out_raw, out_act)
    }

    /// Predict the probability of the input being human.
    /// Returns a value between 0.0 (Bot) and 1.0 (Human).
    pub fn predict(&self, input: &[f32; 8]) -> f32 {
        let (_, _, _, _, _, output) = self.forward(input);
        output
    }

    /// Train the MLP on a dataset of (input, target) pairs.
    /// target: 1.0 for Human, 0.0 for Bot.
    pub fn train(&mut self, dataset: &[([f32; 8], f32)], epochs: usize, lr: f32) -> f32 {
        use rand::seq::SliceRandom;
        let mut total_loss = 0.0;
        let mut rng = rand::thread_rng();
        let mut shuffled_dataset = dataset.to_vec();

        // Initialize momentum velocities
        let mut v_w1 = vec![vec![0.0f32; 8]; 12];
        let mut v_b1 = vec![0.0f32; 12];
        let mut v_w2 = vec![vec![0.0f32; 12]; 8];
        let mut v_b2 = vec![0.0f32; 8];
        let mut v_w3 = vec![0.0f32; 8];
        let mut v_b3 = 0.0f32;

        let momentum = 0.9f32;
        let weight_decay = 0.0001f32; // L2 regularization coefficient

        for _epoch in 0..epochs {
            shuffled_dataset.shuffle(&mut rng);
            total_loss = 0.0;
            for (x, target) in &shuffled_dataset {
                // 1. Forward pass
                let (h1_raw, h1_act, h2_raw, h2_act, _out_raw, out_act) = self.forward(x);

                // Loss (Mean Squared Error)
                let error = out_act - target;
                total_loss += error * error;

                // 2. Backpropagation
                // Layer 3 gradients (output layer)
                let d_out = error * Self::d_sigmoid(out_act);

                // Layer 2 gradients
                let mut d_h2 = vec![0.0; 8];
                for j in 0..8 {
                    d_h2[j] = d_out * self.w3[j] * Self::d_relu(h2_raw[j]);
                }

                // Layer 1 gradients
                let mut d_h1 = vec![0.0; 12];
                for j in 0..12 {
                    let mut sum = 0.0;
                    for k in 0..8 {
                        sum += d_h2[k] * self.w2[k][j];
                    }
                    d_h1[j] = sum * Self::d_relu(h1_raw[j]);
                }

                // Update weights and biases (Gradient Descent with Momentum & Weight Decay)
                // Layer 3
                for j in 0..8 {
                    let grad = d_out * h2_act[j] + weight_decay * self.w3[j];
                    v_w3[j] = momentum * v_w3[j] + grad;
                    self.w3[j] -= lr * v_w3[j];
                }
                {
                    let grad = d_out;
                    v_b3 = momentum * v_b3 + grad;
                    self.b3 -= lr * v_b3;
                }

                // Layer 2
                for i in 0..8 {
                    for j in 0..12 {
                        let grad = d_h2[i] * h1_act[j] + weight_decay * self.w2[i][j];
                        v_w2[i][j] = momentum * v_w2[i][j] + grad;
                        self.w2[i][j] -= lr * v_w2[i][j];
                    }
                    let grad = d_h2[i];
                    v_b2[i] = momentum * v_b2[i] + grad;
                    self.b2[i] -= lr * v_b2[i];
                }

                // Layer 1
                for i in 0..12 {
                    for j in 0..8 {
                        let grad = d_h1[i] * x[j] + weight_decay * self.w1[i][j];
                        v_w1[i][j] = momentum * v_w1[i][j] + grad;
                        self.w1[i][j] -= lr * v_w1[i][j];
                    }
                    let grad = d_h1[i];
                    v_b1[i] = momentum * v_b1[i] + grad;
                    self.b1[i] -= lr * v_b1[i];
                }
            }
            total_loss /= dataset.len() as f32;
        }

        total_loss
    }

    /// Evaluate the model's classification accuracy on a dataset.
    pub fn validate(&self, dataset: &[([f32; 8], f32)]) -> f32 {
        if dataset.is_empty() {
            return 0.0;
        }
        let mut correct = 0;
        for (x, target) in dataset {
            let pred = self.predict(x);
            let pred_label = if pred >= 0.5 { 1.0 } else { 0.0 };
            if (pred_label - target).abs() < 0.01 {
                correct += 1;
            }
        }
        correct as f32 / dataset.len() as f32
    }

    /// Check if the weights and biases are sane (contain no NaN or infinite values).
    pub fn is_sane(&self) -> bool {
        let check_vec = |v: &[f32]| v.iter().all(|&x| x.is_finite());
        let check_matrix = |m: &[Vec<f32>]| m.iter().all(|row| check_vec(row));
        
        check_matrix(&self.w1)
            && check_vec(&self.b1)
            && check_matrix(&self.w2)
            && check_vec(&self.b2)
            && check_vec(&self.w3)
            && self.b3.is_finite()
    }


    /// Generates synthetic training data for humans and bots.
    /// This is used to initialize the model if no pre-trained weights exist.
    pub fn generate_synthetic_dataset() -> Vec<([f32; 8], f32)> {
        let mut rng = rand::thread_rng();
        let mut dataset = Vec::new();

        // 1. Generate Bot Features - Group A: Linear bots (target = 0.0)
        // Bots: straight lines, constant speeds, no jitter, no entropy.
        for _ in 0..500 {
            let straightness = rng.gen_range(0.98..1.0);
            let avg_speed = rng.gen_range(0.3..0.9);
            let speed_var = rng.gen_range(0.0..0.02); // very low variance
            let angular_jitter = rng.gen_range(0.0..0.01); // no jitter
            let total_duration = rng.gen_range(0.05..0.2); // very fast movements
            let line_deviation = rng.gen_range(0.0..0.01); // almost perfectly straight
            let point_count = rng.gen_range(0.1..0.4);
            let entropy = rng.gen_range(0.0..0.05);

            dataset.push(([
                straightness,
                avg_speed,
                speed_var,
                angular_jitter,
                total_duration,
                line_deviation,
                point_count,
                entropy,
            ], 0.0));
        }

        // 2. Generate Bot Features - Group B: Bezier / Curve bots (target = 0.0)
        // These bots use curved trajectories but lack human angular jitter and entropy.
        for _ in 0..500 {
            let straightness = rng.gen_range(0.7..0.95);
            let avg_speed = rng.gen_range(0.2..0.7);
            let speed_var = rng.gen_range(0.02..0.35); // some speed changes
            let angular_jitter = rng.gen_range(0.0..0.03); // extremely smooth path
            let total_duration = rng.gen_range(0.1..0.5);
            let line_deviation = rng.gen_range(0.02..0.15); // curved path
            let point_count = rng.gen_range(0.15..0.6);
            let entropy = rng.gen_range(0.0..0.08); // very low direction entropy

            dataset.push(([
                straightness,
                avg_speed,
                speed_var,
                angular_jitter,
                total_duration,
                line_deviation,
                point_count,
                entropy,
            ], 0.0));
        }

        // 3. Generate Human Features (target = 1.0)
        // Humans: curves, speed variations, pause adjustments, jitter, entropy.
        for _ in 0..1000 {
            let straightness = rng.gen_range(0.5..0.92); // curved path
            let avg_speed = rng.gen_range(0.1..0.6);
            let speed_var = rng.gen_range(0.15..0.7); // high speed variance (acceleration/deceleration)
            let angular_jitter = rng.gen_range(0.05..0.4); // some direction change jitter
            let total_duration = rng.gen_range(0.15..0.8); // slower overall duration
            let line_deviation = rng.gen_range(0.02..0.2); // deviates from the straight line
            let point_count = rng.gen_range(0.2..0.8);
            let entropy = rng.gen_range(0.15..0.65); // significant direction entropy

            dataset.push(([
                straightness,
                avg_speed,
                speed_var,
                angular_jitter,
                total_duration,
                line_deviation,
                point_count,
                entropy,
            ], 1.0));
        }

        // 4. Generate Human Slider Drag Features (target = 1.0)
        // Humans dragging a slider: straight path, but with human speed variations and slight wobble.
        for _ in 0..500 {
            let straightness = rng.gen_range(0.96..0.998); // very straight, but not mathematically perfect
            let avg_speed = rng.gen_range(0.15..0.6);
            let speed_var = rng.gen_range(0.12..0.5); // human acceleration/deceleration/adjustments
            let angular_jitter = rng.gen_range(0.01..0.08); // small wobble in y-axis
            let total_duration = rng.gen_range(0.2..1.0); // realistic time to drag
            let line_deviation = rng.gen_range(0.002..0.02); // very small perpendicular deviation
            let point_count = rng.gen_range(0.25..0.7);
            let entropy = rng.gen_range(0.03..0.2); // low but non-zero direction changes

            dataset.push(([
                straightness,
                avg_speed,
                speed_var,
                angular_jitter,
                total_duration,
                line_deviation,
                point_count,
                entropy,
            ], 1.0));
        }

        dataset
    }

    /// Load default trained weights. If training fails or is skipped, we can train on demand.
    pub fn new_default() -> Self {
        let mut model = Self::new_random();
        let dataset = Self::generate_synthetic_dataset();
        // Train it quickly (takes ~20ms)
        model.train(&dataset, 200, 0.05);
        model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp_validation_and_sanity() {
        let model = MLP::new_default();
        assert!(model.is_sane());
        
        let dataset = MLP::generate_synthetic_dataset();
        let accuracy = model.validate(&dataset);
        assert!(accuracy >= 0.90, "Accuracy was too low: {}", accuracy);
        
        // Corrupt model with NaN to test is_sane
        let mut corrupted_model = model;
        corrupted_model.b3 = f32::NAN;
        assert!(!corrupted_model.is_sane());
    }
}

