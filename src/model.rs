use serde::{Deserialize, Serialize};
use rand::Rng;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiniTransformer {
    // Input sequence size T = 8 (features), embedding dimension D = 16
    pub w_emb: [[f32; 16]; 8],
    pub e_pos: [[f32; 16]; 8],
    
    // Self-Attention projections (Q, K, V)
    pub w_q: [[f32; 16]; 16],
    pub b_q: [f32; 16],
    pub w_k: [[f32; 16]; 16],
    pub b_k: [f32; 16],
    pub w_v: [[f32; 16]; 16],
    pub b_v: [f32; 16],

    // Feed-Forward network (FFN) with D_FF = 32
    pub w1: [[f32; 32]; 16],
    pub b1: [f32; 32],
    pub w2: [[f32; 16]; 32],
    pub b2: [f32; 16],

    // Classification Head
    pub w_out: [f32; 16],
    pub b_out: f32,
}

#[allow(dead_code)]
struct TransformerForwardCache {
    x_emb: [[f32; 16]; 8],
    q: [[f32; 16]; 8],
    k: [[f32; 16]; 8],
    v: [[f32; 16]; 8],
    s: [[f32; 8]; 8],
    a: [[f32; 8]; 8],
    o: [[f32; 16]; 8],
    x_prime: [[f32; 16]; 8],
    h_raw: [[f32; 32]; 8],
    h_act: [[f32; 32]; 8],
    f: [[f32; 16]; 8],
    x_double_prime: [[f32; 16]; 8],
    p: [f32; 16],
    z: f32,
    y: f32,
}

impl MiniTransformer {
    pub fn new_random() -> Self {
        let mut rng = rand::thread_rng();
        
        let mut w_emb = [[0.0f32; 16]; 8];
        let mut e_pos = [[0.0f32; 16]; 8];
        let mut w_q = [[0.0f32; 16]; 16];
        let mut w_k = [[0.0f32; 16]; 16];
        let mut w_v = [[0.0f32; 16]; 16];
        let mut w1 = [[0.0f32; 32]; 16];
        let mut w2 = [[0.0f32; 16]; 32];
        let mut w_out = [0.0f32; 16];

        let mut fill_matrix = |mat: &mut [f32], rows: usize, cols: usize| {
            let limit = (6.0 / (rows + cols) as f32).sqrt();
            for val in mat.iter_mut() {
                *val = rng.gen_range(-limit..limit);
            }
        };

        for row in w_emb.iter_mut() {
            fill_matrix(row, 8, 16);
        }
        for row in e_pos.iter_mut() {
            fill_matrix(row, 8, 16);
        }
        for row in w_q.iter_mut() {
            fill_matrix(row, 16, 16);
        }
        for row in w_k.iter_mut() {
            fill_matrix(row, 16, 16);
        }
        for row in w_v.iter_mut() {
            fill_matrix(row, 16, 16);
        }
        for row in w1.iter_mut() {
            fill_matrix(row, 16, 32);
        }
        for row in w2.iter_mut() {
            fill_matrix(row, 32, 16);
        }
        
        let limit = (6.0f32 / (16.0f32 + 1.0f32)).sqrt();
        for val in w_out.iter_mut() {
            *val = rng.gen_range(-limit..limit);
        }

        MiniTransformer {
            w_emb,
            e_pos,
            w_q,
            b_q: [0.0; 16],
            w_k,
            b_k: [0.0; 16],
            w_v,
            b_v: [0.0; 16],
            w1,
            b1: [0.0; 32],
            w2,
            b2: [0.0; 16],
            w_out,
            b_out: 0.0,
        }
    }

    // Sigmoid function
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    // Forward pass
    fn forward(&self, x: &[f32; 8]) -> TransformerForwardCache {
        let t = 8;
        let d = 16;
        let d_ff = 32;

        // 1. Embeddings: X_i,j = x_i * w_emb_i,j + e_pos_i,j
        let mut x_emb = [[0.0f32; 16]; 8];
        for i in 0..t {
            for j in 0..d {
                x_emb[i][j] = x[i] * self.w_emb[i][j] + self.e_pos[i][j];
            }
        }

        // 2. Q, K, V Projections
        let mut q = [[0.0f32; 16]; 8];
        let mut k = [[0.0f32; 16]; 8];
        let mut v = [[0.0f32; 16]; 8];

        for i in 0..t {
            for j in 0..d {
                let mut q_sum = self.b_q[j];
                let mut k_sum = self.b_k[j];
                let mut v_sum = self.b_v[j];
                for m in 0..d {
                    q_sum += x_emb[i][m] * self.w_q[m][j];
                    k_sum += x_emb[i][m] * self.w_k[m][j];
                    v_sum += x_emb[i][m] * self.w_v[m][j];
                }
                q[i][j] = q_sum;
                k[i][j] = k_sum;
                v[i][j] = v_sum;
            }
        }

        // 3. Attention logits: S_i,j = (1 / sqrt(d)) * sum_m (Q_i,m * K_j,m)
        let scale = 1.0 / (d as f32).sqrt();
        let mut s = [[0.0f32; 8]; 8];
        for i in 0..t {
            for j in 0..t {
                let mut sum = 0.0f32;
                for m in 0..d {
                    sum += q[i][m] * k[j][m];
                }
                s[i][j] = sum * scale;
            }
        }

        // 4. Softmax over rows to get A
        let mut a = [[0.0f32; 8]; 8];
        for i in 0..t {
            let mut max_val = s[i][0];
            for j in 1..t {
                if s[i][j] > max_val {
                    max_val = s[i][j];
                }
            }
            let mut sum_exp = 0.0f32;
            let mut exps = [0.0f32; 8];
            for j in 0..t {
                let exp_val = (s[i][j] - max_val).exp();
                exps[j] = exp_val;
                sum_exp += exp_val;
            }
            for j in 0..t {
                a[i][j] = exps[j] / sum_exp;
            }
        }

        // 5. Attention output: O_i,j = sum_m (A_i,m * V_m,j)
        let mut o = [[0.0f32; 16]; 8];
        for i in 0..t {
            for j in 0..d {
                let mut sum = 0.0f32;
                for m in 0..t {
                    sum += a[i][m] * v[m][j];
                }
                o[i][j] = sum;
            }
        }

        // 6. First residual: X'_i,j = X_emb_i,j + O_i,j
        let mut x_prime = [[0.0f32; 16]; 8];
        for i in 0..t {
            for j in 0..d {
                x_prime[i][j] = x_emb[i][j] + o[i][j];
            }
        }

        // 7. FFN Layer 1 (ReLU): H_i,j = ReLU( b1_j + sum_m (X'_i,m * W1_m,j) )
        let mut h_raw = [[0.0f32; 32]; 8];
        let mut h_act = [[0.0f32; 32]; 8];
        for i in 0..t {
            for j in 0..d_ff {
                let mut sum = self.b1[j];
                for m in 0..d {
                    sum += x_prime[i][m] * self.w1[m][j];
                }
                h_raw[i][j] = sum;
                h_act[i][j] = sum.max(0.0);
            }
        }

        // 8. FFN Layer 2: F_i,j = b2_j + sum_m (H_i,m * W2_m,j)
        let mut f = [[0.0f32; 16]; 8];
        for i in 0..t {
            for j in 0..d {
                let mut sum = self.b2[j];
                for m in 0..d_ff {
                    sum += h_act[i][m] * self.w2[m][j];
                }
                f[i][j] = sum;
            }
        }

        // 9. Second residual: X''_i,j = X'_i,j + F_i,j
        let mut x_double_prime = [[0.0f32; 16]; 8];
        for i in 0..t {
            for j in 0..d {
                x_double_prime[i][j] = x_prime[i][j] + f[i][j];
            }
        }

        // 10. Average pooling: P_j = (1 / T) * sum_i (X''_i,j)
        let mut p = [0.0f32; 16];
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..t {
                sum += x_double_prime[i][j];
            }
            p[j] = sum / (t as f32);
        }

        // 11. Output layer: z = b_out + sum_j (P_j * W_out_j)
        let mut z = self.b_out;
        for j in 0..d {
            z += p[j] * self.w_out[j];
        }
        let y = Self::sigmoid(z);

        TransformerForwardCache {
            x_emb,
            q,
            k,
            v,
            s,
            a,
            o,
            x_prime,
            h_raw,
            h_act,
            f,
            x_double_prime,
            p,
            z,
            y,
        }
    }

    /// Predict the probability of the input being human.
    /// Returns a value between 0.0 (Bot) and 1.0 (Human).
    pub fn predict(&self, input: &[f32; 8]) -> f32 {
        let cache = self.forward(input);
        cache.y
    }

    /// Train the MiniTransformer on a dataset of (input, target) pairs.
    /// target: 1.0 for Human, 0.0 for Bot.
    pub fn train(&mut self, dataset: &[([f32; 8], f32)], epochs: usize, lr: f32) -> f32 {
        use rand::seq::SliceRandom;
        let mut total_loss = 0.0;
        let mut rng = rand::thread_rng();
        let mut shuffled_dataset = dataset.to_vec();

        let t = 8;
        let d = 16;
        let d_ff = 32;

        // Initialize momentum velocities
        let mut v_w_emb = [[0.0f32; 16]; 8];
        let mut v_e_pos = [[0.0f32; 16]; 8];
        let mut v_w_q = [[0.0f32; 16]; 16];
        let mut v_b_q = [0.0f32; 16];
        let mut v_w_k = [[0.0f32; 16]; 16];
        let mut v_b_k = [0.0f32; 16];
        let mut v_w_v = [[0.0f32; 16]; 16];
        let mut v_b_v = [0.0f32; 16];
        let mut v_w1 = [[0.0f32; 32]; 16];
        let mut v_b1 = [0.0f32; 32];
        let mut v_w2 = [[0.0f32; 16]; 32];
        let mut v_b2 = [0.0f32; 16];
        let mut v_w_out = [0.0f32; 16];
        let mut v_b_out = 0.0f32;

        let momentum = 0.9f32;
        let weight_decay = 0.0001f32;

        for _epoch in 0..epochs {
            shuffled_dataset.shuffle(&mut rng);
            total_loss = 0.0;
            for (x, target) in &shuffled_dataset {
                // 1. Forward pass
                let cache = self.forward(x);

                // Loss (MSE)
                let error = cache.y - target;
                total_loss += error * error;

                // 2. Backpropagation
                // Output layer
                let d_z = 2.0 * error * cache.y * (1.0 - cache.y);

                let d_b_out = d_z;
                let mut d_w_out = [0.0f32; 16];
                let mut d_p = [0.0f32; 16];
                for j in 0..d {
                    d_w_out[j] = d_z * cache.p[j];
                    d_p[j] = d_z * self.w_out[j];
                }

                // Average pooling
                let mut d_x_double_prime = [[0.0f32; 16]; 8];
                for i in 0..t {
                    for j in 0..d {
                        d_x_double_prime[i][j] = d_p[j] / (t as f32);
                    }
                }

                // Second residual connection
                let mut d_f = [[0.0f32; 16]; 8];
                let mut d_x_prime = [[0.0f32; 16]; 8];
                for i in 0..t {
                    for j in 0..d {
                        d_f[i][j] = d_x_double_prime[i][j];
                        d_x_prime[i][j] = d_x_double_prime[i][j];
                    }
                }

                // FFN Layer 2
                let mut d_b2 = [0.0f32; 16];
                let mut d_w2 = [[0.0f32; 16]; 32];
                let mut d_h_act = [[0.0f32; 32]; 8];
                for i in 0..t {
                    for j in 0..d {
                        d_b2[j] += d_f[i][j];
                        for m in 0..d_ff {
                            d_w2[m][j] += cache.h_act[i][m] * d_f[i][j];
                            d_h_act[i][m] += d_f[i][j] * self.w2[m][j];
                        }
                    }
                }

                // FFN Layer 1 (ReLU)
                let mut d_h_raw = [[0.0f32; 32]; 8];
                for i in 0..t {
                    for j in 0..d_ff {
                        if cache.h_raw[i][j] > 0.0 {
                            d_h_raw[i][j] = d_h_act[i][j];
                        }
                    }
                }

                // FFN Layer 1
                let mut d_b1 = [0.0f32; 32];
                let mut d_w1 = [[0.0f32; 32]; 16];
                for i in 0..t {
                    for j in 0..d_ff {
                        d_b1[j] += d_h_raw[i][j];
                        for m in 0..d {
                            d_w1[m][j] += cache.x_prime[i][m] * d_h_raw[i][j];
                            d_x_prime[i][m] += d_h_raw[i][j] * self.w1[m][j];
                        }
                    }
                }

                // First residual connection
                let mut d_o = [[0.0f32; 16]; 8];
                let mut d_x_emb = [[0.0f32; 16]; 8];
                for i in 0..t {
                    for j in 0..d {
                        d_o[i][j] = d_x_prime[i][j];
                        d_x_emb[i][j] = d_x_prime[i][j];
                    }
                }

                // Attention output
                let mut d_a = [[0.0f32; 8]; 8];
                let mut d_v = [[0.0f32; 16]; 8];
                for i in 0..t {
                    for j in 0..d {
                        for m in 0..t {
                            d_a[i][m] += d_o[i][j] * cache.v[m][j];
                            d_v[m][j] += d_o[i][j] * cache.a[i][m];
                        }
                    }
                }

                // Softmax
                let mut d_s = [[0.0f32; 8]; 8];
                for i in 0..t {
                    let mut sum_a_da = 0.0f32;
                    for l in 0..t {
                        sum_a_da += cache.a[i][l] * d_a[i][l];
                    }
                    for j in 0..t {
                        d_s[i][j] = cache.a[i][j] * (d_a[i][j] - sum_a_da);
                    }
                }

                // Attention logits
                let scale = 1.0 / (d as f32).sqrt();
                let mut d_q = [[0.0f32; 16]; 8];
                let mut d_k = [[0.0f32; 16]; 8];
                for i in 0..t {
                    for j in 0..t {
                        for m in 0..d {
                            d_q[i][m] += scale * d_s[i][j] * cache.k[j][m];
                            d_k[j][m] += scale * d_s[i][j] * cache.q[i][m];
                        }
                    }
                }

                // Q, K, V Projections
                let mut d_b_q = [0.0f32; 16];
                let mut d_w_q = [[0.0f32; 16]; 16];
                for i in 0..t {
                    for j in 0..d {
                        d_b_q[j] += d_q[i][j];
                        for m in 0..d {
                            d_w_q[m][j] += cache.x_emb[i][m] * d_q[i][j];
                            d_x_emb[i][m] += d_q[i][j] * self.w_q[m][j];
                        }
                    }
                }

                let mut d_b_k = [0.0f32; 16];
                let mut d_w_k = [[0.0f32; 16]; 16];
                for i in 0..t {
                    for j in 0..d {
                        d_b_k[j] += d_k[i][j];
                        for m in 0..d {
                            d_w_k[m][j] += cache.x_emb[i][m] * d_k[i][j];
                            d_x_emb[i][m] += d_k[i][j] * self.w_k[m][j];
                        }
                    }
                }

                let mut d_b_v = [0.0f32; 16];
                let mut d_w_v = [[0.0f32; 16]; 16];
                for i in 0..t {
                    for j in 0..d {
                        d_b_v[j] += d_v[i][j];
                        for m in 0..d {
                            d_w_v[m][j] += cache.x_emb[i][m] * d_v[i][j];
                            d_x_emb[i][m] += d_v[i][j] * self.w_v[m][j];
                        }
                    }
                }

                // Embedding layer
                let mut d_w_emb = [[0.0f32; 16]; 8];
                let mut d_e_pos = [[0.0f32; 16]; 8];
                for i in 0..t {
                    for j in 0..d {
                        d_e_pos[i][j] = d_x_emb[i][j];
                        d_w_emb[i][j] = d_x_emb[i][j] * x[i];
                    }
                }

                // Update weights and biases
                for i in 0..t {
                    for j in 0..d {
                        let grad = d_w_emb[i][j] + weight_decay * self.w_emb[i][j];
                        v_w_emb[i][j] = momentum * v_w_emb[i][j] + grad;
                        self.w_emb[i][j] -= lr * v_w_emb[i][j];
                    }
                }
                for i in 0..t {
                    for j in 0..d {
                        let grad = d_e_pos[i][j] + weight_decay * self.e_pos[i][j];
                        v_e_pos[i][j] = momentum * v_e_pos[i][j] + grad;
                        self.e_pos[i][j] -= lr * v_e_pos[i][j];
                    }
                }
                for i in 0..d {
                    for j in 0..d {
                        let grad = d_w_q[i][j] + weight_decay * self.w_q[i][j];
                        v_w_q[i][j] = momentum * v_w_q[i][j] + grad;
                        self.w_q[i][j] -= lr * v_w_q[i][j];
                    }
                    let grad = d_b_q[i];
                    v_b_q[i] = momentum * v_b_q[i] + grad;
                    self.b_q[i] -= lr * v_b_q[i];
                }
                for i in 0..d {
                    for j in 0..d {
                        let grad = d_w_k[i][j] + weight_decay * self.w_k[i][j];
                        v_w_k[i][j] = momentum * v_w_k[i][j] + grad;
                        self.w_k[i][j] -= lr * v_w_k[i][j];
                    }
                    let grad = d_b_k[i];
                    v_b_k[i] = momentum * v_b_k[i] + grad;
                    self.b_k[i] -= lr * v_b_k[i];
                }
                for i in 0..d {
                    for j in 0..d {
                        let grad = d_w_v[i][j] + weight_decay * self.w_v[i][j];
                        v_w_v[i][j] = momentum * v_w_v[i][j] + grad;
                        self.w_v[i][j] -= lr * v_w_v[i][j];
                    }
                    let grad = d_b_v[i];
                    v_b_v[i] = momentum * v_b_v[i] + grad;
                    self.b_v[i] -= lr * v_b_v[i];
                }
                for i in 0..d {
                    for j in 0..d_ff {
                        let grad = d_w1[i][j] + weight_decay * self.w1[i][j];
                        v_w1[i][j] = momentum * v_w1[i][j] + grad;
                        self.w1[i][j] -= lr * v_w1[i][j];
                    }
                }
                for j in 0..d_ff {
                    let grad = d_b1[j];
                    v_b1[j] = momentum * v_b1[j] + grad;
                    self.b1[j] -= lr * v_b1[j];
                }
                for i in 0..d_ff {
                    for j in 0..d {
                        let grad = d_w2[i][j] + weight_decay * self.w2[i][j];
                        v_w2[i][j] = momentum * v_w2[i][j] + grad;
                        self.w2[i][j] -= lr * v_w2[i][j];
                    }
                }
                for j in 0..d {
                    let grad = d_b2[j];
                    v_b2[j] = momentum * v_b2[j] + grad;
                    self.b2[j] -= lr * v_b2[j];
                }
                for j in 0..d {
                    let grad = d_w_out[j] + weight_decay * self.w_out[j];
                    v_w_out[j] = momentum * v_w_out[j] + grad;
                    self.w_out[j] -= lr * v_w_out[j];
                }
                {
                    let grad = d_b_out;
                    v_b_out = momentum * v_b_out + grad;
                    self.b_out -= lr * v_b_out;
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
        let check_val = |x: &f32| x.is_finite();
        self.w_emb.iter().flatten().all(check_val)
            && self.e_pos.iter().flatten().all(check_val)
            && self.w_q.iter().flatten().all(check_val)
            && self.b_q.iter().all(check_val)
            && self.w_k.iter().flatten().all(check_val)
            && self.b_k.iter().all(check_val)
            && self.w_v.iter().flatten().all(check_val)
            && self.b_v.iter().all(check_val)
            && self.w1.iter().flatten().all(check_val)
            && self.b1.iter().all(check_val)
            && self.w2.iter().flatten().all(check_val)
            && self.b2.iter().all(check_val)
            && self.w_out.iter().all(check_val)
            && self.b_out.is_finite()
    }

    /// Generates synthetic training data for humans and bots.
    pub fn generate_synthetic_dataset() -> Vec<([f32; 8], f32)> {
        let mut rng = rand::thread_rng();
        let mut dataset = Vec::new();

        // 1. Generate Bot Features - Group A: Linear bots (target = 0.0)
        for _ in 0..500 {
            let straightness = rng.gen_range(0.98..1.0);
            let avg_speed = rng.gen_range(0.3..0.9);
            let speed_var = rng.gen_range(0.0..0.02);
            let angular_jitter = rng.gen_range(0.0..0.01);
            let total_duration = rng.gen_range(0.05..0.2);
            let line_deviation = rng.gen_range(0.0..0.01);
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
        for _ in 0..500 {
            let straightness = rng.gen_range(0.7..0.95);
            let avg_speed = rng.gen_range(0.2..0.7);
            let speed_var = rng.gen_range(0.02..0.35);
            let angular_jitter = rng.gen_range(0.0..0.03);
            let total_duration = rng.gen_range(0.1..0.5);
            let line_deviation = rng.gen_range(0.02..0.15);
            let point_count = rng.gen_range(0.15..0.6);
            let entropy = rng.gen_range(0.0..0.08);

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
        for _ in 0..1000 {
            let straightness = rng.gen_range(0.5..0.92);
            let avg_speed = rng.gen_range(0.1..0.6);
            let speed_var = rng.gen_range(0.15..0.7);
            let angular_jitter = rng.gen_range(0.05..0.4);
            let total_duration = rng.gen_range(0.15..0.8);
            let line_deviation = rng.gen_range(0.02..0.2);
            let point_count = rng.gen_range(0.2..0.8);
            let entropy = rng.gen_range(0.15..0.65);

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
        for _ in 0..500 {
            let straightness = rng.gen_range(0.96..0.998);
            let avg_speed = rng.gen_range(0.15..0.6);
            let speed_var = rng.gen_range(0.12..0.5);
            let angular_jitter = rng.gen_range(0.01..0.08);
            let total_duration = rng.gen_range(0.2..1.0);
            let line_deviation = rng.gen_range(0.002..0.02);
            let point_count = rng.gen_range(0.25..0.7);
            let entropy = rng.gen_range(0.03..0.2);

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

    /// Load default trained weights.
    pub fn new_default() -> Self {
        let mut model = Self::new_random();
        let dataset = Self::generate_synthetic_dataset();
        model.train(&dataset, 25, 0.05);
        model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_validation_and_sanity() {
        let model = MiniTransformer::new_default();
        assert!(model.is_sane());
        
        let dataset = MiniTransformer::generate_synthetic_dataset();
        let accuracy = model.validate(&dataset);
        assert!(accuracy >= 0.90, "Accuracy was too low: {}", accuracy);
        
        // Corrupt model with NaN to test is_sane
        let mut corrupted_model = model;
        corrupted_model.b_out = f32::NAN;
        assert!(!corrupted_model.is_sane());
    }
}
