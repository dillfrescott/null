#![allow(clippy::needless_range_loop, clippy::type_complexity)]

use rand::Rng;
use serde::{Deserialize, Serialize};

const T: usize = 13;
const D: usize = 16;
const D_FF: usize = 32;
const NUM_HEADS: usize = 2;
const HEAD_DIM: usize = D / NUM_HEADS;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiniTransformer {
    // Input sequence size T = 13 (features), embedding dimension D = 16
    pub w_emb: [[f32; D]; T],
    pub e_pos: [[f32; D]; T],

    // Self-Attention projections (Q, K, V) -> D x D
    pub w_q: [[f32; D]; D],
    pub b_q: [f32; D],
    pub w_k: [[f32; D]; D],
    pub b_k: [f32; D],
    pub w_v: [[f32; D]; D],
    pub b_v: [f32; D],

    // Feed-Forward network (FFN) with D_FF = 32
    pub w1: [[f32; D_FF]; D],
    pub b1: [f32; D_FF],
    pub w2: [[f32; D]; D_FF],
    pub b2: [f32; D],

    // Classification Head
    pub w_out: [f32; D],
    pub b_out: f32,
}

#[allow(dead_code)]
struct TransformerForwardCache {
    x_emb: [[f32; D]; T],
    q: [[f32; D]; T],
    k: [[f32; D]; T],
    v: [[f32; D]; T],
    // Multi-head attention: 2 heads of HEAD_DIM
    s0: [[f32; T]; T],
    a0: [[f32; T]; T],
    o0: [[f32; HEAD_DIM]; T],
    s1: [[f32; T]; T],
    a1: [[f32; T]; T],
    o1: [[f32; HEAD_DIM]; T],
    o: [[f32; D]; T],
    x_prime: [[f32; D]; T],
    h_raw: [[f32; D_FF]; T],
    h_act: [[f32; D_FF]; T],
    f: [[f32; D]; T],
    x_double_prime: [[f32; D]; T],
    p: [f32; D],
    z: f32,
    y: f32,
}

impl MiniTransformer {
    pub fn new_random() -> Self {
        let mut rng = rand::thread_rng();
        let mut fill_matrix = |mat: &mut [f32], rows: usize, cols: usize| {
            let limit = (6.0 / (rows + cols) as f32).sqrt();
            for val in mat.iter_mut() {
                *val = rng.gen_range(-limit..limit);
            }
        };

        let mut w_emb = [[0.0f32; D]; T];
        let mut e_pos = [[0.0f32; D]; T];
        for row in w_emb.iter_mut() {
            fill_matrix(row, T, D);
        }
        for row in e_pos.iter_mut() {
            fill_matrix(row, T, D);
        }

        let mut w_q = [[0.0f32; D]; D];
        let mut w_k = [[0.0f32; D]; D];
        let mut w_v = [[0.0f32; D]; D];
        for row in w_q.iter_mut() {
            fill_matrix(row, D, D);
        }
        for row in w_k.iter_mut() {
            fill_matrix(row, D, D);
        }
        for row in w_v.iter_mut() {
            fill_matrix(row, D, D);
        }

        let mut w1 = [[0.0f32; D_FF]; D];
        let mut w2 = [[0.0f32; D]; D_FF];
        for row in w1.iter_mut() {
            fill_matrix(row, D, D_FF);
        }
        for row in w2.iter_mut() {
            fill_matrix(row, D_FF, D);
        }

        let mut w_out = [0.0f32; D];
        let limit = (6.0f32 / (D as f32 + 1.0f32)).sqrt();
        for val in w_out.iter_mut() {
            *val = rng.gen_range(-limit..limit);
        }

        MiniTransformer {
            w_emb,
            e_pos,
            w_q,
            b_q: [0.0; D],
            w_k,
            b_k: [0.0; D],
            w_v,
            b_v: [0.0; D],
            w1,
            b1: [0.0; D_FF],
            w2,
            b2: [0.0; D],
            w_out,
            b_out: 0.0,
        }
    }

    // Sigmoid function
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    // Softmax helper for a single row
    fn softmax_row(row: &[f32; T]) -> [f32; T] {
        let mut max_val = row[0];
        for j in 1..T {
            if row[j] > max_val {
                max_val = row[j];
            }
        }
        let mut exps = [0.0f32; T];
        let mut sum_exp = 0.0f32;
        for j in 0..T {
            let v = (row[j] - max_val).exp();
            exps[j] = v;
            sum_exp += v;
        }
        for j in 0..T {
            exps[j] /= sum_exp;
        }
        exps
    }

    // Attention sub-function for one head (head_offset: 0 or HEAD_DIM)
    // Given Qh, Kh, Vh each [T; HEAD_DIM], returns (s, a, o) where o is [T; HEAD_DIM]
    fn attention_head(
        q: &[[f32; D]; T],
        k: &[[f32; D]; T],
        v: &[[f32; D]; T],
        head_offset: usize,
    ) -> ([[f32; T]; T], [[f32; T]; T], [[f32; HEAD_DIM]; T]) {
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let mut s = [[0.0f32; T]; T];
        let mut a = [[0.0f32; T]; T];
        let mut o = [[0.0f32; HEAD_DIM]; T];

        for i in 0..T {
            for j in 0..T {
                let mut sum = 0.0f32;
                for m in 0..HEAD_DIM {
                    sum += q[i][head_offset + m] * k[j][head_offset + m];
                }
                s[i][j] = sum * scale;
            }
        }

        for i in 0..T {
            let soft = Self::softmax_row(&s[i]);
            a[i] = soft;
        }

        for i in 0..T {
            for j in 0..HEAD_DIM {
                let mut sum = 0.0f32;
                for m in 0..T {
                    sum += a[i][m] * v[m][head_offset + j];
                }
                o[i][j] = sum;
            }
        }

        (s, a, o)
    }

    // Forward pass
    fn forward(&self, x: &[f32; T]) -> TransformerForwardCache {
        // 1. Embeddings: X_i,j = x_i * w_emb_i,j + e_pos_i,j
        let mut x_emb = [[0.0f32; D]; T];
        for i in 0..T {
            for j in 0..D {
                x_emb[i][j] = x[i] * self.w_emb[i][j] + self.e_pos[i][j];
            }
        }

        // 2. Q, K, V Projections
        let mut q = [[0.0f32; D]; T];
        let mut k = [[0.0f32; D]; T];
        let mut v = [[0.0f32; D]; T];

        for i in 0..T {
            for j in 0..D {
                let mut q_sum = self.b_q[j];
                let mut k_sum = self.b_k[j];
                let mut v_sum = self.b_v[j];
                for m in 0..D {
                    q_sum += x_emb[i][m] * self.w_q[m][j];
                    k_sum += x_emb[i][m] * self.w_k[m][j];
                    v_sum += x_emb[i][m] * self.w_v[m][j];
                }
                q[i][j] = q_sum;
                k[i][j] = k_sum;
                v[i][j] = v_sum;
            }
        }

        // 3. Multi-head attention (2 heads of HEAD_DIM)
        let (s0, a0, o0) = Self::attention_head(&q, &k, &v, 0);
        let (s1, a1, o1) = Self::attention_head(&q, &k, &v, HEAD_DIM);

        // 4. Concatenate heads
        let mut o = [[0.0f32; D]; T];
        for i in 0..T {
            for j in 0..HEAD_DIM {
                o[i][j] = o0[i][j];
                o[i][HEAD_DIM + j] = o1[i][j];
            }
        }

        // 5. First residual: X'_i,j = X_emb_i,j + O_i,j
        let mut x_prime = [[0.0f32; D]; T];
        for i in 0..T {
            for j in 0..D {
                x_prime[i][j] = x_emb[i][j] + o[i][j];
            }
        }

        // 6. FFN Layer 1 (ReLU): H_i,j = ReLU( b1_j + sum_m (X'_i,m * W1_m,j) )
        let mut h_raw = [[0.0f32; D_FF]; T];
        let mut h_act = [[0.0f32; D_FF]; T];
        for i in 0..T {
            for j in 0..D_FF {
                let mut sum = self.b1[j];
                for m in 0..D {
                    sum += x_prime[i][m] * self.w1[m][j];
                }
                h_raw[i][j] = sum;
                h_act[i][j] = sum.max(0.0);
            }
        }

        // 7. FFN Layer 2: F_i,j = b2_j + sum_m (H_i,m * W2_m,j)
        let mut f = [[0.0f32; D]; T];
        for i in 0..T {
            for j in 0..D {
                let mut sum = self.b2[j];
                for m in 0..D_FF {
                    sum += h_act[i][m] * self.w2[m][j];
                }
                f[i][j] = sum;
            }
        }

        // 8. Second residual: X''_i,j = X'_i,j + F_i,j
        let mut x_double_prime = [[0.0f32; D]; T];
        for i in 0..T {
            for j in 0..D {
                x_double_prime[i][j] = x_prime[i][j] + f[i][j];
            }
        }

        // 9. Average pooling: P_j = (1 / T) * sum_i (X''_i,j)
        let mut p = [0.0f32; D];
        for j in 0..D {
            let mut sum = 0.0f32;
            for i in 0..T {
                sum += x_double_prime[i][j];
            }
            p[j] = sum / (T as f32);
        }

        // 10. Output layer: z = b_out + sum_j (P_j * W_out_j)
        let mut z = self.b_out;
        for j in 0..D {
            z += p[j] * self.w_out[j];
        }
        let y = Self::sigmoid(z);

        TransformerForwardCache {
            x_emb,
            q,
            k,
            v,
            s0,
            a0,
            o0,
            s1,
            a1,
            o1,
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
    pub fn predict(&self, input: &[f32; T]) -> f32 {
        let cache = self.forward(input);
        cache.y
    }

    /// Train the MiniTransformer on a dataset of (input, target) pairs.
    /// target: 1.0 for Human, 0.0 for Bot.
    pub fn train(&mut self, dataset: &[([f32; T], f32)], epochs: usize, lr: f32) -> f32 {
        if dataset.is_empty() || epochs == 0 || !lr.is_finite() || lr <= 0.0 {
            return 0.0;
        }

        use rand::seq::SliceRandom;
        let mut total_loss = 0.0;
        let mut rng = rand::thread_rng();
        let mut shuffled_dataset = dataset.to_vec();

        // Initialize momentum velocities
        let mut v_w_emb = [[0.0f32; D]; T];
        let mut v_e_pos = [[0.0f32; D]; T];
        let mut v_w_q = [[0.0f32; D]; D];
        let mut v_b_q = [0.0f32; D];
        let mut v_w_k = [[0.0f32; D]; D];
        let mut v_b_k = [0.0f32; D];
        let mut v_w_v = [[0.0f32; D]; D];
        let mut v_b_v = [0.0f32; D];
        let mut v_w1 = [[0.0f32; D_FF]; D];
        let mut v_b1 = [0.0f32; D_FF];
        let mut v_w2 = [[0.0f32; D]; D_FF];
        let mut v_b2 = [0.0f32; D];
        let mut v_w_out = [0.0f32; D];
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
                let mut d_w_out = [0.0f32; D];
                let mut d_p = [0.0f32; D];
                for j in 0..D {
                    d_w_out[j] = d_z * cache.p[j];
                    d_p[j] = d_z * self.w_out[j];
                }

                // Average pooling
                let mut d_x_double_prime = [[0.0f32; D]; T];
                for i in 0..T {
                    for j in 0..D {
                        d_x_double_prime[i][j] = d_p[j] / (T as f32);
                    }
                }

                // Second residual connection
                let mut d_f = [[0.0f32; D]; T];
                let mut d_x_prime = [[0.0f32; D]; T];
                for i in 0..T {
                    for j in 0..D {
                        d_f[i][j] = d_x_double_prime[i][j];
                        d_x_prime[i][j] = d_x_double_prime[i][j];
                    }
                }

                // FFN Layer 2
                let mut d_b2 = [0.0f32; D];
                let mut d_w2 = [[0.0f32; D]; D_FF];
                let mut d_h_act = [[0.0f32; D_FF]; T];
                for i in 0..T {
                    for j in 0..D {
                        d_b2[j] += d_f[i][j];
                        for m in 0..D_FF {
                            d_w2[m][j] += cache.h_act[i][m] * d_f[i][j];
                            d_h_act[i][m] += d_f[i][j] * self.w2[m][j];
                        }
                    }
                }

                // FFN Layer 1 (ReLU backward)
                let mut d_h_raw = [[0.0f32; D_FF]; T];
                for i in 0..T {
                    for j in 0..D_FF {
                        if cache.h_raw[i][j] > 0.0 {
                            d_h_raw[i][j] = d_h_act[i][j];
                        }
                    }
                }

                // FFN Layer 1 projection
                let mut d_b1 = [0.0f32; D_FF];
                let mut d_w1 = [[0.0f32; D_FF]; D];
                for i in 0..T {
                    for j in 0..D_FF {
                        d_b1[j] += d_h_raw[i][j];
                        for m in 0..D {
                            d_w1[m][j] += cache.x_prime[i][m] * d_h_raw[i][j];
                            d_x_prime[i][m] += d_h_raw[i][j] * self.w1[m][j];
                        }
                    }
                }

                // First residual connection
                let mut d_o = [[0.0f32; D]; T];
                let mut d_x_emb = [[0.0f32; D]; T];
                for i in 0..T {
                    for j in 0..D {
                        d_o[i][j] = d_x_prime[i][j];
                        d_x_emb[i][j] = d_x_prime[i][j];
                    }
                }

                // === Multi-head attention backprop ===
                // Split d_o into 2 heads
                let mut d_o0 = [[0.0f32; HEAD_DIM]; T];
                let mut d_o1 = [[0.0f32; HEAD_DIM]; T];
                for i in 0..T {
                    for j in 0..HEAD_DIM {
                        d_o0[i][j] = d_o[i][j];
                        d_o1[i][j] = d_o[i][HEAD_DIM + j];
                    }
                }

                // Accumulated gradients across heads for Q, K, V projections
                let mut d_q = [[0.0f32; D]; T];
                let mut d_k = [[0.0f32; D]; T];
                let mut d_v = [[0.0f32; D]; T];
                let scale = 1.0 / (HEAD_DIM as f32).sqrt();

                // Backprop through head 0
                {
                    // Attention output -> V grad
                    let mut d_a0 = [[0.0f32; T]; T];
                    for i in 0..T {
                        for j in 0..HEAD_DIM {
                            for m in 0..T {
                                d_a0[i][m] += d_o0[i][j] * cache.v[m][j];
                                d_v[m][j] += d_o0[i][j] * cache.a0[i][m];
                            }
                        }
                    }

                    // Softmax backward
                    let mut d_s0 = [[0.0f32; T]; T];
                    for i in 0..T {
                        let mut sum_a_da = 0.0f32;
                        for l in 0..T {
                            sum_a_da += cache.a0[i][l] * d_a0[i][l];
                        }
                        for j in 0..T {
                            d_s0[i][j] = cache.a0[i][j] * (d_a0[i][j] - sum_a_da);
                        }
                    }

                    // Attention logits -> Q, K
                    for i in 0..T {
                        for j in 0..T {
                            for m in 0..HEAD_DIM {
                                d_q[i][m] += scale * d_s0[i][j] * cache.k[j][m];
                                d_k[j][m] += scale * d_s0[i][j] * cache.q[i][m];
                            }
                        }
                    }
                }

                // Backprop through head 1
                {
                    let mut d_a1 = [[0.0f32; T]; T];
                    for i in 0..T {
                        for j in 0..HEAD_DIM {
                            for m in 0..T {
                                d_a1[i][m] += d_o1[i][j] * cache.v[m][HEAD_DIM + j];
                                d_v[m][HEAD_DIM + j] += d_o1[i][j] * cache.a1[i][m];
                            }
                        }
                    }

                    let mut d_s1 = [[0.0f32; T]; T];
                    for i in 0..T {
                        let mut sum_a_da = 0.0f32;
                        for l in 0..T {
                            sum_a_da += cache.a1[i][l] * d_a1[i][l];
                        }
                        for j in 0..T {
                            d_s1[i][j] = cache.a1[i][j] * (d_a1[i][j] - sum_a_da);
                        }
                    }

                    for i in 0..T {
                        for j in 0..T {
                            for m in 0..HEAD_DIM {
                                d_q[i][HEAD_DIM + m] +=
                                    scale * d_s1[i][j] * cache.k[j][HEAD_DIM + m];
                                d_k[j][HEAD_DIM + m] +=
                                    scale * d_s1[i][j] * cache.q[i][HEAD_DIM + m];
                            }
                        }
                    }
                }

                // Q, K, V Projections backprop
                let mut d_b_q = [0.0f32; D];
                let mut d_w_q = [[0.0f32; D]; D];
                for i in 0..T {
                    for j in 0..D {
                        d_b_q[j] += d_q[i][j];
                        for m in 0..D {
                            d_w_q[m][j] += cache.x_emb[i][m] * d_q[i][j];
                            d_x_emb[i][m] += d_q[i][j] * self.w_q[m][j];
                        }
                    }
                }

                let mut d_b_k = [0.0f32; D];
                let mut d_w_k = [[0.0f32; D]; D];
                for i in 0..T {
                    for j in 0..D {
                        d_b_k[j] += d_k[i][j];
                        for m in 0..D {
                            d_w_k[m][j] += cache.x_emb[i][m] * d_k[i][j];
                            d_x_emb[i][m] += d_k[i][j] * self.w_k[m][j];
                        }
                    }
                }

                let mut d_b_v = [0.0f32; D];
                let mut d_w_v = [[0.0f32; D]; D];
                for i in 0..T {
                    for j in 0..D {
                        d_b_v[j] += d_v[i][j];
                        for m in 0..D {
                            d_w_v[m][j] += cache.x_emb[i][m] * d_v[i][j];
                            d_x_emb[i][m] += d_v[i][j] * self.w_v[m][j];
                        }
                    }
                }

                // Embedding layer
                let mut d_w_emb = [[0.0f32; D]; T];
                let mut d_e_pos = [[0.0f32; D]; T];
                for i in 0..T {
                    for j in 0..D {
                        d_e_pos[i][j] = d_x_emb[i][j];
                        d_w_emb[i][j] = d_x_emb[i][j] * x[i];
                    }
                }

                // Update weights and biases with SGD + momentum + weight decay
                for i in 0..T {
                    for j in 0..D {
                        let grad = d_w_emb[i][j] + weight_decay * self.w_emb[i][j];
                        v_w_emb[i][j] = momentum * v_w_emb[i][j] + grad;
                        self.w_emb[i][j] -= lr * v_w_emb[i][j];
                    }
                }
                for i in 0..T {
                    for j in 0..D {
                        let grad = d_e_pos[i][j] + weight_decay * self.e_pos[i][j];
                        v_e_pos[i][j] = momentum * v_e_pos[i][j] + grad;
                        self.e_pos[i][j] -= lr * v_e_pos[i][j];
                    }
                }
                for i in 0..D {
                    for j in 0..D {
                        let grad = d_w_q[i][j] + weight_decay * self.w_q[i][j];
                        v_w_q[i][j] = momentum * v_w_q[i][j] + grad;
                        self.w_q[i][j] -= lr * v_w_q[i][j];
                    }
                    let grad = d_b_q[i];
                    v_b_q[i] = momentum * v_b_q[i] + grad;
                    self.b_q[i] -= lr * v_b_q[i];
                }
                for i in 0..D {
                    for j in 0..D {
                        let grad = d_w_k[i][j] + weight_decay * self.w_k[i][j];
                        v_w_k[i][j] = momentum * v_w_k[i][j] + grad;
                        self.w_k[i][j] -= lr * v_w_k[i][j];
                    }
                    let grad = d_b_k[i];
                    v_b_k[i] = momentum * v_b_k[i] + grad;
                    self.b_k[i] -= lr * v_b_k[i];
                }
                for i in 0..D {
                    for j in 0..D {
                        let grad = d_w_v[i][j] + weight_decay * self.w_v[i][j];
                        v_w_v[i][j] = momentum * v_w_v[i][j] + grad;
                        self.w_v[i][j] -= lr * v_w_v[i][j];
                    }
                    let grad = d_b_v[i];
                    v_b_v[i] = momentum * v_b_v[i] + grad;
                    self.b_v[i] -= lr * v_b_v[i];
                }
                for i in 0..D {
                    for j in 0..D_FF {
                        let grad = d_w1[i][j] + weight_decay * self.w1[i][j];
                        v_w1[i][j] = momentum * v_w1[i][j] + grad;
                        self.w1[i][j] -= lr * v_w1[i][j];
                    }
                }
                for j in 0..D_FF {
                    let grad = d_b1[j];
                    v_b1[j] = momentum * v_b1[j] + grad;
                    self.b1[j] -= lr * v_b1[j];
                }
                for i in 0..D_FF {
                    for j in 0..D {
                        let grad = d_w2[i][j] + weight_decay * self.w2[i][j];
                        v_w2[i][j] = momentum * v_w2[i][j] + grad;
                        self.w2[i][j] -= lr * v_w2[i][j];
                    }
                }
                for j in 0..D {
                    let grad = d_b2[j];
                    v_b2[j] = momentum * v_b2[j] + grad;
                    self.b2[j] -= lr * v_b2[j];
                }
                for j in 0..D {
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
    pub fn validate(&self, dataset: &[([f32; T], f32)]) -> f32 {
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
    /// Includes adversarial bot samples that try to mimic human movement patterns.
    pub fn generate_synthetic_dataset() -> Vec<([f32; T], f32)> {
        let mut rng = rand::thread_rng();
        let mut dataset = Vec::new();

        // 1. Generate Bot Features - Group A: Naive linear bots (target = 0.0)
        for _ in 0..400 {
            dataset.push((
                [
                    rng.gen_range(0.98..1.0),
                    rng.gen_range(0.3..0.9),
                    rng.gen_range(0.0..0.02),
                    rng.gen_range(0.0..0.01),
                    rng.gen_range(0.05..0.2),
                    rng.gen_range(0.0..0.01),
                    rng.gen_range(0.1..0.4),
                    rng.gen_range(0.0..0.05),
                    rng.gen_range(0.0..0.03),
                    rng.gen_range(0.0..0.01),
                    rng.gen_range(0.0..0.01),
                    rng.gen_range(0.0..0.02),
                    rng.gen_range(0.0..0.05),
                ],
                0.0,
            ));
        }

        // 2. Generate Bot Features - Group B: Bezier / Curve bots (target = 0.0)
        for _ in 0..400 {
            dataset.push((
                [
                    rng.gen_range(0.7..0.95),
                    rng.gen_range(0.2..0.7),
                    rng.gen_range(0.02..0.35),
                    rng.gen_range(0.0..0.03),
                    rng.gen_range(0.1..0.5),
                    rng.gen_range(0.02..0.15),
                    rng.gen_range(0.15..0.6),
                    rng.gen_range(0.0..0.08),
                    rng.gen_range(0.01..0.2),
                    rng.gen_range(0.0..0.03),
                    rng.gen_range(0.0..0.02),
                    rng.gen_range(0.0..0.04),
                    rng.gen_range(0.0..0.08),
                ],
                0.0,
            ));
        }

        // 3. Generate Bot Features - Group C: Adversarial bots mimicking humans (target = 0.0)
        // These bots add human-like noise, jitter, and speed variance to evade simple detection.
        for _ in 0..600 {
            let straightness = rng.gen_range(0.5..0.94);
            let avg_speed = rng.gen_range(0.1..0.6);
            let speed_var = rng.gen_range(0.08..0.5);
            let angular_jitter = rng.gen_range(0.02..0.2);
            let total_duration = rng.gen_range(0.15..0.7);
            let line_deviation = rng.gen_range(0.01..0.15);
            let point_count = rng.gen_range(0.2..0.7);
            let entropy = rng.gen_range(0.05..0.3);
            // Adversarial bots have lower accel variance and timing jitter
            let accel_var = rng.gen_range(0.02..0.2);
            let curvature_change = rng.gen_range(0.01..0.08);
            let overshoot = rng.gen_range(0.0..0.02);
            let dwell_ratio = rng.gen_range(0.01..0.06);
            let timing_jitter = rng.gen_range(0.02..0.1);

            dataset.push((
                [
                    straightness,
                    avg_speed,
                    speed_var,
                    angular_jitter,
                    total_duration,
                    line_deviation,
                    point_count,
                    entropy,
                    accel_var,
                    curvature_change,
                    overshoot,
                    dwell_ratio,
                    timing_jitter,
                ],
                0.0,
            ));
        }

        // 4. Generate Bot Features - Group D: Replay/segmented bots (target = 0.0)
        // Bots that replay recorded human paths but with robotic timing.
        for _ in 0..200 {
            dataset.push((
                [
                    rng.gen_range(0.6..0.96),
                    rng.gen_range(0.2..0.7),
                    rng.gen_range(0.05..0.3),
                    rng.gen_range(0.02..0.15),
                    rng.gen_range(0.1..0.6),
                    rng.gen_range(0.01..0.12),
                    rng.gen_range(0.2..0.6),
                    rng.gen_range(0.05..0.25),
                    rng.gen_range(0.01..0.08), // low accel variance
                    rng.gen_range(0.01..0.06),
                    rng.gen_range(0.0..0.01),
                    rng.gen_range(0.0..0.03),  // low dwell
                    rng.gen_range(0.01..0.06), // low timing jitter
                ],
                0.0,
            ));
        }

        // 5. Generate Human Features (target = 1.0)
        for _ in 0..1000 {
            let straightness = rng.gen_range(0.5..0.92);
            let avg_speed = rng.gen_range(0.1..0.6);
            let speed_var = rng.gen_range(0.15..0.7);
            let angular_jitter = rng.gen_range(0.05..0.4);
            let total_duration = rng.gen_range(0.15..0.8);
            let line_deviation = rng.gen_range(0.02..0.2);
            let point_count = rng.gen_range(0.2..0.8);
            let entropy = rng.gen_range(0.15..0.65);
            // Human features show higher acceleration variance, curvature changes,
            // more overshoot, more dwell time, and higher timing jitter
            let accel_var = rng.gen_range(0.08..0.6);
            let curvature_change = rng.gen_range(0.03..0.3);
            let overshoot = rng.gen_range(0.0..0.15);
            let dwell_ratio = rng.gen_range(0.02..0.25);
            let timing_jitter = rng.gen_range(0.05..0.5);

            dataset.push((
                [
                    straightness,
                    avg_speed,
                    speed_var,
                    angular_jitter,
                    total_duration,
                    line_deviation,
                    point_count,
                    entropy,
                    accel_var,
                    curvature_change,
                    overshoot,
                    dwell_ratio,
                    timing_jitter,
                ],
                1.0,
            ));
        }

        // 6. Generate Human Slider Drag Features (target = 1.0)
        for _ in 0..500 {
            let straightness = rng.gen_range(0.96..0.998);
            let avg_speed = rng.gen_range(0.15..0.6);
            let speed_var = rng.gen_range(0.12..0.5);
            let angular_jitter = rng.gen_range(0.01..0.08);
            let total_duration = rng.gen_range(0.2..1.0);
            let line_deviation = rng.gen_range(0.002..0.02);
            let point_count = rng.gen_range(0.25..0.7);
            let entropy = rng.gen_range(0.03..0.2);
            // Slider drags have moderate accel variance, smooth curvature, some overshoot
            let accel_var = rng.gen_range(0.06..0.4);
            let curvature_change = rng.gen_range(0.01..0.1);
            let overshoot = rng.gen_range(0.0..0.08);
            let dwell_ratio = rng.gen_range(0.01..0.08);
            let timing_jitter = rng.gen_range(0.03..0.2);

            dataset.push((
                [
                    straightness,
                    avg_speed,
                    speed_var,
                    angular_jitter,
                    total_duration,
                    line_deviation,
                    point_count,
                    entropy,
                    accel_var,
                    curvature_change,
                    overshoot,
                    dwell_ratio,
                    timing_jitter,
                ],
                1.0,
            ));
        }

        dataset
    }

    /// Build a validated baseline model. Random initialization occasionally
    /// converges poorly, so reject weak candidates instead of serving them.
    pub fn new_default() -> Self {
        let validation = Self::generate_synthetic_dataset();
        let mut best: Option<(Self, f32)> = None;

        for _ in 0..3 {
            let mut candidate = Self::new_random();
            let training = Self::generate_synthetic_dataset();
            candidate.train(&training, 35, 0.01);
            let accuracy = candidate.validate(&validation);

            if candidate.is_sane() && accuracy >= 0.90 {
                return candidate;
            }
            if candidate.is_sane()
                && best
                    .as_ref()
                    .is_none_or(|(_, best_accuracy)| accuracy > *best_accuracy)
            {
                best = Some((candidate, accuracy));
            }
        }

        best.map(|(model, _)| model)
            .expect("model training produced no numerically sane candidate")
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
