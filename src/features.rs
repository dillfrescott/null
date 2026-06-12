use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryPoint {
    pub x: f32,
    pub y: f32,
    pub t: f64, // timestamp in milliseconds
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    pub straightness: f32,       // Straight-line distance / Path length
    pub avg_speed: f32,          // Average velocity (px/ms)
    pub speed_var: f32,          // Standard deviation of speed / Mean speed
    pub angular_jitter: f32,     // Sum of absolute changes in angle
    pub total_duration: f32,     // Duration in ms, normalized
    pub line_deviation: f32,     // Avg perpendicular distance to direct line
    pub point_count: f32,        // Number of points, normalized
    pub entropy: f32,            // Entropy of direction changes
}

impl FeatureVector {
    pub fn to_array(&self) -> [f32; 8] {
        [
            self.straightness,
            self.avg_speed,
            self.speed_var,
            self.angular_jitter,
            self.total_duration,
            self.line_deviation,
            self.point_count,
            self.entropy,
        ]
    }
}

pub fn extract_features(points: &[TelemetryPoint]) -> Option<FeatureVector> {
    if points.len() < 5 {
        return None;
    }

    // Sort points by timestamp just in case
    let mut points = points.to_vec();
    points.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));

    // Remove duplicates or points with duplicate timestamps
    let mut unique_points = Vec::new();
    for p in points {
        if unique_points.is_empty() {
            unique_points.push(p);
        } else {
            let last = unique_points.last().unwrap();
            // Allow small delta time
            if (p.t - last.t).abs() > 0.001 {
                unique_points.push(p);
            }
        }
    }

    if unique_points.len() < 5 {
        return None;
    }

    let n = unique_points.len();
    let start_p = &unique_points[0];
    let end_p = &unique_points[n - 1];

    let total_duration = (end_p.t - start_p.t) as f32; // ms
    if total_duration <= 0.0 {
        return None;
    }

    // Calculate path lengths and velocities
    let mut path_length = 0.0f32;
    let mut speeds = Vec::with_capacity(n - 1);
    let mut angles = Vec::with_capacity(n - 1);

    for i in 0..(n - 1) {
        let p1 = &unique_points[i];
        let p2 = &unique_points[i + 1];

        let dx = p2.x - p1.x;
        let dy = p2.y - p1.y;
        let dist = (dx * dx + dy * dy).sqrt();
        path_length += dist;

        let dt = (p2.t - p1.t) as f32;
        if dt > 0.0 {
            speeds.push(dist / dt);
        }
        
        let angle = dy.atan2(dx);
        angles.push(angle);
    }

    if path_length <= 0.0 {
        return Some(FeatureVector {
            straightness: 0.0,
            avg_speed: 0.0,
            speed_var: 0.0,
            angular_jitter: 0.0,
            total_duration: (total_duration / 2000.0).min(1.0),
            line_deviation: 0.0,
            point_count: (n as f32 / 50.0).min(1.0),
            entropy: 0.0,
        });
    }

    // 1. Straightness
    let dx_total = end_p.x - start_p.x;
    let dy_total = end_p.y - start_p.y;
    let straight_dist = (dx_total * dx_total + dy_total * dy_total).sqrt();
    let straightness = (straight_dist / path_length).min(1.0).max(0.0);

    // 2. Avg speed
    let avg_speed = if !speeds.is_empty() {
        speeds.iter().sum::<f32>() / speeds.len() as f32
    } else {
        0.0
    };

    // 3. Speed variance
    let speed_var = if speeds.len() > 1 && avg_speed > 0.0 {
        let variance = speeds.iter()
            .map(|&s| {
                let diff = s - avg_speed;
                diff * diff
            })
            .sum::<f32>() / (speeds.len() - 1) as f32;
        (variance.sqrt() / avg_speed).min(2.0) // Normalize/cap relative standard deviation
    } else {
        0.0
    };

    // 4. Angular Jitter
    let mut angular_jitter = 0.0f32;
    let mut angle_diffs = Vec::new();
    for i in 0..(angles.len() - 1) {
        let mut diff = angles[i + 1] - angles[i];
        // Normalize to [-pi, pi]
        while diff > std::f32::consts::PI {
            diff -= 2.0 * std::f32::consts::PI;
        }
        while diff < -std::f32::consts::PI {
            diff += 2.0 * std::f32::consts::PI;
        }
        angular_jitter += diff.abs();
        angle_diffs.push(diff.abs());
    }
    // Normalize angular jitter by points
    angular_jitter = if angles.len() > 1 {
        angular_jitter / (angles.len() - 1) as f32
    } else {
        0.0
    };

    // 5. Line deviation
    // Perpendicular distance of each point to the line connecting start and end
    let mut total_deviation = 0.0f32;
    if straight_dist > 0.0 {
        // Line equation ax + by + c = 0
        // for line through (x1,y1) and (x2,y2):
        // (y1 - y2)x + (x2 - x1)y + (x1*y2 - x2*y1) = 0
        let a = start_p.y - end_p.y;
        let b = end_p.x - start_p.x;
        let c = start_p.x * end_p.y - end_p.x * start_p.y;
        let denom = (a * a + b * b).sqrt();
        
        if denom > 0.0 {
            for p in &unique_points {
                let dist = (a * p.x + b * p.y + c).abs() / denom;
                total_deviation += dist;
            }
            total_deviation /= n as f32;
        }
    }
    // Normalize deviation relative to straight line distance
    let line_deviation = if straight_dist > 0.0 {
        (total_deviation / straight_dist).min(2.0)
    } else {
        0.0
    };

    // 6. Entropy of direction changes
    // Group angle differences into 8 bins and compute entropy
    let entropy = if !angle_diffs.is_empty() {
        let mut bins = [0; 8];
        for &diff in &angle_diffs {
            let bin_idx = ((diff / std::f32::consts::PI) * 8.0).floor() as usize;
            let bin_idx = bin_idx.min(7);
            bins[bin_idx] += 1;
        }
        let mut ent = 0.0f32;
        let total_diffs = angle_diffs.len() as f32;
        for &count in &bins {
            if count > 0 {
                let p = count as f32 / total_diffs;
                ent -= p * p.ln();
            }
        }
        // Normalize entropy by max possible entropy log(8) ~ 2.079
        (ent / 2.079).min(1.0).max(0.0)
    } else {
        0.0
    };

    Some(FeatureVector {
        straightness,
        avg_speed: (avg_speed / 5.0).min(1.0), // normalize: speed of 5px/ms is very fast
        speed_var,
        angular_jitter: (angular_jitter / std::f32::consts::PI).min(1.0), // normalize to [0, 1]
        total_duration: (total_duration / 2000.0).min(1.0),             // normalize relative to 2 seconds
        line_deviation,
        point_count: (n as f32 / 50.0).min(1.0),                         // normalize relative to 50 points
        entropy,
    })
}
