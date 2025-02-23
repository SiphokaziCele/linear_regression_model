use rand::Rng;
use rand::distr::Uniform;  // Fix for distributions


pub fn generate_data(samples: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = rand::thread_rng();
    let mut x_values = Vec::new();
    let mut y_values = Vec::new();

    for _ in 0..samples {
        let x = rng.sample(Uniform::new(-10.0, 10.0));  // ✅ Corrected
        let noise: f32 = rng.sample(Uniform::new(-1.0, 1.0));  // ✅ Corrected
        let y = 2.0 * x + 1.0 + noise;  // y = 2x + 1 + noise

        x_values.push(x);
        y_values.push(y);
    }

    (x_values, y_values)
}
