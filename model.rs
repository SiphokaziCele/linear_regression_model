use burn::prelude::*;  // Fix for Module
use burn::nn::{Linear};
use burn_ndarray::NdArray;



#[derive(Module, Debug)]
pub struct LinearRegression {
    layer: Linear<NdArray>,  // 1 input, 1 output
}

impl LinearRegression {
    pub fn new() -> Self {
        Self {
            layer: Linear::new(1, 1),  // ✅ Corrected
            // Initialize layer
        }
    }

    pub fn forward(&self, x: Tensor<NdArray, 1>) -> Tensor<NdArray, 1> {
        self.layer.forward(x)  // Compute y = wx + b
    }
}

fn mean_squared_error(predictions: &Tensor<NdArray, 1>, targets: &Tensor<NdArray, 1>) -> Tensor<NdArray, 1> {
    let diff = predictions.sub(targets);  // ✅ Corrected subtraction
    let squared_diff = diff.mul(&diff);  // ✅ Element-wise multiplication
    squared_diff.mean()  // ✅ Returns a tensor, not a scalar
}

