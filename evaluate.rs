
use burn::tensor::{Tensor, TensorData, Device};
use burn_ndarray::NdArray;
use textplots::{Chart, Shape};

use crate::model::LinearRegression;



pub fn evaluate_model(model: &LinearRegression) {
    let x_test: Vec<f32> = (-10..10).map(|x| x as f32).collect();
    let device = Device::<NdArray>::default();
    let x_tensor = Tensor::<NdArray, 1>::from_data(TensorData::new(x_test.clone(), [x_test.len()]), &device);

    let predictions = model.forward(x_tensor).into_data().to_vec();

    println!("Testing Model Performance:");

    Chart::new(100, 40, -10.0, 10.0);

    let points: Vec<(f32, f32)> = x_test.iter().cloned().zip(predictions.iter().map(|y| y[0])).collect();

    Chart::new(100, 40, -10.0, 10.0)
        .lineplot(&Shape::Lines(&points))
        .display();


}
