


use burn::tensor::{Tensor, TensorData, Device};  // Fix for Tensor creation
use burn_ndarray::NdArray;
use burn::optim::Adam;
use burn::prelude::*;
use crate::model::{LinearRegression, mean_squared_error};



pub fn train_model(model: &mut LinearRegression, x_train: Vec<f32>, y_train: Vec<f32>, epochs: usize) {
    let optimizer = Adam::new(0.01);  // ✅ Corrected (learning rate set to 0.01)
    // Adam optimizer

    for epoch in 0..epochs {
        let device = Device::<NdArray>::default();
        let x_tensor = Tensor::<NdArray, 1>::from_data(TensorData::from_vec(x_train.clone(), [x_train.len()]), &device);
        let y_tensor = Tensor::<NdArray, 1>::from_data(TensorData::from_vec(y_train.clone(), [y_train.len()]), &device);



        let predictions = model.forward(x_tensor.clone());
        let loss = mean_squared_error(&predictions, &y_tensor);

        let loss_tensor = Tensor::from_data(loss);
        let gradients = loss.clone().backward();  // ✅ Corrected
        optimizer.step(model, &gradients);

        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {}", epoch, loss.clone().into_data().to_vec()[0]);  // ✅ Corrected

        }
    }
}
