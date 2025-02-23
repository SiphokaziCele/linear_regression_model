# linear_regression_model
Introduction

This project implements a simple linear regression model using the Rust programming language and the Burn library. The goal is to predict the function y = 2x + 1 using synthetic data. The model is trained with added noise to simulate real-world conditions. The project follows the requirements specified in the assignment and includes necessary dependencies.

Description of the Approach

Setup Instructions

1. Install Rust

Download and install Rust from Rust's official website.

Verify the installation by running:

rustc --version

2. Install Rust Rover IDE

Download and install Rust Rover from JetBrains' official website.

3. Create a New Rust Project

Open a terminal and run:

cargo new linear_regression_model
cd linear_regression_model

Open the project in Rust Rover.

4. Update Cargo.toml

Replace the contents of Cargo.toml with the following:

[dependencies]
burn = { version = "0.16.0", features = ["wgpu", "train"] }
burn-ndarray = "0.16.0"
rand = "0.9.0"
rgb = "0.8.50"
textplots = "0.8.6"

Note: Any changes to the dependencies section will result in a zero score.

5. Connect Rust Rover to GitHub

Install Git from Git's official website.

Run the following commands to initialize and push your project:

git init
git remote add origin <your-github-repo-url>
git add .
git commit -m "Initial commit"
git push -u origin main

Results and Evaluation of the Model

1. Generating Synthetic Data

The dataset consists of (x, y) pairs where y = 2x + 1 with some added noise:

use rand::Rng;

fn generate_data(num_samples: usize) -> Vec<(f32, f32)> {
    let mut rng = rand::thread_rng();
    (0..num_samples)
        .map(|_| {
            let x: f32 = rng.gen_range(0.0..10.0);
            let noise: f32 = rng.gen_range(-0.5..0.5);
            let y = 2.0 * x + 1.0 + noise;
            (x, y)
        })
        .collect()
}

2. Defining the Model

A simple linear regression model is implemented using the Burn library:

use burn::tensor::{Tensor, Data};
use burn::module::Module;

#[derive(Module, Debug)]
struct LinearRegression {
    weight: Tensor<f32>,
    bias: Tensor<f32>,
}

impl LinearRegression {
    fn new() -> Self {
        Self {
            weight: Tensor::from_data(Data::from([0.0])),
            bias: Tensor::from_data(Data::from([0.0])),
        }
    }

    fn forward(&self, x: &Tensor<f32>) -> Tensor<f32> {
        &self.weight * x + &self.bias
    }
}

3. Training the Model

A training loop using Mean Squared Error (MSE) and gradient descent:

fn train_model(model: &mut LinearRegression, data: &[(f32, f32)], epochs: usize, lr: f32) {
    for _ in 0..epochs {
        let mut loss = 0.0;
        for &(x, y) in data {
            let x_tensor = Tensor::from_data(Data::from([x]));
            let y_true = Tensor::from_data(Data::from([y]));
            let y_pred = model.forward(&x_tensor);
            let error = &y_pred - &y_true;
            let grad = error.clone() * 2.0;
            model.weight = &model.weight - &grad * lr;
            model.bias = &model.bias - &grad * lr;
            loss += (error.clone() * error.clone()).to_data().value[0];
        }
        println!("Loss: {}", loss);
    }
}

4. Evaluating the Model

The model is tested on unseen data, and results are visualized using textplots:

use textplots::{Chart, Plot, Shape};

fn plot_results(data: &[(f32, f32)], model: &LinearRegression) {
    let plot_data: Vec<(f32, f32)> = data.iter()
        .map(|&(x, _)| (x, model.forward(&Tensor::from_data(Data::from([x]))).to_data().value[0]))
        .collect();
    Chart::new(100, 30, 0.0, 10.0)
        .lineplot(&Shape::Points(&plot_data))
        .display();
}

Reflection on the Learning Process

Challenges Faced

Setting up Rust and Rust Rover took time due to missing dependencies.

Debugging burn library issues required checking multiple sources.

Training instability was initially present but improved after tweaking learning rates.

Resources Used

Rust Official Documentation

Burn Library Docs

Rust GitHub Guide

AI tools for syntax debugging

Learning Reflections

This project enhanced my understanding of Rust and AI model training.

I learned how to handle dependencies and compile large Rust projects efficiently.

Debugging and using AI-assisted tools helped speed up implementation.

Submission

The complete source code is available on GitHub at [your-repo-link].

The Cargo.toml file remains unchanged.

This README.md file documents the entire process.

The project was submitted via Blackboard LMS.
