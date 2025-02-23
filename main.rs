mod data;
mod model;
mod train;
mod evaluate;

use model::LinearRegression;
use train::train_model;
use evaluate::evaluate_model;
use data::generate_data;

fn main() {
    let (x_train, y_train) = generate_data(100);  // Generate 100 samples
    let mut model = LinearRegression::new();

    println!("Training the model...");
    train_model(&mut model, x_train.clone(), y_train.clone(), 100);  // Train for 100 epochs

    println!("Evaluating the model...");
    evaluate_model(&model);
}
