use csv::ReaderBuilder;
use rust_lion::{LionConfig, lion_optimize};
use std::env;
use std::error::Error;
use std::fs::File;
use std::path::Path;

/// Load tab-separated numeric data from a file.
fn load_numeric_tsv<P: AsRef<Path>>(path: P) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(false)
        .from_reader(file);

    let mut data: Vec<Vec<f64>> = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let row: Result<Vec<f64>, _> = record.iter().map(|field| field.parse::<f64>()).collect();

        data.push(row?);
    }

    Ok(data)
}

/// Compute regression metrics: MAE, RMSE, and R².
/// Returns (mae, rmse, r_squared)
fn compute_metrics(data: &[&Vec<f64>], coefficients: &[f64], n_features: usize) -> (f64, f64, f64) {
    let mut sum_abs_error = 0.0;
    let mut sum_sq_error = 0.0;
    let mut sum_target = 0.0;
    let mut sum_sq_diff_from_mean = 0.0;

    for row in data {
        let features = &row[..n_features];
        let target = row[n_features];

        // Compute prediction
        let mut pred = 0.0;
        for (i, &feature) in features.iter().enumerate() {
            pred += coefficients[i] * feature;
        }

        let error = target - pred;
        sum_abs_error += error.abs();
        sum_sq_error += error * error;
        sum_target += target;
    }

    let n = data.len() as f64;
    let mean_target = sum_target / n;

    // Compute total sum of squares (for R²)
    for row in data {
        let target = row[n_features];
        let diff = target - mean_target;
        sum_sq_diff_from_mean += diff * diff;
    }

    let mae = sum_abs_error / n;
    let rmse = (sum_sq_error / n).sqrt();
    let r_squared = if sum_sq_diff_from_mean > 0.0 {
        1.0 - (sum_sq_error / sum_sq_diff_from_mean)
    } else {
        0.0
    };

    (mae, rmse, r_squared)
}

/// Main entry point: loads dataset from command-line argument and optimizes multi-feature linear regression.
/// Usage: cargo run -- <path_to_data_file>
/// Last column is treated as the target; all preceding columns are features.
fn main() -> Result<(), Box<dyn Error>> {
    // Parse command-line arguments
    let args: Vec<String> = env::args().collect();

    let file_path: &Path = args
        .get(1)
        .map(Path::new)
        .unwrap_or_else(|| Path::new("airfoil_self_noise.dat"));

    // Load the dataset
    let data = load_numeric_tsv(file_path)?;

    if data.is_empty() {
        eprintln!("Error: Dataset is empty.");
        return Err("Empty dataset".into());
    }

    // Determine dimensions: all columns except the last are features
    let n_cols = data[0].len();
    if n_cols < 2 {
        eprintln!("Error: Dataset must have at least 2 columns (features + target).");
        return Err("Insufficient columns".into());
    }

    let n_features = n_cols - 1; // All columns except the last
    let dim = n_features;

    // Filter rows with all numeric values
    let valid_data: Vec<_> = data.iter().filter(|row| row.len() > n_features).collect();

    if valid_data.is_empty() {
        eprintln!("Error: No valid data rows found.");
        return Err("No valid data".into());
    }

    println!("Loaded {} valid samples from dataset.", valid_data.len());
    println!(
        "Features: {}, Target column: {}",
        n_features,
        n_features + 1
    );

    // Configure the Lion algorithm
    let config = LionConfig::new(dim)
        .with_bounds(vec![-1.0; dim], vec![1.0; dim])
        .with_cubs_per_generation(16)
        .with_max_generations(100)
        .with_maturity_age(3)
        .with_crossover_probs(0.3, 0.6)
        .with_mutation_prob(0.4)
        .with_seed(42);

    // Define the objective function: minimize MSE (Mean Squared Error)
    let objective = |params: &[f64]| -> f64 {
        let mut mse = 0.0;
        let mut count = 0;

        for row in &valid_data {
            // Extract features (all columns except the last)
            let features = &row[..n_features];
            // Extract target (last column)
            let target = row[n_features];

            // Compute prediction: sum of (param[i] * feature[i])
            let mut pred = 0.0;
            for (i, &feature) in features.iter().enumerate() {
                pred += params[i] * feature;
            }

            // Accumulate squared error
            let err = target - pred;
            mse += err * err;
            count += 1;
        }

        // Return average MSE
        if count > 0 {
            mse / count as f64
        } else {
            f64::INFINITY
        }
    };

    // Run the Lion algorithm
    println!("\nRunning Lion algorithm optimization...");
    let result = lion_optimize(&config, objective);

    // Print results
    println!("\n=== Lion Algorithm Optimization Results ===");
    println!("Best fitness (MSE): {:.6}", result.best_fitness);
    println!("Generations completed: {}", result.generations);
    println!("\nOptimized regression coefficients:");
    for (i, &coeff) in result.best_position.iter().enumerate() {
        println!("  Feature {}: {:.6}", i + 1, coeff);
    }

    // Compute regression metrics on the full dataset
    let (mae, rmse, r_squared) = compute_metrics(&valid_data, &result.best_position, n_features);

    // Print regression performance metrics
    println!("\n=== Regression Performance Metrics ===");
    println!("Mean Absolute Error (MAE): {:.6}", mae);
    println!("Root Mean Squared Error (RMSE): {:.6}", rmse);
    println!("R² (Coefficient of Determination): {:.6}", r_squared);

    // Show predictions on a sample of data points
    println!("\nSample predictions (first 10 samples):");
    println!("Target\tPredicted\tError");
    for row in valid_data.iter().take(10) {
        let features = &row[..n_features];
        let target = row[n_features];

        let mut pred = 0.0;
        for (i, &feature) in features.iter().enumerate() {
            pred += result.best_position[i] * feature;
        }

        let error = (target - pred).abs();
        println!("{:.2}\t{:.2}\t\t{:.2}", target, pred, error);
    }

    Ok(())
}
