use csv::ReaderBuilder;
use rust_lion::{lion_optimize, LionConfig};
use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::str::FromStr;

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

/// Normalize data to [0, 1] range using min-max scaling.
/// Returns normalized data and (min_vals, max_vals) for each column.
fn normalize_data(data: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    if data.is_empty() {
        return (vec![], vec![], vec![]);
    }

    let n_cols = data[0].len();
    let mut min_vals = vec![f64::INFINITY; n_cols];
    let mut max_vals = vec![f64::NEG_INFINITY; n_cols];

    // Find min/max for each column
    for row in data {
        for (j, &val) in row.iter().enumerate() {
            if val < min_vals[j] {
                min_vals[j] = val;
            }
            if val > max_vals[j] {
                max_vals[j] = val;
            }
        }
    }

    // Normalize data
    let mut normalized = Vec::with_capacity(data.len());
    for row in data {
        let mut norm_row = Vec::with_capacity(n_cols);
        for (j, &val) in row.iter().enumerate() {
            let range = max_vals[j] - min_vals[j];
            let normalized_val = if range > 0.0 {
                (val - min_vals[j]) / range
            } else {
                // If all values are the same, normalize to midpoint
                0.5
            };
            norm_row.push(normalized_val);
        }
        normalized.push(norm_row);
    }

    (normalized, min_vals, max_vals)
}

/// Denormalize a prediction value back to original scale.
fn denormalize_target(normalized_val: f64, min_val: f64, max_val: f64) -> f64 {
    let range = max_val - min_val;
    if range > 0.0 {
        normalized_val * range + min_val
    } else {
        min_val
    }
}

/// Compute regression metrics: MAE, RMSE, and R^2.
fn compute_metrics(data: &[&Vec<f64>], coefficients: &[f64], n_features: usize) -> (f64, f64, f64) {
    let mut sum_abs_error = 0.0;
    let mut sum_sq_error = 0.0;
    let mut sum_target = 0.0;
    let mut sum_sq_diff_from_mean = 0.0;

    let bias = coefficients[0];

    for row in data {
        let features = &row[..n_features];
        let target = row[n_features];

        // prediction = bias + sum w_i(offset by one because of bias term) * x_i
        let mut pred = bias;
        for (i, &feature) in features.iter().enumerate() {
            pred += coefficients[i + 1] * feature;
        }

        let error = target - pred;
        sum_abs_error += error.abs();
        sum_sq_error += error * error;
        sum_target += target;
    }

    let n = data.len() as f64;
    let mean_target = sum_target / n;

    // Compute total sum of squares
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
/// Usage: cargo run -- <path_to_data_file> [--key=value ...]
/// Last column is treated as the target; all preceding columns are features.
fn main() -> Result<(), Box<dyn Error>> {
    // Parse command-line arguments
    let args: Vec<String> = env::args().collect();

    // Default to airfoil_self_noise.dat in current directory
    let file_path: PathBuf = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("airfoil_self_noise.dat"));

    // Load the dataset
    let data = load_numeric_tsv(&file_path)?;

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

    // All columns except the last
    let n_features = n_cols - 1;

    // Keep only rows with enough columns
    let valid_data: Vec<Vec<f64>> = data
        .into_iter()
        .filter(|row| row.len() > n_features)
        .collect();

    if valid_data.is_empty() {
        eprintln!("Error: No valid data rows found.");
        return Err("No valid data".into());
    }

    println!("Loaded {} valid samples from dataset.", valid_data.len());
    println!("Features: {}, Target index: {}", n_features, n_features);

    // Normalize data
    let (normalized_data, min_vals, max_vals) = normalize_data(&valid_data);

    // Convert back to references for optimization
    let normalized_refs: Vec<_> = normalized_data.iter().collect();

    // We use: params[0] = bias, params[1..=n_features] = weights
    let dim = n_features + 1;

    // Parse optional command-line flags for Lion parameters
    let mut flag_map: HashMap<String, String> = HashMap::new();

    for arg in args.iter().skip(2) {
        if let Some(stripped) = arg.strip_prefix("--")
            && let Some((key, val)) = stripped.split_once('=')
        {
            flag_map.insert(key.to_string(), val.to_string());
        }
    }

    // Helper function to parse typed flags with a default fallback
    fn get_flag<T: FromStr + Copy>(
    map: &HashMap<String, String>,
    key: &str,
    default: T,
    ) -> T {
        map.get(key)
            .and_then(|s| s.parse::<T>().ok())
            .unwrap_or(default)
    }

    let max_generations: u32       = get_flag(&flag_map, "generations", 100);
    let cubs_per_generation: usize = get_flag(&flag_map, "cubs", 16);
    let maturity_age: u32          = get_flag(&flag_map, "maturity", 3);
    let crossover1: f64            = get_flag(&flag_map, "crossover1", 0.3);
    let crossover2: f64            = get_flag(&flag_map, "crossover2", 0.6);
    let mutation_prob: f64         = get_flag(&flag_map, "mutation", 0.4);
    let bound_min: f64             = get_flag(&flag_map, "bound-min", -2.0);
    let bound_max: f64             = get_flag(&flag_map, "bound-max",  2.0);



    // Configure the Lion algorithm with the parsed or default values
    let config = LionConfig::new(dim)
        .with_bounds(vec![bound_min; dim], vec![bound_max; dim])
        .with_cubs_per_generation(cubs_per_generation)
        .with_max_generations(max_generations)
        .with_maturity_age(maturity_age)
        .with_crossover_probs(crossover1, crossover2)
        .with_mutation_prob(mutation_prob);

    // Define the objective function: minimize Mean Squared Error on normalized data
    let objective = |params: &[f64]| -> f64 {
        let mut mse = 0.0;
        let mut count = 0;

        let bias = params[0];

        for row in &normalized_refs {
            // Extract features 
            let features = &row[..n_features];
            // Extract target
            let target = row[n_features];

            // prediction = bias + sum w_i * x_i
            let mut pred = bias;
            for (i, &feature) in features.iter().enumerate() {
                pred += params[i + 1] * feature;
            }

            let err = target - pred;
            mse += err * err;
            count += 1;
        }

        if count > 0 {
            mse / count as f64
        } else {
            f64::INFINITY
        }
    };

    // Run the Lion algorithm
    println!("Running Lion algorithm optimization on normalized data...");
    let result = lion_optimize(&config, objective);

    // Print results
    println!("\n=== Lion Algorithm Optimization Results ===");
    println!(
        "Best fitness (MSE on normalized data): {:.6}",
        result.best_fitness
    );
    println!("Generations completed: {}", result.generations);
    println!("\nOptimized regression parameters (on normalized data):");
    println!("  Bias: {:.6}", result.best_position[0]);
    for i in 0..n_features {
        println!(
            "  Feature {} weight: {:.6}",
            i + 1,
            result.best_position[i + 1]
        );
    }

    // Compute regression metrics on the normalized dataset
    let (mae, rmse, r_squared) =
        compute_metrics(&normalized_refs, &result.best_position, n_features);

    // Print regression performance metrics
    println!("\n=== Regression Performance Metrics (on normalized data) ===");
    println!("Mean Absolute Error (MAE): {:.6}", mae);
    println!("Root Mean Squared Error (RMSE): {:.6}", rmse);
    println!("R^2 (Coefficient of Determination): {:.6}", r_squared);

    // Show predictions on a sample of data points
    println!("\nSample predictions 10 predicitions:");
    println!("Target\tPredicted\tError");
    let target_min = min_vals[n_features];
    let target_max = max_vals[n_features];
    for row in normalized_refs.iter().take(10) {
        let features = &row[..n_features];
        let target_normalized = row[n_features];

        // Compute prediction in normalized scale
        let mut pred_normalized = result.best_position[0];
        for (i, &feature) in features.iter().enumerate() {
            pred_normalized += result.best_position[i + 1] * feature;
        }

        // Denormalize to original scale
        let target_original = denormalize_target(target_normalized, target_min, target_max);
        let pred_original = denormalize_target(pred_normalized, target_min, target_max);
        let error = (target_original - pred_original).abs();
        println!(
            "{:.2}\t{:.2}\t\t{:.2}",
            target_original, pred_original, error
        );
    }

    Ok(())
}
