use rust_lion::{LionConfig, lion_optimize};
/// A simple demonstration of using the Lion optimization algorithm
fn main() {
    let data: Vec<(f64, f64)> = vec![
        (0.0, 1.1),
        (0.5, 2.3),
        (1.0, 3.4),
        (1.5, 4.0),
        (2.0, 5.2),
        (2.5, 7.0),
        (3.0, 8.0),
    ];

    let dim = 2;
    // Set bounds and configurations for Lion Algorithm
    let config = LionConfig::new(dim)
        .with_bounds(vec![-10.0, -10.0], vec![10.0, 10.0])
        .with_cubs_per_generation(10)
        .with_max_generations(200)
        .with_maturity_age(3)
        .with_crossover_probs(0.3, 0.6)
        .with_mutation_prob(0.3)
        .with_seed(1234);

    let objective = |params: &[f64]| -> f64 {
        let a = params[0];
        let b = params[1];

        let mut mse = 0.0;
        for (x, y) in &data {
            let pred = a * x + b;
            let err = y - pred;
            mse += err * err;
        }
        mse / data.len() as f64
    };

    // Run optimization
    let result = lion_optimize(&config, objective);

    // Print results
    println!("=== Lion Algorithm Demo on Simple Dataset ===");
    println!("Best fitness (MSE): {:.6}", result.best_fitness);
    println!(
        "Best parameters: a = {:.6}, b = {:.6}",
        result.best_position[0], result.best_position[1]
    );
    println!("Generations: {}", result.generations);

    // Optional: Show predictions for each data point
    println!("\nData vs. model predictions:");
    let a = result.best_position[0];
    let b = result.best_position[1];
    for (x, y) in &data {
        let pred = a * x + b;
        println!("x = {:>4.2}, y = {:>5.2}, pred = {:>7.3}", x, y, pred);
    }
}
