# Rust-Lion
**Author:** Jacob Olinger  
**Email:** jacoboli@pdx.edu  

## Overview  
**Rust-Lion** is a bio-inspired optimization library implementing the **Lion Algorithm (LA)** in **Rust**.  
The Lion Algorithm models social behaviors of lions, such as pride formation, nomadic exploration, and territorial defense, to solve optimization problems efficiently.

**Reference:** [Lion Algorithm Paper](https://www.sciencedirect.com/science/article/pii/S2212017312005610?ref=pdf_download&fr=RR-2&rr=99ca054a098787ab)

## Key Features
- **Efficient optimization**: Reduced cloning via reference-based candidate selection and smart best tracking
- **Idiomatic Rust**: Uses `sort_unstable_by` with `total_cmp` for fast, stable sorting
- **Configurable**: Flexible algorithm parameters (bounds, generations, mutation rates, etc.)
- **Multi-feature linear regression**: Driver program supports any TSV dataset with automatic feature detection

## Library Usage

### Basic Example
```rust
use rust_lion::{LionConfig, lion_optimize};

fn main() {
    let dim = 2;
    let config = LionConfig::new(dim)
        .with_bounds(vec![-10.0, -10.0], vec![10.0, 10.0])
        .with_max_generations(100)
        .with_seed(42);

    let objective = |params: &[f64]| -> f64 {
        // Minimize sum of squares
        params.iter().map(|x| x * x).sum()
    };

    let result = lion_optimize(&config, objective);
    println!("Best fitness: {}", result.best_fitness);
    println!("Best position: {:?}", result.best_position);
}
```

## Driver Program: Linear Regression via Lion Algorithm

The `main.rs` driver program uses the Lion Algorithm to optimize multi-feature linear regression coefficients on arbitrary TSV datasets.

### Usage
```bash
cargo run -- <path_to_data_file>
```

### Example
```bash
cargo run -- airfoil_self_noise.dat
```

### Input Format
- **Tab-separated values (TSV)** with numeric data only
- **Last column**: Target variable (y-values to predict)
- **All preceding columns**: Features (x₁, x₂, ..., xₙ)
- **No headers** expected

### Example Dataset
```
800    0      0.3048  71.3   0.00266337  126.201
1000   0      0.3048  71.3   0.00266337  125.201
...
```
Features: frequency, angle, chord_length, velocity, displacement_thickness  
Target: noise_level (last column)

### Output
The program prints:
- **Sample size** and feature/target column information
- **Best fitness** (Mean Squared Error achieved during optimization)
- **Optimized regression coefficients** for each feature
- **Regression Performance Metrics**:
  - **MAE** (Mean Absolute Error): Average absolute difference between predictions and targets
  - **RMSE** (Root Mean Squared Error): Standard deviation of prediction errors
  - **R^2** (Coefficient of Determination): Proportion of variance explained (0-1, higher is better)
- **Sample predictions** on the first 10 rows (target, predicted, absolute error)

### Assumptions
1. All data is numeric (no missing values or non-numeric entries)
2. Rows with fewer columns than expected are filtered out
3. Linear regression model: `y = w₁·x₁ + w₂·x₂ + ... + wₙ·xₙ`
4. Objective: Minimize Mean Squared Error (MSE) across all samples

## Performance Improvements
The library has been optimized for efficiency:
- **Reference-based candidate selection**: Avoids cloning entire cub pools during territorial takeover
- **Smart best-tracking**: Iterates references and clones only the champion lion when improved
- **Idiomatic sorting**: Uses `sort_unstable_by` with `f64::total_cmp` for speed and correctness
- **Minimal allocations**: Reuses vectors and leverages Rust's move semantics

## Data Source
The included [`airfoil_self_noise.dat`](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise) is a NASA dataset from aerodynamic tests of airfoil blade sections.  
This non-linear real-world dataset tests the algorithm.
The included [`energy_efficiency_y1.dat`](https://archive.ics.uci.edu/dataset/242/energy+efficiency) is a UC Irving dataset looking at the heating and cooling requirements of buildings

## Quick Start
Running the Program

Run the optimizer using the default dataset (airfoil_self_noise.dat):

```cargo run```


Run the optimizer with a custom dataset:

```cargo run -- path/to/datafile.dat```


The dataset must be tab-separated, where:

All columns except the last are treated as input features

The last column is treated as the regression target

Configuration Parameters

You can customize the Lion Algorithm using flags in the following format:

```--key=value```


Example:

```cargo run -- airfoil_self_noise.dat --generations=200 --mutation=0.2```

## Available Parameters

| Parameter        | Default | Description                                   |
|------------------|---------|-----------------------------------------------|
| `--generations=N` | `100`   | Number of Lion Algorithm generations.         |
| `--cubs=N`        | `16`    | Number of cubs produced per generation.       |
| `--maturity=N`    | `3`     | Maturity age for adult lions.                 |
| `--crossover1=P`  | `0.3`   | First crossover probability.                  |
| `--crossover2=P`  | `0.6`   | Second crossover probability.                 |
| `--mutation=P`    | `0.4`   | Mutation probability.                         |
| `--bound-min=X`   | `-2.0`  | Minimum bound for weights and bias.           |
| `--bound-max=X`   | `2.0`   | Maximum bound for weights and bias.           |
| `--seed=N`        | `42`    | Optional random seed for reproducible results.|