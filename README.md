# Rust-Lion
**Author:** Jacob Olinger  
**Email:** jacoboli@pdx.edu  

## Overview  
**Rust-Lion** is a bio-inspired optimization library implementing the **Lion Algorithm (LA)** in **Rust**.  
The Lion Algorithm models social behaviors of lions, such as pride formation, nomadic exploration, and territorial defense, to solve optimization problems efficiently.

**Reference:** [Lion Algorithm Paper](https://www.sciencedirect.com/science/article/pii/S2212017312005610?ref=pdf_download&fr=RR-2&rr=99ca054a098787ab)

## Key Features
- **Efficient optimization**: Reduced cloning via reference-based candidate selection and smart best tracking
- **Idiomatic Rust**: Uses `sort_by` with `total_cmp` for fast, stable sorting
- **Configurable**: Flexible algorithm parameters (bounds, generations, mutation rates, etc.)
- **Multi-feature linear regression**: Driver program supports any numeric tab seperated dataset assuming the last column is the target and the data is well formed

## Driver Program: Linear Regression via Lion Algorithm

The `main.rs` driver program uses the Lion Algorithm to optimize multi-feature linear regression coefficients on arbitrary numeric tab seperated datasets.

### Usage
```bash
cargo run -- <path_to_data_file>
```

### Input Format
- **Tab-separated values** with numeric data only
- **Last column**: Target variable (y-values to predict)
- **All preceding columns**: Features (x1, x2, ..., xn)
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
1. All data is numeric no missing values or non-numeric entries
2. Rows with fewer columns than expected are filtered out
3. Linear regression model: `y = w1 * x1 + w2 * x2 + ... + wn * xn + b`
4. Objective: Minimize Mean Squared Error (MSE) across all samples

## Performance Improvements
The library has been optimized for efficiency:
- **Reference-based candidate selection**: Avoids cloning entire cub pools during territorial takeover
- **Smart best-tracking**: Iterates references and clones only the champion lion when improved
- **Minimal allocations**: Reuses vectors and leverages Rust's move semantics

## Data Source
The included [`airfoil_self_noise.dat`](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise) is a NASA dataset from aerodynamic tests of airfoil blade sections that is highly non-linear.
The included [`energy_efficiency_y1.dat`](https://archive.ics.uci.edu/dataset/242/energy+efficiency) is a UC Irving dataset looking at the heating and cooling requirements of buildings that is linear.

## Quick Start
Running the Program

Run the optimizer using the default dataset, energy_efficiency_y1.dat:

```cargo run --release```


Run the optimizer with a custom dataset:

```cargo run -- path/to/datafile.dat```

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
| `--bound-min=X`   | `0`     | Minimum bound for weights and bias.           |
| `--bound-max=X`   | `1.0`   | Maximum bound for weights and bias.           |
| `--seed=N`        | `42`    | Optional random seed for reproducible results |

## Lessons Learned
I struggled early on with lifetimes, references, and learning to code in an appropriate Rust style. It took me longer than expected to get a working version of the Lion algorithm into general library functions.
I also did not expect the driver program to take as long as it did to make. Initially, I got quite poor performance on the linear regression task, so I added in standard pre-processing normalization of the numeric data to the driver program.
I then worked on optimizing the code, which also took longer than expected. I found that the program works well, R^2 around 0.85, on the energy data set because it is more linear while the airfoil datset is very non-linear, so the linear algorithm struggles to give accurate predicitions. I tried to do some work with vizualization, but ended up not succeding in generating a UI and pivoted to focus on the algorithm and driver program.