//! Lion's Algorithm: a nature-inspired search / optimization algorithm.
//!
//! Simplified Rust implementation of:
//! B. R. Rajakumar, "The Lion’s Algorithm: A New Nature-Inspired Search Algorithm"

/// Gender of a lion (used to separate male/female cub groups).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gender {
    Male,
    Female,
}

/// A single lion / solution.
#[derive(Clone, Debug)]
struct Lion {
    position: Vec<f64>,
    fitness: f64,
    gender: Gender,
    age: u32,
}

/// Configuration parameters for the Lion algorithm.
///
/// You can construct it via `LionConfig::new(dim)`
#[derive(Clone, Debug)]
pub struct LionConfig {
    /// Dimensionality of the solution vector.
    pub dim: usize,
    /// Lower bounds per dimension (length = dim).
    pub min_bounds: Vec<f64>,
    /// Upper bounds per dimension (length = dim).
    pub max_bounds: Vec<f64>,

    /// Number of cubs generated per generation (will be rounded up to even).
    pub cubs_per_generation: usize,

    /// Cub maturity age: after this, cubs can be considered for takeover.
    pub maturity_age: u32,

    /// Maximum number of generations.
    pub max_generations: u32,

    /// Crossover probabilities (p1, p2) for the dual-probability single-point crossover.
    pub crossover_probs: (f64, f64),

    /// Mutation probability per gene.
    pub mutation_prob: f64,

    /// Seed for RNG (if None, will use entropy from OS).
    pub seed: Option<u64>,
}

impl LionConfig {
    /// Create a config with basic defaults for a given dimensionality.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            min_bounds: vec![-10.0; dim],
            max_bounds: vec![10.0; dim],
            cubs_per_generation: 8,
            maturity_age: 3,
            max_generations: 100,
            crossover_probs: (0.2, 0.6),
            mutation_prob: 0.5,
            seed: None,
        }
    }

    /// Set per-dimension bounds.
    pub fn with_bounds(mut self, min_bounds: Vec<f64>, max_bounds: Vec<f64>) -> Self {
        assert_eq!(min_bounds.len(), self.dim);
        assert_eq!(max_bounds.len(), self.dim);
        self.min_bounds = min_bounds;
        self.max_bounds = max_bounds;
        self
    }

    /// Set the number of generations.
    pub fn with_max_generations(mut self, max_generations: u32) -> Self {
        self.max_generations = max_generations;
        self
    }

    /// Set number of cubs per generation.
    pub fn with_cubs_per_generation(mut self, cubs_per_generation: usize) -> Self {
        // Bounded as minimum two by the paper
        self.cubs_per_generation = cubs_per_generation.max(2);
        self
    }

    /// Set maturity age for cubs.
    pub fn with_maturity_age(mut self, maturity_age: u32) -> Self {
        self.maturity_age = maturity_age.max(1);
        self
    }

    /// Set crossover probabilities (p1, p2).
    pub fn with_crossover_probs(mut self, p1: f64, p2: f64) -> Self {
        self.crossover_probs = (p1, p2);
        self
    }

    /// Set mutation probability.
    pub fn with_mutation_prob(mut self, p: f64) -> Self {
        self.mutation_prob = p.clamp(0.0, 1.0);
        self
    }

    /// Set random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Result of running the Lion algorithm.
#[derive(Clone, Debug)]
pub struct OptimizationResult {
    pub best_position: Vec<f64>,
    pub best_fitness: f64,
    pub generations: u32,
}

/// Main entry point: run the Lion algorithm with the given configuration and
/// objective function.
/// The objective function should minimize its return value.
pub fn lion_optimize<F>(config: &LionConfig, objective: F) -> OptimizationResult
where
    F: Fn(&[f64]) -> f64,
{
    // Seed global RNG if configured
    if let Some(seed) = config.seed {
        fastrand::seed(seed);
    }

    // Ensure even number of cubs per generation
    let cubs_per_generation = if config.cubs_per_generation.is_multiple_of(2) {
        config.cubs_per_generation
    } else {
        config.cubs_per_generation + 1
    };

    // Pride Generation: initial male and female lions
    let mut male = random_lion(config, Gender::Male, &objective);
    let mut female = random_lion(config, Gender::Female, &objective);

    // Cub pools
    let mut male_cubs: Vec<Lion> = Vec::new();
    let mut female_cubs: Vec<Lion> = Vec::new();

    // Best overall
    let mut best = if male.fitness <= female.fitness {
        male.clone()
    } else {
        female.clone()
    };

    // Internal breeding count tracking for female takeover-like behavior.
    let mut female_breed_count: u32 = 0;
    // simplified analogue of Bstrength in the paper
    let max_female_breed_strength: u32 = 5;

    for generation in 0..config.max_generations {
        // Mating Step
        // Enforce that only Male–Female pairs mate.
        debug_assert_eq!(male.gender, Gender::Male, "Territorial male is not Male");
        debug_assert_eq!(
            female.gender,
            Gender::Female,
            "Territorial female is not Female"
        );

        let mut new_cubs: Vec<Lion> =
            mate_and_generate_cubs(config, &male, &female, cubs_per_generation, &objective);

        // Assign genders via gender grouping
        for cub in new_cubs.drain(..) {
            if fastrand::bool() {
                male_cubs.push(Lion {
                    gender: Gender::Male,
                    ..cub
                });
            } else {
                female_cubs.push(Lion {
                    gender: Gender::Female,
                    ..cub
                });
            }
        }

        // Keep cub pools balanced
        balance_cub_pools(&mut male_cubs, &mut female_cubs);

        // Territorial Defense Step
        territorial_defense(
            config,
            &objective,
            &mut male,
            &mut female,
            &mut male_cubs,
            &mut female_cubs,
        );

        // Age cubs
        for c in male_cubs.iter_mut().chain(female_cubs.iter_mut()) {
            c.age += 1;
        }

        // Territorial Takeover Step (simplified)
        // Choose the best mature male candidate (clone only the selected candidate).
        let best_male_candidate: Option<Lion> = male_cubs
            .iter()
            .filter(|c| c.age >= config.maturity_age && c.gender == Gender::Male)
            .min_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal))
            .cloned();

        if let Some(candidate) = best_male_candidate {
            debug_assert_eq!(candidate.gender, Gender::Male);
            if candidate.fitness < male.fitness {
                male = candidate;
            }
        }

        // Choose the best mature female candidate similarly and apply takeover logic.
        let best_female_candidate: Option<Lion> = female_cubs
            .iter()
            .filter(|c| c.age >= config.maturity_age && c.gender == Gender::Female)
            .min_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal))
            .cloned();

        if let Some(candidate) = best_female_candidate {
            debug_assert_eq!(candidate.gender, Gender::Female);

            if candidate.fitness < female.fitness {
                // Found a stronger female, replace immediately.
                female = candidate;
                female_breed_count = 0;
            } else {
                // No better one; increment breed count.
                female_breed_count += 1;
                if female_breed_count >= max_female_breed_strength {
                    // Force takeover by the best candidate even if slightly worse.
                    female = candidate;
                    female_breed_count = 0;
                }
            }
        }

        // Keep cub pools from exploding: truncate to maintain diversity
        truncate_cub_pools(&mut male_cubs, &mut female_cubs, cubs_per_generation);

        // Track global best by iterating references; clone only when a better lion is found.
        for l in std::iter::once(&male)
            .chain(std::iter::once(&female))
            .chain(male_cubs.iter())
            .chain(female_cubs.iter())
        {
            if l.fitness < best.fitness {
                best = l.clone();
            }
        }

        if generation == config.max_generations - 1 {
            return OptimizationResult {
                best_position: best.position,
                best_fitness: best.fitness,
                generations: config.max_generations,
            };
        }
    }

    OptimizationResult {
        best_position: best.position,
        best_fitness: best.fitness,
        generations: config.max_generations,
    }
}

// Internal helpers
/// Generate a random f64 in [min, max).
fn random_in_range(min: f64, max: f64) -> f64 {
    min + (max - min) * fastrand::f64()
}

/// Generate a random lion of the given gender
fn random_lion<F>(config: &LionConfig, gender: Gender, objective: &F) -> Lion
where
    F: Fn(&[f64]) -> f64,
{
    let mut pos = Vec::with_capacity(config.dim);
    for i in 0..config.dim {
        let min = config.min_bounds[i];
        let max = config.max_bounds[i];
        let v = random_in_range(min, max);
        pos.push(v);
    }
    let fitness = objective(&pos);
    Lion {
        position: pos,
        fitness,
        gender,
        age: 0,
    }
}

/// Mate a male and female lion to produce cubs.
fn mate_and_generate_cubs<F>(
    config: &LionConfig,
    male: &Lion,
    female: &Lion,
    cub_count: usize,
    objective: &F,
) -> Vec<Lion>
where
    F: Fn(&[f64]) -> f64,
{
    // Enforce only male–female mating
    assert!(
        male.gender == Gender::Male && female.gender == Gender::Female,
        "Mating is only allowed between a Male lion and a Female lion"
    );

    let mut cubs = Vec::with_capacity(cub_count);

    for _ in 0..cub_count {
        let child_pos = single_point_crossover_dual_prob(
            &male.position,
            &female.position,
            config.crossover_probs,
        );

        let mutated = mutate_vector(&child_pos, config);

        let fitness = objective(&mutated);
        cubs.push(Lion {
            position: mutated,
            fitness,
            // Gender will be assigned later at crossover
            gender: Gender::Male,
            age: 0,
        });
    }

    cubs
}

/// Single-point crossover with two probability modes.
///
/// We pick a crossover point and then, depending on a random draw, either use
/// parent1-head + parent2-tail or parent2-head + parent1-tail with different
/// probabilities. If `dim < 2`, just return a random parent clone.
fn single_point_crossover_dual_prob(p1: &[f64], p2: &[f64], probs: (f64, f64)) -> Vec<f64> {
    let dim = p1.len();
    assert_eq!(p1.len(), p2.len());

    if dim < 2 {
        return if fastrand::bool() {
            p1.to_vec()
        } else {
            p2.to_vec()
        };
    }

    // cut in [1, dim-1]
    let cut = fastrand::usize(1..dim);
    let mode_draw: f64 = fastrand::f64();
    let (p_head, p_tail) = if mode_draw < probs.0 {
        (p1, p2)
    } else if mode_draw < probs.0 + probs.1 {
        (p2, p1)
    } else {
        (p1, p2)
    };

    let mut child = Vec::with_capacity(dim);
    for i in 0..dim {
        if i < cut {
            child.push(p_head[i]);
        } else {
            child.push(p_tail[i]);
        }
    }

    child
}

/// Mutate a solution vector based on mutation probability.
fn mutate_vector(x: &[f64], config: &LionConfig) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.len());
    for (i, &v) in x.iter().enumerate() {
        if fastrand::f64() < config.mutation_prob {
            let min = config.min_bounds[i];
            let max = config.max_bounds[i];
            out.push(random_in_range(min, max));
        } else {
            out.push(v);
        }
    }
    out
}

/// Keep male and female cub pools the same size by trimming the worse fitness from larger pool
fn balance_cub_pools(male_cubs: &mut Vec<Lion>, female_cubs: &mut Vec<Lion>) {
    // sort ascending by fitness (best first) using total order and an unstable sort
    male_cubs.sort_unstable_by(|a, b| a.fitness.total_cmp(&b.fitness));
    female_cubs.sort_unstable_by(|a, b| a.fitness.total_cmp(&b.fitness));

    let mut m = male_cubs.len();
    let mut f = female_cubs.len();

    while m > f {
        // remove worst male
        male_cubs.pop();
        m -= 1;
    }
    while f > m {
        female_cubs.pop();
        f -= 1;
    }
}

/// Compute a simple pride strength as the average fitness of all lions
fn pride_strength(male: &Lion, female: &Lion, male_cubs: &[Lion], female_cubs: &[Lion]) -> f64 {
    let mut sum = male.fitness + female.fitness;
    let mut count = 2usize;

    for c in male_cubs.iter().chain(female_cubs.iter()) {
        sum += c.fitness;
        count += 1;
    }

    sum / count as f64
}

/// Territorial defense: a nomadic lion may invade and replace the male and kill cubs if it is stronger than both the male and the pride.
fn territorial_defense<F>(
    config: &LionConfig,
    objective: &F,
    male: &mut Lion,
    female: &mut Lion,
    male_cubs: &mut Vec<Lion>,
    female_cubs: &mut Vec<Lion>,
) where
    F: Fn(&[f64]) -> f64,
{
    // Generate a nomadic lion (always modeled as male in the paper).
    let mut nomad = random_lion(config, Gender::Male, objective);
    debug_assert_eq!(nomad.gender, Gender::Male);

    let pride_str = pride_strength(male, female, male_cubs, female_cubs);

    if nomad.fitness < male.fitness && nomad.fitness < pride_str {
        // Nomad takes over: becomes new male, cubs are killed
        nomad.gender = Gender::Male;
        *male = nomad;
        male_cubs.clear();
        female_cubs.clear();
    }
}

/// Prevent cub pool from growing without bound.
fn truncate_cub_pools(
    male_cubs: &mut Vec<Lion>,
    female_cubs: &mut Vec<Lion>,
    max_cubs_per_gender: usize,
) {
    male_cubs.sort_unstable_by(|a, b| a.fitness.total_cmp(&b.fitness));
    female_cubs.sort_unstable_by(|a, b| a.fitness.total_cmp(&b.fitness));

    if male_cubs.len() > max_cubs_per_gender {
        male_cubs.truncate(max_cubs_per_gender);
    }
    if female_cubs.len() > max_cubs_per_gender {
        female_cubs.truncate(max_cubs_per_gender);
    }
}
