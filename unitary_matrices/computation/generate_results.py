import numpy as np
import pandas as pd
from computation import gen_points_explicit_block, estimate_pi

def generate_results(size_block_map, seed):
    results = []

    for size, block_sizes in size_block_map.items():
        for block_size in block_sizes:
            # Generate points for CMC
            cmc_points = gen_points_explicit_block("CMC", size, seed=seed, B=block_size)
            cmc_estimate = estimate_pi(cmc_points)

            # Generate points for Ginibre
            ginibre_points = gen_points_explicit_block("ginibre", size, seed=seed, B=block_size)
            ginibre_estimate = estimate_pi(ginibre_points)

            results.append({
                'size': size,
                'block_size': block_size,
                'seed': seed,
                'cmc_estimate': cmc_estimate,
                'ginibre_estimate': ginibre_estimate
            })

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(results)
    return df

def run_multiple_seeds(size_block_map, num_seeds):
    all_results = []

    for seed in np.random.randint(0, 1000000, num_seeds):
        results_df = generate_results(size_block_map, seed)
        all_results.append(results_df)

    # Concatenate all results and save to CSV
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv('results_multiple_seeds.csv', index=False)

if __name__ == "__main__":
    size_block_map = {900: [900, 30, 3]}  # Example input
    num_seeds = 10  # Number of random seeds to generate
    run_multiple_seeds(size_block_map, num_seeds)
