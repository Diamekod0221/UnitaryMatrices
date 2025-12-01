import numpy as np
import pandas as pd
from computation import gen_points_explicit_block, estimate_pi

def generate_results(size_block_map):
    results = []

    for size, block_sizes in size_block_map.items():
        for block_size in block_sizes:
            # Generate points for CMC
            cmc_points = gen_points_explicit_block("CMC", size, seed=None, B=block_size)
            cmc_estimate = estimate_pi(cmc_points)

            # Generate points for Ginibre
            ginibre_points = gen_points_explicit_block("ginibre", size, seed=None, B=block_size)
            ginibre_estimate = estimate_pi(ginibre_points)

            results.append({
                'size': size,
                'block_size': block_size,
                'cmc_estimate': cmc_estimate,
                'ginibre_estimate': ginibre_estimate
            })

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv('results.csv', index=False)

if __name__ == "__main__":
    size_block_map = {100: [10, 2, 3]}  # Example input
    generate_results(size_block_map)
