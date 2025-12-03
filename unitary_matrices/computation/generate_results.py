import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

def visualize_results(csv_file):
    df = pd.read_csv(csv_file)

    # Plot for CMC estimates
    cmc_df = df[['block_size', 'cmc_estimate']]
    melted_cmc_df = cmc_df.melt(id_vars=['block_size'], value_vars=['cmc_estimate'], 
                                  var_name='method', value_name='estimated_val')

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='block_size', y='estimated_val', data=melted_cmc_df, 
                palette='Set2', showfliers=False)
    plt.title('CMC Estimates by Block Size')
    plt.xlabel('Block Size')
    plt.ylabel('Estimated Value')
    plt.ylim(2.4, 3.4)  # Assuming estimates are in the range of pi
    plt.axhline(y=3.14, color='r', linestyle='--', label='True Value of Pi')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('results_cmc_boxplot.png')
    plt.close()

    # Plot for Ginibre estimates
    ginibre_df = df[['block_size', 'ginibre_estimate']]
    melted_ginibre_df = ginibre_df.melt(id_vars=['block_size'], value_vars=['ginibre_estimate'], 
                                          var_name='method', value_name='estimated_val')

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='block_size', y='estimated_val', data=melted_ginibre_df, 
                palette='Set2', showfliers=False)
    plt.title('Ginibre Estimates by Block Size')
    plt.xlabel('Block Size')
    plt.ylabel('Estimated Value')
    plt.ylim(2.4, 3.4)  # Assuming estimates are in the range of pi
    plt.axhline(y=3.14, color='r', linestyle='--', label='True Value of Pi')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('results_ginibre_boxplot.png')
    plt.close()

    # Combined plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='block_size', y='estimated_val', hue='method', data=pd.concat([melted_cmc_df, melted_ginibre_df]), 
                palette='Set2', showfliers=False)
    plt.title('Estimates by Block Size (CMC and Ginibre)')
    plt.xlabel('Block Size')
    plt.ylabel('Estimated Value')
    plt.ylim(2.4, 3.4)  # Assuming estimates are in the range of pi
    plt.axhline(y=3.14, color='r', linestyle='--', label='True Value of Pi')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('results_combined_boxplot.png')
    plt.close()

if __name__ == "__main__":
    size_block_map = {2500: [2500, 50, 2]}  # Example input
    num_seeds = 500# Number of random seeds to generate
    run_multiple_seeds(size_block_map, num_seeds)
    visualize_results('results_multiple_seeds.csv')
