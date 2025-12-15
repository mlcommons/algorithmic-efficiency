# plot_results.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import collections

# --- Configuration ---
# The base directory to start searching from.
# The script assumes it's run from the 'experiment_runs' directory.
SEARCH_DIR = Path("~/experiment_runs/tests/regression_tests/adamw").expanduser()

# The directory where the output plots will be saved.
OUTPUT_DIR = Path("ssim_plots")

# The columns to use for the x and y axes.
X_AXIS_COL = "global_step"
Y_AXIS_CANDIDATES = ["validation/loss", "validation/ctc_loss"]
# ---------------------

def generate_plots():
    """
    Finds all 'measurements.csv' files, groups them by workflow,
    and generates a JAX vs. PyTorch plot for each.
    """
    # Create the output directory if it doesn't already exist
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"📊 Plots will be saved to the '{OUTPUT_DIR}' directory.")

    # Use a dictionary to group file paths by their workflow name
    # e.g., {'fastmri': [...], 'wmt': [...]}
    workflow_files = collections.defaultdict(list)

    # Recursively find all 'measurements.csv' files in the search directory
    for csv_path in SEARCH_DIR.rglob("measurements.csv"):
        try:
            # The directory name looks like 'fastmri_jax' or 'wmt_pytorch'.
            # We get this from the parent of the parent of the csv file.
            # e.g., .../fastmri_jax/trial_1/measurements.csv
            workflow_framework_name = csv_path.parent.parent.name
            
            # Split the name to get the framework (last part) and workflow (everything else)
            parts = workflow_framework_name.split('_')
            framework = parts[-1]
            workflow = '_'.join(parts[:-1])
            
            # Store the path and framework for this workflow
            if framework in ['jax', 'pytorch']:
                workflow_files[workflow].append({'path': csv_path, 'framework': framework})

        except IndexError:
            # This handles cases where the directory name might not match the expected pattern
            print(f"⚠️ Could not parse workflow/framework from path: {csv_path}")
            continue

    if not workflow_files:
        print("❌ No 'measurements.csv' files found. Check the SEARCH_DIR variable and your folder structure.")
        return

    print(f"\nFound {len(workflow_files)} workflows. Generating plots...")

    # Iterate through each workflow and its associated files to create a plot
    for workflow, files in workflow_files.items():
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))

        print(f"  -> Processing workflow: '{workflow}'")

        y_axis_col_used = None # To store the name of the y-axis column for the plot labels
        
        # Plot data for each framework (JAX and PyTorch) on the same figure
        for item in files:
            try:
                df = pd.read_csv(item['path'])
                
                y_axis_col = None
                for candidate in Y_AXIS_CANDIDATES:
                    if candidate in df.columns:
                        y_axis_col = candidate
                        if not y_axis_col_used:
                            y_axis_col_used = y_axis_col # Set the label from the first file
                        break # Found a valid column, no need to check further
                
                # if item['framework'] == 'jax':
                #     y_axis_col = None

                # Check if the required columns exist in the CSV
                if X_AXIS_COL in df.columns and y_axis_col:
                                        
                    # 1. Forward-fill 'global_step' to propagate the last valid step downwards.
                    df[X_AXIS_COL] = df[X_AXIS_COL].ffill()
                    
                    # 2. Drop any rows where 'validation/ssim' is empty (NaN).
                    df_cleaned = df.dropna(subset=[y_axis_col])
                    
                    # Plot the cleaned data
                    ax.plot(
                        df_cleaned[X_AXIS_COL],
                        df_cleaned[y_axis_col],
                        label=item['framework'].capitalize(), # e.g., 'Jax'
                        marker='.',
                        linestyle='-',
                        alpha=0.8
                    )
                else:
                    print(f"     - Skipping {item['path']} (missing required columns).")

            except Exception as e:
                print(f"     - ❗️ Error reading {item['path']}: {e}")

        # Customize and save the plot
        ax.set_title(f'Validation loss vs. Global Step for {workflow.replace("_", " ").title()}', fontsize=16)
        ax.set_xlabel("Global Step", fontsize=12)
        ax.set_ylabel("Validation loss", fontsize=12)
        ax.legend(title="Framework", fontsize=10)
        plt.tight_layout()
        plt.yscale('log')

        # Define the output filename and save the figure
        output_filename = OUTPUT_DIR / f"{workflow}_comparison.png"
        plt.savefig(output_filename, dpi=150)
        plt.close(fig) # Close the figure to free up memory

    print("\n✅ All plots generated successfully!")


if __name__ == "__main__":
    generate_plots()