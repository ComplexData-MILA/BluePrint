import json
import matplotlib.pyplot as plt
import os
import textwrap

# Define whether to include models with "finetuned" in their name
INCLUDE_FINETUNED = True

# Define the input JSON file path and output directory
json_file_path = '/home/s4yor1/scratch/model_evaluation/model_evaluation_metrics.json'
output_dir = '/home/s4yor1/scratch/output_plots/'

# Create the output directory if it doesn\'t exist
os.makedirs(output_dir, exist_ok=True)

# Load the JSON data
with open(json_file_path, 'r') as f:
    data = json.load(f)

# Iterate through each top-level metric in the JSON data
for metric_name, metric_data in data.items():
    if metric_name == "diagonality_scores":
        # Handle diagonality_scores separately
        plt.figure(figsize=(10, 6))
        models = list(metric_data.keys())
        # Filter out finetuned models if INCLUDE_FINETUNED is False
        if not INCLUDE_FINETUNED:
            models = [model for model in models if "finetuned" not in model.lower()]
        scores = [metric_data[model] for model in models]
        plt.bar(models, scores)
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.title(f'{metric_name}')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plot_filename = os.path.join(output_dir, f'{metric_name}.png')
        # plt.savefig(plot_filename)
        plt.close()
        print(f"Saved plot: {plot_filename}")
    else:
        # Handle other metrics
        for run_id, run_data in metric_data.items():
            plt.figure(figsize=(10, 6))
            models = list(run_data.keys())
            # Filter out finetuned models if INCLUDE_FINETUNED is False
            if not INCLUDE_FINETUNED:
                models = [model for model in models if "finetuned" not in model.lower()]
            scores = [run_data[model] for model in models]
            
            plt.bar(models, scores)
            
            plt.xlabel('Model Type')
            plt.ylabel('Score')
            plt.title(f'{metric_name} - Run {run_id}')
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            
            # Ensure the filename is valid
            safe_metric_name = "".join(c if c.isalnum() else "_" for c in metric_name)
            safe_run_id = "".join(c if c.isalnum() else "_" for c in str(run_id))
            
            plot_filename = os.path.join(output_dir, f'{safe_metric_name}_run_{safe_run_id}.png')
            # plt.savefig(plot_filename)
            plt.close() # Close the plot to free memory
            print(f"Saved plot: {plot_filename}")

# After loading the data
# Calculate average metrics for each model across all runs
avg_metrics = {}

for metric_name, metric_data in data.items():
    if metric_name != "diagonality_scores":  # Skip diagonality scores as they're already per model
        for run_id, run_data in metric_data.items():
            for model, score in run_data.items():
                # Skip finetuned models if INCLUDE_FINETUNED is False
                if not INCLUDE_FINETUNED and "finetuned" in model.lower():
                    continue
                
                if model not in avg_metrics:
                    avg_metrics[model] = {}
                
                if metric_name not in avg_metrics[model]:
                    avg_metrics[model][metric_name] = []
                    
                avg_metrics[model][metric_name].append(score)

# Calculate the average for each metric and model
for model in avg_metrics:
    for metric in avg_metrics[model]:
        avg_metrics[model][metric] = sum(avg_metrics[model][metric]) / len(avg_metrics[model][metric])

# Create a plot for average metrics by metric
plt.figure(figsize=(12, 8))
models = list(avg_metrics.keys())
metrics = list(next(iter(avg_metrics.values())).keys())

x = range(len(metrics))
width = 0.8 / len(models)

for i, model in enumerate(models):
    values = [avg_metrics[model][metric] for metric in metrics]
    plt.bar([pos + width * i for pos in x], values, width=width, label=model)

plt.xlabel('Metrics', fontsize=14, fontweight='bold')
plt.ylabel('Average Score', fontsize=14, fontweight='bold')
plt.title('Average Metrics Comparison')

# Wrap long metric names
wrapped_metrics = ['\n'.join(textwrap.wrap(metric, width=15)) for metric in metrics]
plt.xticks([pos + width * (len(models) - 1) / 2 for pos in x], wrapped_metrics, ha='center')

plt.legend()
plt.tight_layout()

# Save the average metrics plot
avg_plot_filename = os.path.join(output_dir, f"average_metrics_comparison{'' if INCLUDE_FINETUNED else '_no_finetuned'}.png")
plt.savefig(avg_plot_filename)
plt.close()
print(f"Saved average metrics plot: {avg_plot_filename}")

print("All plots generated successfully.")
