import argparse
import os
import csv
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

def get_max_value(log_dir, tag="val/accuracy"):
    """
    Finds the maximum value for a specific tag in a single TensorBoard log directory.
    """
    try:
        ea = event_accumulator.EventAccumulator(str(log_dir), size_guidance={
            event_accumulator.SCALARS: 0,
        })
        ea.Reload()
        tags = ea.Tags().get('scalars', [])
        if tag not in tags:
            return None
        events = ea.Scalars(tag)
        values = [event.value for event in events]
        return max(values) if values else None
    except Exception as e:
        print(f"Error reading {log_dir}: {e}")
        return None

def process_all_logs(base_dir, tag="val/accuracy", output_csv="experiment_results.csv"):
    """
    Recursively finds tb_logs directories and extracts info based on the project's structure:
    base_dir / model_name / exp_name / timestamp / tb_logs
    """
    results = []
    base_path = Path(base_dir)
    
    # Search for all 'tb_logs' directories recursively
    # We look for directories that contain tfevents files
    for tb_log_path in base_path.rglob('tb_logs'):
        if not tb_log_path.is_dir():
            continue
            
        # Extract info from path
        # Path structure: ... / model_name / exp_name / timestamp / tb_logs
        parts = tb_log_path.parts
        if len(parts) < 4:
            # Not enough parts to match the expected structure, skip or handle differently
            model_name = "unknown"
            exp_name = "unknown"
            timestamp = "unknown"
        else:
            # Assuming the last 4 parts are model/exp/timestamp/tb_logs
            # We take them relative to the base_dir if possible for better accuracy
            rel_path = tb_log_path.relative_to(base_path)
            rel_parts = rel_path.parts
            
            if len(rel_parts) >= 3:
                model_name = rel_parts[0]
                exp_name = rel_parts[1]
                timestamp = rel_parts[2]
            else:
                model_name = "unknown"
                exp_name = "unknown"
                timestamp = "unknown"

        max_val = get_max_value(tb_log_path, tag)
        
        if max_val is not None:
            results.append({
                "Model": model_name,
                "Experiment/Split": exp_name,
                "Timestamp": timestamp,
                f"Max {tag}": f"{max_val:.4f}",
                "Path": str(tb_log_path)
            })
            print(f"Found: {model_name} | {exp_name} | Max: {max_val:.4f}")

    if not results:
        print("No results found. Check your base directory and tag.")
        return

    # Write to CSV
    keys = results[0].keys()
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
    
    print(f"\nSuccessfully saved {len(results)} results to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Batch process TensorBoard logs and export to CSV.")
    parser.add_argument("base_dir", type=str, help="Base output directory to search (e.g., 'outputs').")
    parser.add_argument("--tag", type=str, default="val/accuracy", help="Metric tag to search for.")
    parser.add_argument("--output", type=str, default="experiment_results.csv", help="Output CSV filename.")

    args = parser.parse_args()

    process_all_logs(args.base_dir, args.tag, args.output)

if __name__ == "__main__":
    main()
