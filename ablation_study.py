#!/usr/bin/env python3
import argparse
import itertools
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

# Define all possible parameter variations for studies
ALL_PARAM_VARIATIONS = {
    'tp': [4, 8],
    'ep': [4, 8],
    'max_batch_size': [64, 128, 256, 384, 512],
    'max_num_tokens': [1024, 1536, 2048],
    'concurrency': [512, 1024, 2048, 3072, 4096],
    'kv_cache_free_gpu_mem_fraction': [0.7, 0.8, 0.85, 0.9],
    'use_cuda_graph': [True, False],
    'cuda_graph_padding_enabled': [True, False],
    'enable_overlap_scheduler': [True, False],
    'enable_attention_dp': [True, False],
    'num_requests': [4, 400, 4000, 10000, 16000, 32000, 64000]
}

TIMEOUT_IN_SECONDS = 90 * 60  # 5400 seconds
DEBUG = os.environ.get('DEBUG', 'False').lower() in ['true', '1', 'yes', 'y']


@dataclass
class BenchmarkParams:
    """Class to represent benchmark parameters with proper typing and documentation."""
    # Model parameters
    model_path: str = "/home/scratch.trt_llm_data/llm-models/DeepSeek-R1/DeepSeek-R1-FP4"
    dataset_path: str = "./dataset.txt"

    # Dataset parameters
    dataset_tokenizer: str = "nvidia/DeepSeek-R1-FP4"
    dataset_type: str = "token-norm-dist"
    dataset_input_mean: int = 1024 if not DEBUG else 10
    dataset_output_mean: int = 2048 if not DEBUG else 10
    dataset_input_stdev: int = 0
    dataset_output_stdev: int = 0
    dataset_num_requests: int = 49152 if not DEBUG else 4

    # Parallelism parameters
    tp: int = 8  # Tensor Parallelism
    ep: int = 8  # Expert Parallelism

    # Benchmark configuration
    warmup: int = 0
    num_requests: int = 10000 if not DEBUG else 4
    concurrency: int = 3072
    max_batch_size: int = 384
    max_num_tokens: int = 1536
    kv_cache_free_gpu_mem_fraction: float = 0.85

    # pytorch_backend_config
    use_cuda_graph: bool = True
    cuda_graph_padding_enabled: bool = True
    cuda_graph_batch_sizes: List[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128, 256, 384])
    print_iter_log: bool = True
    enable_overlap_scheduler: bool = True

    # extra configs outside the "pytorch_backend_config" field
    enable_attention_dp: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, params_dict: Dict[str, Any]) -> 'BenchmarkParams':
        """Create a BenchmarkParams instance from a dictionary."""
        return cls(**params_dict)

    def create_variation(self, param_name: str,
                         value: Any) -> 'BenchmarkParams':
        """Create a new BenchmarkParams with one parameter changed."""
        params_dict = self.to_dict()
        params_dict[param_name] = value
        return self.from_dict(params_dict)


def extract_metrics_from_output(output: str) -> Dict[str, float]:
    """
    Extract all performance metrics from the benchmark output.

    Args:
        output: Raw text output from the benchmark run

    Returns:
        Dictionary mapping metric names to their values
    """
    metrics = {}

    # Define all the metrics to extract with their regex patterns
    metric_patterns = {
        'request_throughput': r'Request Throughput \(req/sec\):\s+([\d.]+)',
        'total_output_throughput':
        r'Total Output Throughput \(tokens/sec\):\s+([\d.]+)',
        'per_user_throughput':
        r'Per User Output Throughput \(tokens/sec/user\):\s+([\d.]+)',
        'per_gpu_throughput':
        r'Per GPU Output Throughput \(tokens/sec/gpu\):\s+([\d.]+)',
        'total_token_throughput':
        r'Total Token Throughput \(tokens/sec\):\s+([\d.]+)',
        'total_latency': r'Total Latency \(ms\):\s+([\d.]+)',
        'avg_request_latency': r'Average request latency \(ms\):\s+([\d.]+)',
        'latency_p50': r'\[Latency\] P50\s+: ([\d.]+)',
        'latency_p90': r'\[Latency\] P90\s+: ([\d.]+)',
        'latency_p95': r'\[Latency\] P95\s+: ([\d.]+)',
        'latency_p99': r'\[Latency\] P99\s+: ([\d.]+)',
        'latency_min': r'\[Latency\] MINIMUM: ([\d.]+)',
        'latency_max': r'\[Latency\] MAXIMUM: ([\d.]+)',
        'latency_avg': r'\[Latency\] AVERAGE: ([\d.]+)'
    }

    # Extract each metric using its regex pattern
    for metric_name, pattern in metric_patterns.items():
        match = re.search(pattern, output)
        if match:
            metrics[metric_name] = float(match.group(1))
        else:
            metrics[metric_name] = None

    return metrics


def run_benchmark(params: BenchmarkParams, run_dir: str,
                  run_id: int) -> Dict[str, Any]:
    """
    Run the benchmark with the given parameters and return results.

    Args:
        params: Benchmark parameters
        run_dir: Directory to store run artifacts
        run_id: Unique ID for this run
    """
    # Create config file in the run directory
    config_path = os.path.join(run_dir, f'config_{run_id}.yml')
    config = {
        'pytorch_backend_config': {
            'use_cuda_graph': params.use_cuda_graph,
            'cuda_graph_padding_enabled': params.cuda_graph_padding_enabled,
            'cuda_graph_batch_sizes': params.cuda_graph_batch_sizes,
            'print_iter_log': params.print_iter_log,
            'enable_overlap_scheduler': params.enable_overlap_scheduler
        },
        'enable_attention_dp': params.enable_attention_dp
    }

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # yapf: disable
    cmd = [
        'python3', '-m', 'tensorrt_llm.commands.bench',
        '-m', params.model_path,
        'throughput',
        '--tp', str(params.tp),
        '--ep', str(params.ep),
        '--warmup', str(params.warmup),
        '--dataset', params.dataset_path,
        '--backend', 'pytorch',
        '--max_batch_size', str(params.max_batch_size),
        '--max_num_tokens', str(params.max_num_tokens),
        '--num_requests', str(params.num_requests),
        '--concurrency', str(params.concurrency),
        '--kv_cache_free_gpu_mem_fraction', str(params.kv_cache_free_gpu_mem_fraction),
        '--extra_llm_api_options', config_path
    ]
    # yapf: enable

    # Create a reproducible shell script
    repro_script_path = os.path.join(run_dir, f'repro_{run_id}.sh')
    with open(repro_script_path, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write("# Reproducible script for benchmark run\n")
        f.write(f"# Run ID: {run_id}\n")

        # Add environment variables
        f.write("# Set environment variables\n")
        f.write("export TQDM_MININTERVAL=1000\n\n")

        # Add dataset generation command to repro script
        f.write("# Generate dataset\n")
        dataset_filename = "./dataset.txt"
        dataset_cmd = [
            'python',
            'benchmarks/cpp/prepare_dataset.py',
            '--stdout',
            '--tokenizer',
            params.dataset_tokenizer,
            params.dataset_type,
            '--input-mean',
            str(params.dataset_input_mean),
            '--output-mean',
            str(params.dataset_output_mean),
            '--input-stdev',
            str(params.dataset_input_stdev),
            '--output-stdev',
            str(params.dataset_output_stdev),
            '--num-requests',
            str(params.dataset_num_requests),
        ]
        f.write(f"{' '.join(dataset_cmd)} > {dataset_filename}\n\n")

        # Include the config file content
        with open(config_path, 'r') as config_file:
            config_content = config_file.read()
        f.write("# Create config file\n")
        f.write("cat > ./config.yml << 'EOL'\n")
        f.write(config_content)
        f.write("EOL\n\n")

        # Add the command, but use the newly created config file
        f.write("# Run benchmark command\n")
        # Copy the command but replace the config path
        cmd_copy = cmd.copy()
        for i, arg in enumerate(cmd_copy):
            if arg == config_path:
                cmd_copy[i] = "./config.yml"
            if arg == params.dataset_path:
                cmd_copy[i] = dataset_filename
        f.write(" ".join(cmd_copy) + "\n")

    # Make the script executable
    os.chmod(repro_script_path, 0o755)

    print(f"Running benchmark with parameters: {params}")
    start_time = time.time()

    timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    print(f"Start time: {timestamp}")

    # Set up environment with TQDM_MININTERVAL=1000
    env = os.environ.copy()
    env['TQDM_MININTERVAL'] = '1000'

    # Use Popen to be able to stream output in real-time
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Redirect stderr to stdout
        text=True,
        bufsize=1,  # Line buffered
        universal_newlines=True,
        env=env  # Use the modified environment
    )

    # Collect output for later analysis
    output_lines = []

    # Set timeout to 1.5 hours (90 minutes)
    timeout_seconds = TIMEOUT_IN_SECONDS
    timed_out = False

    # Stream and capture output with timeout check
    while True:
        # Check if process has completed
        if process.poll() is not None:
            break

        # Check for timeout
        current_time = time.time()
        if current_time - start_time > timeout_seconds:
            print(
                f"Benchmark timeout after {timeout_seconds/60:.1f} minutes. Terminating process."
            )
            process.terminate()
            try:
                # Wait a bit for graceful termination
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate
                print("Process not responding to termination. Force killing.")
                process.kill()
                process.wait()
            timed_out = True
            break

        # Read output with a timeout to allow for regular timeout checks
        try:
            line = process.stdout.readline()
            if not line:
                # No more output but process is still running
                time.sleep(1)  # Avoid busy waiting
                continue

            print(line.rstrip())  # Print line in real-time
            output_lines.append(line)  # Store for later analysis

        except Exception as e:
            print(f"Error reading process output: {e}")
            break

    # Consume any remaining output
    for line in process.stdout:
        print(line.rstrip())
        output_lines.append(line)

    # Get return code
    return_code = process.wait()
    output = ''.join(output_lines)

    end_time = time.time()
    elapsed = end_time - start_time

    timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    print(f"End time: {timestamp}")
    print(f"Elapsed time: {elapsed:.2f} seconds")

    # Handle non-zero return code
    if timed_out:
        print(f"Benchmark timed out after {timeout_seconds/60:.1f} minutes")
        # Add timeout information to output
        timeout_msg = f"\n\nBENCHMARK TIMED OUT AFTER {timeout_seconds/60:.1f} MINUTES\n\n"
        output += timeout_msg
        return_code = -999  # Special code for timeout
    elif return_code != 0:
        print(f"Command failed with return code {return_code}")
        print(f"Command was: {' '.join(cmd)}")

    # Save output to a file in the run directory
    output_file = os.path.join(run_dir, f'run_{run_id}_output.txt')
    with open(output_file, 'w') as f:
        f.write(output)

    # Extract metrics from the output
    metrics = extract_metrics_from_output(output)

    result = {
        'run_id': run_id,
        'params': params.to_dict(),
        'metrics': metrics,
        'elapsed_time': elapsed,
        'raw_output': output,
        'return_code': return_code,
        'output_file': output_file,
        'config_file': config_path,
        'repro_script': repro_script_path,
        'timed_out': timed_out
    }

    return result


def run_baseline(base_params: BenchmarkParams, run_dir: str,
                 results_file: str) -> List[Dict[str, Any]]:
    """Run a single benchmark with the base parameters."""
    print("Running benchmark with base parameters...")
    result = run_benchmark(base_params, run_dir, 0)

    # Save result immediately
    save_single_result(result, results_file)

    return [result]  # Return as a list for consistency with other modes


def run_ablation_study(
        base_params: BenchmarkParams, param_variations: Dict[str, List[Any]],
        run_dir: str,
        results_file: str) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
    """Run ablation study by varying one parameter at a time."""
    results = []
    run_ids_of_each_param = {}

    # First run with base parameters
    print("Running benchmark with base parameters...")
    base_result = run_benchmark(base_params, run_dir, 0)
    base_result['ablation'] = 'base'
    save_single_result(base_result, results_file)
    results.append(base_result)

    # For each parameter, run with different values
    run_id = 1
    for param_name, param_values in param_variations.items():
        run_ids_of_each_param[param_name] = [0]
        current_value = getattr(base_params, param_name)
        for value in param_values:
            if value == current_value:
                continue  # Skip if it's the same as base value

            print(f"Running benchmark with {param_name} = {value}...")
            modified_params = base_params.create_variation(param_name, value)
            result = run_benchmark(modified_params, run_dir, run_id)
            result['ablation'] = param_name
            run_ids_of_each_param[param_name].append(run_id)
            save_single_result(result, results_file)
            run_id += 1
            results.append(result)

    return results, run_ids_of_each_param


def run_grid_search(base_params: BenchmarkParams, param_grid: Dict[str,
                                                                   List[Any]],
                    run_dir: str, results_file: str) -> List[Dict[str, Any]]:
    """Run a grid search over parameter combinations."""
    results = []

    # Generate all combinations of parameters
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]

    run_id = 0
    for values in itertools.product(*param_values):
        params_dict = base_params.to_dict()
        for name, value in zip(param_names, values):
            params_dict[name] = value

        modified_params = BenchmarkParams.from_dict(params_dict)
        result = run_benchmark(modified_params, run_dir, run_id)
        save_single_result(result, results_file)
        run_id += 1
        results.append(result)

    return results


def save_single_result(result: Dict[str, Any], results_file: str) -> None:
    """Save a single benchmark result to CSV file, creating or appending as needed."""
    # Create row from the result
    row = {'run_id': result['run_id'], 'elapsed_time': result['elapsed_time']}

    # Add all metrics
    for metric_name, metric_value in result['metrics'].items():
        row[metric_name] = metric_value

    # Add all parameters
    for key, value in result['params'].items():
        if isinstance(value, list):
            row[key] = str(value)
        else:
            row[key] = value

    # Add timeout status
    row['timed_out'] = result.get('timed_out', False)

    # Create DataFrame with single row
    df = pd.DataFrame([row])

    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(results_file)

    # Write to CSV (append if file exists)
    if file_exists:
        df.to_csv(results_file, mode='a', header=False, index=False)
    else:
        df.to_csv(results_file, index=False)

    print(f"Result for run_id {result['run_id']} saved to {results_file}")


def validate_study_params(parser, study_params_str):
    """Validate study parameters against available choices."""
    params = study_params_str.split(',')
    valid_params = list(ALL_PARAM_VARIATIONS.keys()) + ['all']

    invalid_params = [
        param.strip() for param in params
        if param.strip() != 'all' and param.strip() not in ALL_PARAM_VARIATIONS
    ]

    if invalid_params:
        parser.error(
            f"Invalid parameter(s) in --study-params: {', '.join(invalid_params)}. "
            f"Valid choices are: {', '.join(valid_params)}")

    return study_params_str


def plot_benchmark_results(results_file: str, plots_dir: str,
                           param_variations: Dict[str, List[Any]],
                           run_ids_of_each_param: Dict[str, List[int]]) -> None:
    """
    Create plots for benchmark results saved in a CSV file.

    Args:
        results_file: Path to the CSV file with benchmark results
        plots_dir: Directory to save the plots
        param_variations: Dictionary of parameter names and their possible values
    """
    print(f"Generating plots from results in {results_file}")

    # Ensure plots directory exists
    os.makedirs(plots_dir, exist_ok=True)

    # Key metrics to plot
    key_metrics = ['per_gpu_throughput', 'per_user_throughput', 'elapsed_time']

    try:
        # Create plots for each parameter that has multiple values in the dataset
        for param in param_variations.keys():
            # Load results from CSV file
            df = pd.read_csv(results_file)
            if len(df) <= 1:
                print(
                    "Not enough data points for plotting. Skipping plot generation."
                )
                return
            # filter out the rows where the run_id is not in the run_ids_of_each_param[param]
            df = df[df['run_id'].isin(run_ids_of_each_param[param])]
            if param not in df.columns:
                print(
                    f"Parameter '{param}' not found in results file. Skipping.")
                continue

            # Convert parameter column to appropriate type if needed
            # This handles cases where True/False might be stored as strings
            if df[param].dtype == object:
                try:
                    if all(
                            str(val).lower() in ['true', 'false']
                            for val in df[param].unique() if pd.notna(val)):
                        df[param] = df[param].apply(lambda x: str(x).lower(
                        ) == 'true' if pd.notna(x) else x)
                except:
                    pass  # Keep as object if conversion fails

            # Check if we have multiple values for this parameter
            unique_values = df[param].dropna().unique()
            if len(unique_values) <= 1:
                print(f"Parameter '{param}' has only one value. Skipping plot.")
                continue

            # Filter rows where this parameter is not null
            plot_df = df[~df[param].isna()]

            if len(plot_df) > 1:
                print(f"Creating plots for parameter: {param}")
                for metric in key_metrics:
                    if metric not in plot_df.columns:
                        print(
                            f"Metric '{metric}' not found in results. Skipping."
                        )
                        continue

                    # Skip if all values for this metric are null
                    if plot_df[metric].isna().all():
                        print(
                            f"All values for metric '{metric}' are null. Skipping."
                        )
                        continue

                    plot_file = os.path.join(plots_dir,
                                             f'{param}_vs_{metric}.png')

                    plt.figure(figsize=(12, 6))

                    # Use different plot types depending on parameter data type
                    if plot_df[param].dtype == bool or plot_df[
                            param].dtype == object:
                        # For categorical parameters, use bar plot
                        ax = sns.barplot(x=param, y=metric, data=plot_df)
                        plt.xticks(rotation=45)
                    else:
                        # For numeric parameters, use line plot
                        # Sort by parameter value to ensure line connects points in order
                        plot_df = plot_df.sort_values(by=param)
                        ax = sns.lineplot(x=param,
                                          y=metric,
                                          data=plot_df,
                                          marker='o')

                    # Add value annotations to data points
                    for i, (x_val, y_val) in enumerate(
                            zip(plot_df[param], plot_df[metric])):
                        if pd.notna(y_val):  # Only annotate non-NaN values
                            ax.annotate(
                                f'{y_val:.1f}',
                                (i if plot_df[param].dtype == bool
                                 or plot_df[param].dtype == object else x_val,
                                 y_val),
                                xytext=(0, 10),  # 10 points vertical offset
                                textcoords='offset points',
                                ha='center',
                                va='bottom',
                                fontsize=9)

                    # Set chart labels and title
                    metric_title = metric.replace("_", " ").title()
                    param_title = param.replace("_", " ").title()
                    plt.title(f'Effect of {param_title} on {metric_title}')
                    plt.ylabel(metric_title)
                    plt.tight_layout()

                    # Save the plot
                    plt.savefig(plot_file)
                    plt.close()  # Close the figure to free memory
                    print(f"Saved plot to {plot_file}")

    except Exception as e:
        print(f"Error generating plots: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Run ablation study on TRT-LLM benchmarking parameters')
    parser.add_argument(
        '--mode',
        choices=['baseline', 'ablation', 'grid'],
        default='baseline',
        help=
        'Study mode: baseline (single run), ablation (change one parameter at a time), '
        'or grid (try all combinations)')
    parser.add_argument('--output-dir',
                        default='benchmark_runs',
                        help='Directory to store all benchmark artifacts')
    parser.add_argument('--no-plots',
                        action='store_true',
                        help='Skip generating plots')
    # Ablation study parameters
    parser.add_argument(
        '--study-params',
        default='use_cuda_graph',
        help='Comma-separated list of parameters to study in ablation mode. '
        'Valid choices: ' + ', '.join(ALL_PARAM_VARIATIONS.keys()) +
        ', all. Default: use_cuda_graph')

    # Dataset generation parameters
    parser.add_argument('--dataset-type', help='Dataset type')
    parser.add_argument('--dataset-input-mean',
                        type=int,
                        help='Mean input length for dataset generation')
    parser.add_argument('--dataset-output-mean',
                        type=int,
                        help='Mean output length for dataset generation')
    parser.add_argument(
        '--dataset-input-stdev',
        type=int,
        help='Standard deviation of input length for dataset generation')
    parser.add_argument(
        '--dataset-output-stdev',
        type=int,
        help='Standard deviation of output length for dataset generation')
    parser.add_argument('--dataset-num-requests',
                        type=int,
                        help='Number of requests in the dataset')

    args = parser.parse_args()

    # Validate study parameters if in ablation mode
    if args.mode == 'ablation':
        validate_study_params(parser, args.study_params)

    # Create timestamped run directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{args.mode}_{timestamp}"
    run_dir = os.path.join(args.output_dir, run_name)
    # you dont want to overwrite existing runs
    os.makedirs(run_dir, exist_ok=False)
    print(f"Created run directory: {run_dir}")

    # Create directory for plots
    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Results file path
    results_file = os.path.join(run_dir, 'results.csv')

    # Path to dataset in the run directory
    dataset_path = os.path.join(run_dir, 'dataset.txt')

    # Create base parameters using the dataclass
    base_params = BenchmarkParams()
    base_params.dataset_path = dataset_path  # Use dataset in run directory

    # Update dataset parameters from command line args if provided
    if args.dataset_type is not None:
        base_params.dataset_type = args.dataset_type
    if args.dataset_input_mean is not None:
        base_params.dataset_input_mean = args.dataset_input_mean
    if args.dataset_output_mean is not None:
        base_params.dataset_output_mean = args.dataset_output_mean
    if args.dataset_input_stdev is not None:
        base_params.dataset_input_stdev = args.dataset_input_stdev
    if args.dataset_output_stdev is not None:
        base_params.dataset_output_stdev = args.dataset_output_stdev
    if args.dataset_num_requests is not None:
        base_params.dataset_num_requests = args.dataset_num_requests

    # Use only the specified parameters for the ablation study
    if args.mode == 'ablation':
        study_params = args.study_params.split(',')
        param_variations = {}

        # Special case for 'all'
        if 'all' in study_params:
            print("Studying all parameters as requested")
            param_variations = ALL_PARAM_VARIATIONS
        else:
            # Process individual parameters
            for param in study_params:
                param = param.strip()
                if param in ALL_PARAM_VARIATIONS:
                    param_variations[param] = ALL_PARAM_VARIATIONS[param]
                else:
                    print(
                        f"Warning: Unknown parameter '{param}'. Valid parameters: "
                        f"{', '.join(ALL_PARAM_VARIATIONS.keys())}")

            if not param_variations:
                print(
                    "No valid parameters specified for ablation study. Using default (use_cuda_graph)."
                )
                if 'use_cuda_graph' in ALL_PARAM_VARIATIONS:
                    param_variations['use_cuda_graph'] = ALL_PARAM_VARIATIONS[
                        'use_cuda_graph']
    else:
        # For other modes, use all parameter variations
        param_variations = ALL_PARAM_VARIATIONS

    # For grid search, we use a smaller set of parameters to avoid combinatorial explosion
    param_grid = {
        'tp': [4, 8],
        'ep': [4, 8],
        'max_batch_size': [256, 384],
        'concurrency': [2048, 3072]
    }

    # Make sure dataset exists
    if not os.path.exists(dataset_path):
        print(f"Dataset not found. Creating dataset at {dataset_path}...")
        prepare_cmd = [
            'python', 'benchmarks/cpp/prepare_dataset.py', '--stdout',
            '--tokenizer', base_params.dataset_tokenizer,
            base_params.dataset_type, '--input-mean',
            str(base_params.dataset_input_mean), '--output-mean',
            str(base_params.dataset_output_mean), '--input-stdev',
            str(base_params.dataset_input_stdev), '--output-stdev',
            str(base_params.dataset_output_stdev), '--num-requests',
            str(base_params.dataset_num_requests)
        ]

        with open(dataset_path, 'w') as f:
            subprocess.run(prepare_cmd, stdout=f, check=True)

    # Run the study based on the selected mode
    if args.mode == 'baseline':
        results = run_baseline(base_params, run_dir, results_file)
    elif args.mode == 'ablation':
        results, run_ids_of_each_param = run_ablation_study(
            base_params, param_variations, run_dir, results_file)
        # Create plots from the results file
        plot_benchmark_results(results_file, plots_dir, param_variations,
                               run_ids_of_each_param)
    else:  # grid search
        results = run_grid_search(base_params, param_grid, run_dir,
                                  results_file)

    # Create symlink to latest run
    latest_link = os.path.join(args.output_dir, 'latest')
    if os.path.exists(latest_link):
        if os.path.islink(latest_link):
            os.unlink(latest_link)
        else:
            os.remove(latest_link)

    try:
        os.symlink(run_name, latest_link, target_is_directory=True)
        print(f"Created symlink to latest run: {latest_link}")
    except (OSError, NotImplementedError):
        # Symlinks might not be supported on some systems
        print(
            f"Could not create symlink to latest run. Latest run is: {run_dir}")


if __name__ == '__main__':
    main()
