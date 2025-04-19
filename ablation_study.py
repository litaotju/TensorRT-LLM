#!/usr/bin/env python3
import argparse
import itertools
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

# Define all possible parameter variations for studies
ALL_PARAM_VARIATIONS = {
    'tp': [4, 8],
    'ep': [4, 8],
    'max_batch_size': [64, 128, 256, 384, 512],
    'max_num_tokens': [512, 1024, 1536, 2048],
    'concurrency': [512, 1024, 2048, 3072, 4096],
    'kv_cache_free_gpu_mem_fraction': [0.7, 0.8, 0.85, 0.9],
    'use_cuda_graph': [True, False],
    'cuda_graph_padding_enabled': [True, False],
    'enable_overlap_scheduler': [True, False],
    'enable_attention_dp': [True, False],
    'num_requests': [4, 400, 4000, 10000, 16000, 32000, 64000]
}


@dataclass
class BenchmarkParams:
    """Class to represent benchmark parameters with proper typing and documentation."""
    # Model parameters
    model_path: str = "/home/scratch.trt_llm_data/llm-models/DeepSeek-R1/DeepSeek-R1-FP4"
    dataset_path: str = "./dataset.txt"

    # Dataset parameters
    dataset_tokenizer: str = "nvidia/DeepSeek-R1-FP4"
    dataset_type: str = "token-norm-dist"
    dataset_input_mean: int = 1024
    dataset_output_mean: int = 2048
    dataset_input_stdev: int = 0
    dataset_output_stdev: int = 0
    dataset_num_requests: int = 49152

    # Parallelism parameters
    tp: int = 8  # Tensor Parallelism
    ep: int = 8  # Expert Parallelism

    # Benchmark configuration
    warmup: int = 0
    num_requests: int = 10000
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

    # Stream and capture output
    for line in process.stdout:
        print(line.rstrip())  # Print line in real-time (strip trailing newline)
        output_lines.append(line)  # Store for later analysis

    # Wait for process to complete and check return code
    return_code = process.wait()
    output = ''.join(output_lines)

    end_time = time.time()
    elapsed = end_time - start_time

    timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    print(f"End time: {timestamp}")
    print(f"Elapsed time: {elapsed:.2f} seconds")

    # Handle non-zero return code
    if return_code != 0:
        print(f"Command failed with return code {return_code}")
        print(f"Command was: {' '.join(cmd)}")

    # Save output to a file in the run directory
    output_file = os.path.join(run_dir, f'run_{run_id}_output.txt')
    with open(output_file, 'w') as f:
        f.write(output)

    # Extract metrics from the output
    metrics = extract_metrics_from_output(output)

    # Check if we found any metrics
    if not metrics:
        print("Could not find performance metrics in output")
    else:
        print(f"Found {len(metrics)} performance metrics")

    result = {
        'run_id': run_id,
        'params': params.to_dict(),
        'metrics': metrics,
        'elapsed_time': elapsed,
        'raw_output': output,
        'return_code': return_code,
        'output_file': output_file,
        'config_file': config_path,
        'repro_script': repro_script_path
    }

    return result


def run_baseline(base_params: BenchmarkParams,
                 run_dir: str) -> List[Dict[str, Any]]:
    """Run a single benchmark with the base parameters."""
    print("Running benchmark with base parameters...")
    result = run_benchmark(base_params, run_dir, 0)
    return [result]  # Return as a list for consistency with other modes


def run_ablation_study(base_params: BenchmarkParams,
                       param_variations: Dict[str, List[Any]],
                       run_dir: str) -> List[Dict[str, Any]]:
    """Run ablation study by varying one parameter at a time."""
    results = []

    # First run with base parameters
    print("Running benchmark with base parameters...")
    base_result = run_benchmark(base_params, run_dir, 0)
    results.append(base_result)

    # For each parameter, run with different values
    run_id = 1
    for param_name, param_values in param_variations.items():
        current_value = getattr(base_params, param_name)
        for value in param_values:
            if value == current_value:
                continue  # Skip if it's the same as base value

            print(f"Running benchmark with {param_name} = {value}...")
            modified_params = base_params.create_variation(param_name, value)
            result = run_benchmark(modified_params, run_dir, run_id)
            run_id += 1
            results.append(result)

    return results


def run_grid_search(base_params: BenchmarkParams, param_grid: Dict[str,
                                                                   List[Any]],
                    run_dir: str) -> List[Dict[str, Any]]:
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
        run_id += 1
        results.append(result)

    return results


def save_results(results: List[Dict[str, Any]], filename: str) -> pd.DataFrame:
    """Save results to a CSV file."""
    rows = []
    for i, result in enumerate(results):
        # Start with run ID and elapsed time
        row = {'run_id': i, 'elapsed_time': result['elapsed_time']}

        # Add all metrics
        for metric_name, metric_value in result['metrics'].items():
            row[metric_name] = metric_value

        # Add all parameters
        for key, value in result['params'].items():
            if isinstance(value, list):
                row[key] = str(value)
            else:
                row[key] = value

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    return df


def plot_results(df: pd.DataFrame,
                 x_param: str,
                 y_param: str = 'total_output_throughput',
                 output_file: Optional[str] = None) -> None:
    """
    Create a plot showing the effect of a parameter on a performance metric.

    Args:
        df: DataFrame with results
        x_param: Parameter to plot on x-axis
        y_param: Metric to plot on y-axis (default: total_output_throughput)
        output_file: Optional file to save plot to
    """
    plt.figure(figsize=(12, 6))

    if isinstance(df[x_param].iloc[0], list) or df[x_param].dtype == 'object':
        # For parameters like cuda_graph_batch_sizes that are lists
        # We need a simpler representation for the x-axis
        df['param_repr'] = df[x_param].apply(lambda x: str(x)[:10] + '...'
                                             if len(str(x)) > 10 else str(x))
        # Create the bar plot
        ax = sns.barplot(x='param_repr', y=y_param, data=df)

        # Add value annotations to each bar
        for i, p in enumerate(ax.patches):
            value = p.get_height()
            ax.annotate(f'{value:.1f}', (p.get_x() + p.get_width() / 2., value),
                        ha='center',
                        va='bottom',
                        fontsize=9)

        plt.xticks(rotation=45)
    else:
        # For numeric parameters
        ax = sns.lineplot(x=x_param, y=y_param, data=df, marker='o')

        # Add value annotations to each point
        for x, y in zip(df[x_param], df[y_param]):
            ax.annotate(
                f'{y:.1f}',
                (x, y),
                xytext=(0, 10),  # 10 points vertical offset
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=9)

    metric_name = y_param.replace('_', ' ').title()
    plt.title(f'Effect of {x_param} on {metric_name}')
    plt.ylabel(metric_name)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()


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
        results = run_baseline(base_params, run_dir)
    elif args.mode == 'ablation':
        results = run_ablation_study(base_params, param_variations, run_dir)
    else:  # grid search
        results = run_grid_search(base_params, param_grid, run_dir)

    # Save results
    df = save_results(results, results_file)

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

    # Skip plotting for baseline mode or if no-plots is specified
    if args.mode == 'baseline' or args.no_plots:
        print("Skipping plots generation")
        return

    # Create plots for each parameter
    # Key metrics to plot
    key_metrics = ['per_gpu_throughput', 'per_user_throughput', 'elapsed_time']

    for param in param_variations.keys():
        param_results = [r for r in results if param in r['params']]
        if len(param_results
               ) > 1:  # Only plot if we have multiple values for this param
            plot_df = df[~df[param].isna()]
            if len(plot_df) > 1:
                for metric in key_metrics:
                    if metric in df.columns:
                        plot_file = os.path.join(plots_dir,
                                                 f'{param}_vs_{metric}.png')
                        plot_results(plot_df,
                                     x_param=param,
                                     y_param=metric,
                                     output_file=plot_file)


if __name__ == '__main__':
    main()
