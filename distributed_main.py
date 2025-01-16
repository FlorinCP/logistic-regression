import time
from data import load_adult_train_data, load_adult_valid_data
from sgd import extract_features, predict
from distributed_sgd import DistributedSGD
import json
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt


def evaluate_configuration(
        train_data: List[Dict],
        valid_data: List[Dict],
        num_workers: int,
        epochs: int,
        learning_rate: float = 0.001,
        reg_lambda: float = 0.001
) -> Tuple[List[float], float, float, float]:
    """Evaluate model with specified number of workers"""
    print(f"\nEvaluating with {num_workers} workers...")

    sgd = DistributedSGD(num_workers=num_workers)

    start_time = time.time()
    final_model = sgd.train(
        train_data,
        epochs=epochs,
        learning_rate=learning_rate,
        reg_lambda=reg_lambda
    )
    total_time = time.time() - start_time

    # Calculate accuracies
    train_pred = [predict(final_model, p) for p in train_data]
    valid_pred = [predict(final_model, p) for p in valid_data]

    train_acc = sum(1 for p, pred in zip(train_data, train_pred)
                    if (pred >= 0.5) == p['label']) / len(train_data)
    valid_acc = sum(1 for p, pred in zip(valid_data, valid_pred)
                    if (pred >= 0.5) == p['label']) / len(valid_data)

    print(f"\nResults for {num_workers} workers:")
    print(f"Training time: {total_time:.2f} seconds")
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {valid_acc:.4f}")

    return final_model, total_time, train_acc, valid_acc


def plot_results(results: Dict):
    """Create plots comparing different configurations"""
    # Prepare data for plotting
    workers = sorted(list(results.keys()))
    times = [results[w]['time'] for w in workers]
    train_accs = [results[w]['train_acc'] for w in workers]
    valid_accs = [results[w]['valid_acc'] for w in workers]

    # Calculate speedup
    base_time = results[1]['time']  # time for single worker
    speedups = [base_time / time for time in times]

    # Create figure with subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # Plot 1: Time and Speedup
    ax1 = axs[0]
    ax1.plot(workers, times, 'bo-', label='Execution Time')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(workers, speedups, 'ro-', label='Speedup')

    ax1.set_xlabel('Number of Workers')
    ax1.set_ylabel('Execution Time (s)', color='b')
    ax1_twin.set_ylabel('Speedup', color='r')
    ax1.grid(True)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Plot 2: Accuracy
    ax2 = axs[1]
    ax2.plot(workers, train_accs, 'go-', label='Training Accuracy')
    ax2.plot(workers, valid_accs, 'mo-', label='Validation Accuracy')
    ax2.set_xlabel('Number of Workers')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('distributed_sgd_results.png')
    plt.close()


def main():
    """Main function to run the distributed SGD analysis"""
    print("Starting distributed SGD analysis...")

    # Load and prepare data
    train_data = extract_features(load_adult_train_data())
    valid_data = extract_features(load_adult_valid_data())

    # Configuration parameters
    worker_counts = [1, 2, 4, 8]  # Number of workers to test
    epochs = 10
    learning_rate = 0.001
    reg_lambda = 0.001

    # Dictionary to store results
    results = {}

    # Evaluate each worker configuration
    for num_workers in worker_counts:
        model, time_taken, train_acc, valid_acc = evaluate_configuration(
            train_data,
            valid_data,
            num_workers,
            epochs,
            learning_rate,
            reg_lambda
        )

        # Store results
        results[num_workers] = {
            'time': time_taken,
            'train_acc': train_acc,
            'valid_acc': valid_acc
        }

    # Save results to file
    with open('distributed_sgd_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Create plots
    plot_results(results)

    # Print summary
    print("\nSummary of Results:")
    print("-" * 60)
    print(f"{'Workers':<10} {'Time (s)':<12} {'Train Acc':<12} {'Valid Acc':<12} {'Speedup':<10}")
    print("-" * 60)

    base_time = results[1]['time']  # time for single worker
    for workers in sorted(results.keys()):
        r = results[workers]
        speedup = base_time / r['time']
        print(f"{workers:<10} {r['time']:<12.2f} {r['train_acc']:<12.4f} "
              f"{r['valid_acc']:<12.4f} {speedup:<10.2f}")


if __name__ == "__main__":
    main()