import time
import matplotlib.pyplot as plt
from data import load_adult_train_data, load_adult_valid_data
from distributed_sgd import DistributedSGD
from sgd import predict, accuracy, extract_features
import random

def evaluate_epochs(train_data, valid_data, num_workers=4):
    """Evaluate model accuracy vs epochs"""
    epochs_range = [5, 10, 15, 20, 25, 30]
    train_accuracies = []
    valid_accuracies = []

    sgd = DistributedSGD(num_workers=num_workers)

    for epochs in epochs_range:
        model = sgd.train(train_data, epochs=epochs, learning_rate=0.001, reg_lambda=0.001)

        # Calculate accuracies
        train_pred = [predict(model, p) for p in train_data]
        valid_pred = [predict(model, p) for p in valid_data]

        train_accuracies.append(accuracy(train_data, train_pred))
        valid_accuracies.append(accuracy(valid_data, valid_pred))

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs_range, valid_accuracies, 'r-', label='Validation Accuracy')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy vs Number of Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_vs_epochs.png')
    plt.close()


def evaluate_workers(train_data, valid_data):
    """Evaluate execution time vs number of workers"""
    worker_range = [1, 2, 4, 8, 16]
    execution_times = []
    speedups = []

    # Baseline time with 1 worker
    sgd = DistributedSGD(num_workers=1)
    start_time = time.time()
    model = sgd.train(train_data, epochs=20, learning_rate=0.001, reg_lambda=0.001)
    baseline_time = time.time() - start_time
    execution_times.append(baseline_time)
    speedups.append(1.0)

    # Test with different numbers of workers
    for workers in worker_range[1:]:
        sgd = DistributedSGD(num_workers=workers)
        start_time = time.time()
        model = sgd.train(train_data, epochs=20, learning_rate=0.001, reg_lambda=0.001)
        exec_time = time.time() - start_time

        execution_times.append(exec_time)
        speedups.append(baseline_time / exec_time)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(worker_range, execution_times, 'b-o')
    ax1.set_xlabel('Number of Workers')
    ax1.set_ylabel('Execution Time (s)')
    ax1.set_title('Execution Time vs Number of Workers')
    ax1.grid(True)

    ax2.plot(worker_range, speedups, 'r-o')
    ax2.set_xlabel('Number of Workers')
    ax2.set_ylabel('Speedup')
    ax2.set_title('Speedup vs Number of Workers')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('performance_analysis.png')
    plt.close()


def main():
    random.seed(int(time.time() * 1000))

    # Load data
    train_data = extract_features(load_adult_train_data())
    print(f"Loaded {len(train_data)} training examples")

    valid_data = extract_features(load_adult_valid_data())
    print(f"Loaded {len(valid_data)} validation examples")

    # Perform evaluations
    print("\nEvaluating accuracy vs epochs...")
    evaluate_epochs(train_data, valid_data)

    print("\nEvaluating performance vs number of workers...")
    evaluate_workers(train_data, valid_data)


if __name__ == "__main__":
    main()