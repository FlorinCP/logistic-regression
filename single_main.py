import time
from data import load_adult_train_data, load_adult_valid_data
from sgd import predict, accuracy, extract_features, train, submission
import random
import matplotlib.pyplot as plt


def run_simple_sgd():
    # Track start time
    start_time = time.time()

    # Load data and track loading time
    load_start = time.time()
    train_data = extract_features(load_adult_train_data())
    valid_data = extract_features(load_adult_valid_data())
    load_time = time.time() - load_start

    print(f"Loaded {len(train_data)} training examples")
    print(f"Loaded {len(valid_data)} validation examples")
    print(f"Data loading time: {load_time:.2f} seconds")

    # Train model and track training time
    print("\nTraining model...")
    train_start = time.time()
    model = submission(train_data)
    train_time = time.time() - train_start
    print(f"Training time: {train_time:.2f} seconds")

    # Evaluate and track prediction time
    eval_start = time.time()
    train_predictions = [predict(model, p) for p in train_data]
    train_acc = accuracy(train_data, train_predictions)

    valid_predictions = [predict(model, p) for p in valid_data]
    valid_acc = accuracy(valid_data, valid_predictions)
    eval_time = time.time() - eval_start

    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {valid_acc:.4f}")
    print(f"Evaluation time: {eval_time:.2f} seconds")

    # Print feature weights
    feature_names = [
        "bias", "age", "education_num", "hours_per_week",
        "married", "male", "capital_gain", "capital_loss"
    ]
    print("\nModel weights:")
    for name, weight in zip(feature_names, model):
        print(f"{name}: {weight:.4f}")

    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

    # Return timing and performance metrics
    return {
        'load_time': load_time,
        'train_time': train_time,
        'eval_time': eval_time,
        'total_time': total_time,
        'train_acc': train_acc,
        'valid_acc': valid_acc,
        'model': model
    }


def plot_timing_breakdown(metrics):
    # Create timing breakdown plot
    plt.figure(figsize=(8, 6))
    times = [metrics['load_time'], metrics['train_time'], metrics['eval_time']]
    labels = ['Data Loading', 'Training', 'Evaluation']
    plt.bar(labels, times)
    plt.title('Simple SGD Timing Breakdown')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.savefig('simple_sgd_timing.png')
    plt.close()


if __name__ == "__main__":
    random.seed(int(time.time() * 1000))
    metrics = run_simple_sgd()
    plot_timing_breakdown(metrics)