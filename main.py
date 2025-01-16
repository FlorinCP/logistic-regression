import time
import matplotlib.pyplot as plt
from data import load_adult_train_data, load_adult_valid_data
from sgd import predict, accuracy, extract_features, train, submission
from distributed_sgd import DistributedSGD


def run_comparison():
    train_data = extract_features(load_adult_train_data())
    valid_data = extract_features(load_adult_valid_data())

    start_time = time.time()
    simple_model = submission(train_data)
    simple_time = time.time() - start_time

    simple_train_acc = accuracy(train_data, [predict(simple_model, p) for p in train_data])
    simple_valid_acc = accuracy(valid_data, [predict(simple_model, p) for p in valid_data])

    worker_counts = [1, 2, 4, 8]
    distributed_times = []
    distributed_train_accs = []
    distributed_valid_accs = []

    for workers in worker_counts:
        print(f"\nRunning distributed SGD with {workers} workers...")
        start_time = time.time()

        sgd = DistributedSGD(num_workers=workers)
        dist_model = sgd.train(train_data, epochs=20, learning_rate=0.001, reg_lambda=0.001)

        distributed_times.append(time.time() - start_time)
        distributed_train_accs.append(accuracy(train_data, [predict(dist_model, p) for p in train_data]))
        distributed_valid_accs.append(accuracy(valid_data, [predict(dist_model, p) for p in valid_data]))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(worker_counts, distributed_times, 'bo-', label='Distributed SGD')
    plt.axhline(y=simple_time, color='r', linestyle='--', label='Simple SGD')
    plt.xlabel('Number of Workers')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time Comparison')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(worker_counts, distributed_train_accs, 'bo-', label='Dist Train Acc')
    plt.plot(worker_counts, distributed_valid_accs, 'go-', label='Dist Valid Acc')
    plt.axhline(y=simple_train_acc, color='r', linestyle='--', label='Simple Train Acc')
    plt.axhline(y=simple_valid_acc, color='m', linestyle='--', label='Simple Valid Acc')
    plt.xlabel('Number of Workers')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('sgd_comparison.png')
    plt.close()

    # Print summary
    print("\nResults Summary:")
    print("-" * 50)
    print("Simple SGD:")
    print(f"Time: {simple_time:.2f}s")
    print(f"Train Accuracy: {simple_train_acc:.4f}")
    print(f"Valid Accuracy: {simple_valid_acc:.4f}")
    print("\nDistributed SGD:")
    for i, workers in enumerate(worker_counts):
        print(f"\n{workers} Workers:")
        print(f"Time: {distributed_times[i]:.2f}s")
        print(f"Train Accuracy: {distributed_train_accs[i]:.4f}")
        print(f"Valid Accuracy: {distributed_valid_accs[i]:.4f}")


if __name__ == "__main__":
    run_comparison()