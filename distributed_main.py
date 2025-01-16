import time
from data import load_adult_train_data, load_adult_valid_data
from sgd import extract_features, predict
from distributed_sgd import DistributedSGD


def evaluate_single_worker(train_data, valid_data, epochs=5):
    """Evaluate model with a single worker"""
    print("\nEvaluating with single worker...")

    sgd = DistributedSGD(num_workers=3)

    start_time = time.time()
    final_model = sgd.train(train_data, epochs=epochs,
                            learning_rate=0.001, reg_lambda=0.001)
    total_time = time.time() - start_time

    train_pred = [predict(final_model, p) for p in train_data]
    valid_pred = [predict(final_model, p) for p in valid_data]

    train_acc = sum(1 for p, pred in zip(train_data, train_pred)
                    if (pred >= 0.5) == p['label']) / len(train_data)
    valid_acc = sum(1 for p, pred in zip(valid_data, valid_pred)
                    if (pred >= 0.5) == p['label']) / len(valid_data)

    print(f"\nResults:")
    print(f"Training time: {total_time:.2f} seconds")
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {valid_acc:.4f}")

    return final_model, total_time, train_acc, valid_acc


def main():
    """Main function to run the distributed SGD analysis"""

    train_data = extract_features(load_adult_train_data())
    valid_data = extract_features(load_adult_valid_data())

    model, time_taken, train_acc, valid_acc = evaluate_single_worker(
        train_data, valid_data, epochs=10
    )


if __name__ == "__main__":
    main()