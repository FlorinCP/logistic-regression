import time

from data import load_adult_train_data, load_adult_valid_data
from sgd import predict, accuracy, extract_features, train, submission
import random


def main():

    random.seed(int(time.time() * 1000))

    train_data = extract_features(load_adult_train_data())
    print(f"Loaded {len(train_data)} training examples")

    valid_data = extract_features(load_adult_valid_data())
    print(f"Loaded {len(valid_data)} validation examples")

    print("\nTraining model...")
    model = submission(train_data)

    train_predictions = [predict(model, p) for p in train_data]
    train_acc = accuracy(train_data, train_predictions)
    print(f"Training Accuracy: {train_acc:.4f}")

    valid_predictions = [predict(model, p) for p in valid_data]
    valid_acc = accuracy(valid_data, valid_predictions)
    print(f"Validation Accuracy: {valid_acc:.4f}")

    print("\nModel weights:")
    feature_names = [
        "bias", "age", "education_num", "hours_per_week",
        "married", "male", "capital_gain", "capital_loss"
    ]
    for name, weight in zip(feature_names, model):
        print(f"{name}: {weight:.4f}")


if __name__ == "__main__":
    main()