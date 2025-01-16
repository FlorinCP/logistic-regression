from math import exp
import random

def logistic(x):
    """Compute sigmoid function safely."""
    if x < -30:
        return 0
    elif x > 30:
        return 1
    return 1.0 / (1.0 + exp(-x))


def dot(x, y):
    return sum(float(xi) * float(yi) for xi, yi in zip(x, y))


def predict(model, point):
    return logistic(dot(model, point['features']))


def accuracy(data, predictions):
    correct = 0
    for point, pred in zip(data, predictions):
        if (pred >= 0.5) == point['label']:
            correct += 1
    return float(correct) / len(data)


def compute_gradient(point, prediction):
    """
    Compute gradient for logistic regression.
    Returns gradient vector same length as features.
    """
    error = prediction - (1.0 if point['label'] else 0.0)
    gradient = []
    for feature in point['features']:
        grad_component = error * feature
        gradient.append(grad_component)
    return gradient


def update(model, point, learning_rate, reg_lambda):
    """
    Update model weights using computed gradient.
    Includes L2 regularization.
    """
    prediction = predict(model, point)
    gradient = compute_gradient(point, prediction)

    # Update each weight using gradient and regularization
    for j in range(len(model)):
        reg_term = reg_lambda * model[j]
        model[j] -= learning_rate * (gradient[j] + reg_term)


def train(data, epochs, learning_rate, reg_lambda):
    model = initialize_model(len(data[0]['features']))
    print("\nInitial model weights:", [f"{w:.4f}" for w in model])

    best_accuracy = 0
    best_model = None

    for epoch in range(epochs):
        random.shuffle(data)
        epoch_loss = 0

        for i, point in enumerate(data):
            if i % 5000 == 0:
                print(f"Processing point {i}/{len(data)}")

            update(model, point, learning_rate, reg_lambda)

            prediction = predict(model, point)
            error = prediction - (1.0 if point['label'] else 0.0)
            epoch_loss += error * error

        predictions = [predict(model, p) for p in data]
        current_accuracy = accuracy(data, predictions)

        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_model = model.copy()

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Loss: {epoch_loss / len(data):.4f}")
        print(f"Accuracy: {current_accuracy:.4f}")
        print(f"Best Accuracy so far: {best_accuracy:.4f}")
        print()

    return best_model if best_model is not None else model


def initialize_model(k):
    """
    [-0.1, 0.1] random initialization for weights in order to compensate for large number of features.
    """
    return [random.uniform(-0.1, 0.1) for _ in range(k)]


def extract_features(raw):
    """Convert raw data into numerical features."""
    data = []
    for r in raw:
        point = {}
        point["label"] = (r['income'].strip() == '>50K')

        features = [
            1.0,  # bias term
            float(r['age']) / 100.0,
            float(r['education_num']) / 16.0,
            float(r['hr_per_week']) / 100.0,
            1.0 if r['marital'] == 'Married-civ-spouse' else 0.0,
            1.0 if r['sex'] == 'Male' else 0.0,
            float(r['capital_gain']) / 10000.0 if float(r['capital_gain']) > 0 else 0.0,
            float(r['capital_loss']) / 10000.0 if float(r['capital_loss']) > 0 else 0.0
        ]

        point['features'] = features
        data.append(point)

    return data


def submission(data):
    return train(data, epochs=20, learning_rate=0.001, reg_lambda=0.001)