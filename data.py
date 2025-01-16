import csv
import random


def load_csv(filename):
    lines = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

def load_adult_data():
    return load_csv("adult.csv")

def split_train_test(data, train_ratio=2 / 3, random_seed=42):
    """
    Split data into training and test sets.

    Following the original dataset specifications:
    - Uses approximately 2/3 for training, 1/3 for testing on the randomly shuffled data
    """
    random.seed(random_seed)

    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    split_idx = int(len(shuffled_data) * train_ratio)

    train_data = shuffled_data[:split_idx]
    test_data = shuffled_data[split_idx:]

    return train_data, test_data


def load_adult_train_data():
    full_data = load_adult_data()
    train_data, _ = split_train_test(full_data)
    return train_data


def load_adult_valid_data():
    full_data = load_adult_data()
    _, test_data = split_train_test(full_data)
    return test_data