import lithops
import random
from typing import List, Dict, Any
from sgd import initialize_model, logistic, dot, predict,compute_gradient


class DistributedSGD:
    def __init__(self, num_workers: int = 4):
        self.fexec = lithops.FunctionExecutor()
        self.num_workers = num_workers

    def _initialize_model(self, k: int) -> List[float]:
        return initialize_model(k)

    def _logistic(self, x: float) -> float:
        return logistic(x)

    def _dot(self, x: List[float], y: List[float]) -> float:
        return dot(x, y)

    def _predict(self, model: List[float], point: Dict[str, Any]) -> float:
        return predict(model, point)

    def _compute_gradient(self, point: Dict[str, Any], prediction: float) -> List[float]:
        return compute_gradient(point, prediction)

    def _worker_function(self, batch_data: List[Dict[str, Any]],
                         model: List[float],
                         learning_rate: float,
                         reg_lambda: float) -> List[float]:
        local_model = model.copy()

        for point in batch_data:
            prediction = self._predict(local_model, point)

            gradient = self._compute_gradient(point, prediction)

            for j in range(len(local_model)):
                reg_term = reg_lambda * local_model[j]
                local_model[j] -= learning_rate * (gradient[j] + reg_term)

        return local_model

    def train(self, data: List[Dict[str, Any]],
              epochs: int,
              learning_rate: float,
              reg_lambda: float) -> List[float]:
        """
        Train the model using distributed SGD

        Args:
            data: List of data points
            epochs: Number of training epochs
            learning_rate: Learning rate for SGD
            reg_lambda: L2 regularization parameter

        Returns:
            Trained model weights
        """
        # Initialize model
        model = self._initialize_model(len(data[0]['features']))

        batch_size = len(data) // self.num_workers

        for epoch in range(epochs):
            random.shuffle(data)

            batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

            iterdata = [(batch, model, learning_rate, reg_lambda)
                        for batch in batches[:self.num_workers]]

            futures = self.fexec.map(self._worker_function, iterdata)
            results = self.fexec.get_result(futures)

            # Average models from all workers
            model = [sum(weights) / len(weights)
                     for weights in zip(*results)]

            # Optional: Print progress
            if (epoch + 1) % 5 == 0:
                print(f"Completed epoch {epoch + 1}/{epochs}")

        self.fexec.clean()
        return model


def distributed_submission(data: List[Dict[str, Any]], num_workers: int = 4) -> List[float]:
    """Submission function that uses distributed SGD"""
    sgd = DistributedSGD(num_workers=num_workers)
    return sgd.train(data, epochs=20, learning_rate=0.001, reg_lambda=0.001)