import lithops
import random
from typing import List, Dict, Any
import time
from sgd import predict, initialize_model, compute_gradient

def process_batch(data: Dict) -> List[float]:

    try:
        model = data['model'].copy()
        learning_rate = data['learning_rate']
        reg_lambda = data['reg_lambda']
        batch = data['batch']

        print(f"Worker processing batch size: {len(batch)}")

        for i, point in enumerate(batch):

            prediction = predict(model, point)
            gradient = compute_gradient(point, prediction)

            for j in range(len(model)):
                reg_term = reg_lambda * model[j]
                model[j] -= learning_rate * (gradient[j] + reg_term)

        return model

    except Exception as e:
        print(f"Error in worker: {str(e)}")
        raise

class DistributedSGD:
    def __init__(self, num_workers: int = 1):
        config = {
            'lithops': {
                'runtime_memory': 3096,
                'worker_processes': num_workers
            },
            'localhost': {
                'runtime_memory': 3096
            }
        }
        self.fexec = lithops.FunctionExecutor(backend='localhost', config=config)
        self.num_workers = num_workers

    def partition_data(self, data: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        random.shuffle(data)
        batch_size = len(data) // self.num_workers
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def train(self, data: List[Dict[str, Any]],
              epochs: int = 5,
              learning_rate: float = 0.001,
              reg_lambda: float = 0.001) -> List[float]:
        """
        Train the model using distributed SGD.
        Implements the main loop from the algorithm in the PDF.
        """
        print(f"\nInitializing training with {self.num_workers} workers")
        model = initialize_model(len(data[0]['features']))
        print(f"Total data points: {len(data)}")

        start_time = time.time()
        best_accuracy = 0
        best_model = None

        for epoch in range(epochs):
            print(f"\nStarting epoch {epoch + 1}/{epochs}")
            epoch_start_time = time.time()

            partitioned_data = self.partition_data(data)

            worker_data = [{
                'data': {
                    'batch': batch,
                    'model': model,
                    'learning_rate': learning_rate,
                    'reg_lambda': reg_lambda
                }
            } for batch in partitioned_data]

            try:
                futures = self.fexec.map(process_batch, worker_data)
                worker_models = self.fexec.get_result(futures)

                model = [sum(w[j] for w in worker_models) / len(worker_models)
                        for j in range(len(model))]

                predictions = [predict(model, p) for p in data]
                current_accuracy = sum(1 for p, pred in zip(data, predictions)
                                    if (pred >= 0.5) == p['label']) / len(data)

                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    best_model = model.copy()

                epoch_time = time.time() - epoch_start_time
                print(f"Epoch {epoch + 1} completed - Accuracy: {current_accuracy:.4f}")
                print(f"Epoch time: {epoch_time:.2f}s")
                print(f"Best accuracy so far: {best_accuracy:.4f}")

            except Exception as e:
                print(f"Error during training: {str(e)}")
                raise

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f}s")
        print(f"Final best accuracy: {best_accuracy:.4f}")

        self.fexec.clean()
        return best_model if best_model is not None else model