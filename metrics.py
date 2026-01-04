"""Metrics collection module for OCR Neural Network.

Tracks training, prediction, and system performance metrics.
Includes MNIST test accuracy evaluation and network architecture info.
"""

import json
import time
import os
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional


def download_mnist():
    """Download MNIST test dataset if not present."""
    import urllib.request
    import gzip

    mnist_dir = 'mnist_data'
    os.makedirs(mnist_dir, exist_ok=True)

    base_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    files = {
        't10k-images-idx3-ubyte.gz': 'test_images',
        't10k-labels-idx1-ubyte.gz': 'test_labels'
    }

    for filename, _ in files.items():
        filepath = os.path.join(mnist_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, filepath)
            print(f"Downloaded {filename}")

    return mnist_dir


def load_mnist_test(sample_size=1000):
    """Load MNIST test dataset (downsampled to 20x20)."""
    import gzip
    import struct

    mnist_dir = download_mnist()

    # Load test images
    with gzip.open(os.path.join(mnist_dir, 't10k-images-idx3-ubyte.gz'), 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)

    # Load test labels
    with gzip.open(os.path.join(mnist_dir, 't10k-labels-idx1-ubyte.gz'), 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    # Downsample from 28x28 to 20x20 and normalize
    from PIL import Image

    test_data = []
    indices = np.random.choice(len(images), min(sample_size, len(images)), replace=False)

    for idx in indices:
        img = Image.fromarray(images[idx])
        img = img.resize((20, 20), Image.Resampling.LANCZOS)
        pixels = np.array(img).flatten() / 255.0
        test_data.append({
            'pixels': pixels.tolist(),
            'label': int(labels[idx])
        })

    return test_data


class MetricsCollector:
    """Collects and stores metrics for the OCR neural network."""

    METRICS_FILE = 'metrics.json'

    def __init__(self):
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_start = time.time()

        # Training metrics
        self.total_samples_trained = 0
        self.training_history: List[Dict] = []
        self.digit_distribution = defaultdict(int)

        # Prediction metrics
        self.total_predictions = 0
        self.prediction_history: List[Dict] = []
        self.correct_predictions = 0
        self.confusion_matrix = np.zeros((10, 10), dtype=int)

        # Weight metrics
        self.weight_snapshots: List[Dict] = []

        # Performance metrics
        self.request_latencies: List[Dict] = []

        # Load existing metrics if available
        self._load_metrics()

    def _load_metrics(self):
        """Load existing metrics from file."""
        if os.path.exists(self.METRICS_FILE):
            try:
                with open(self.METRICS_FILE, 'r') as f:
                    data = json.load(f)
                    self.total_samples_trained = data.get('total_samples_trained', 0)
                    self.total_predictions = data.get('total_predictions', 0)
                    self.correct_predictions = data.get('correct_predictions', 0)
                    self.digit_distribution = defaultdict(int, data.get('digit_distribution', {}))
                    if 'confusion_matrix' in data:
                        self.confusion_matrix = np.array(data['confusion_matrix'])
            except (json.JSONDecodeError, KeyError):
                pass

    def save_metrics(self):
        """Save current metrics to file."""
        data = {
            'session_id': self.session_id,
            'last_updated': datetime.now().isoformat(),
            'total_samples_trained': self.total_samples_trained,
            'total_predictions': self.total_predictions,
            'correct_predictions': self.correct_predictions,
            'accuracy': self.get_accuracy(),
            'digit_distribution': dict(self.digit_distribution),
            'confusion_matrix': self.confusion_matrix.tolist(),
            'recent_training': self.training_history[-100:],
            'recent_predictions': self.prediction_history[-100:],
            'weight_snapshots': self.weight_snapshots[-10:],
            'performance': {
                'avg_latency_ms': self.get_avg_latency(),
                'total_requests': len(self.request_latencies)
            }
        }
        with open(self.METRICS_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    def record_training(self, samples: List[Dict], theta1: np.ndarray, theta2: np.ndarray,
                       duration_ms: float, loss: Optional[float] = None):
        """Record a training event."""
        timestamp = datetime.now().isoformat()

        for sample in samples:
            label = sample['label']
            self.digit_distribution[str(label)] += 1
            self.total_samples_trained += 1

        # Calculate weight statistics
        weight_stats = self._calculate_weight_stats(theta1, theta2)

        record = {
            'timestamp': timestamp,
            'num_samples': len(samples),
            'labels': [s['label'] for s in samples],
            'duration_ms': round(duration_ms, 2),
            'weight_stats': weight_stats,
            'loss': loss
        }

        self.training_history.append(record)
        self.save_metrics()

        return record

    def record_prediction(self, prediction: int, confidence: float,
                         output_activations: List[float], duration_ms: float,
                         correct_label: Optional[int] = None):
        """Record a prediction event."""
        timestamp = datetime.now().isoformat()
        self.total_predictions += 1

        # Calculate confidence metrics
        sorted_activations = sorted(output_activations, reverse=True)
        confidence_gap = sorted_activations[0] - sorted_activations[1] if len(sorted_activations) > 1 else 0
        entropy = self._calculate_entropy(output_activations)

        is_correct = None
        if correct_label is not None:
            is_correct = prediction == correct_label
            if is_correct:
                self.correct_predictions += 1
            self.confusion_matrix[correct_label][prediction] += 1

        record = {
            'timestamp': timestamp,
            'prediction': prediction,
            'confidence': round(confidence, 4),
            'confidence_gap': round(confidence_gap, 4),
            'entropy': round(entropy, 4),
            'duration_ms': round(duration_ms, 2),
            'is_correct': is_correct,
            'all_confidences': [round(c, 4) for c in output_activations]
        }

        self.prediction_history.append(record)
        self.save_metrics()

        return record

    def record_feedback(self, predicted: int, actual: int):
        """Record user feedback (correction)."""
        self.confusion_matrix[actual][predicted] += 1
        if predicted == actual:
            self.correct_predictions += 1
        self.save_metrics()

    def record_latency(self, endpoint: str, duration_ms: float):
        """Record request latency."""
        self.request_latencies.append({
            'timestamp': datetime.now().isoformat(),
            'endpoint': endpoint,
            'duration_ms': round(duration_ms, 2)
        })

    def snapshot_weights(self, theta1: np.ndarray, theta2: np.ndarray):
        """Take a snapshot of weight statistics."""
        stats = self._calculate_weight_stats(theta1, theta2)
        stats['timestamp'] = datetime.now().isoformat()
        stats['total_samples'] = self.total_samples_trained
        self.weight_snapshots.append(stats)
        self.save_metrics()

    def _calculate_weight_stats(self, theta1: np.ndarray, theta2: np.ndarray) -> Dict:
        """Calculate statistics for weight matrices."""
        return {
            'theta1': {
                'mean': float(np.mean(theta1)),
                'std': float(np.std(theta1)),
                'min': float(np.min(theta1)),
                'max': float(np.max(theta1)),
                'norm': float(np.linalg.norm(theta1)),
                'sparsity': float(np.sum(np.abs(theta1) < 0.01) / theta1.size)
            },
            'theta2': {
                'mean': float(np.mean(theta2)),
                'std': float(np.std(theta2)),
                'min': float(np.min(theta2)),
                'max': float(np.max(theta2)),
                'norm': float(np.linalg.norm(theta2)),
                'sparsity': float(np.sum(np.abs(theta2) < 0.01) / theta2.size)
            }
        }

    def _calculate_entropy(self, probs: List[float]) -> float:
        """Calculate entropy of probability distribution."""
        probs = np.array(probs)
        probs = probs / probs.sum()  # Normalize
        probs = np.clip(probs, 1e-10, 1)  # Avoid log(0)
        return float(-np.sum(probs * np.log2(probs)))

    def get_accuracy(self) -> float:
        """Get overall prediction accuracy."""
        if self.total_predictions == 0:
            return 0.0
        return round(self.correct_predictions / self.total_predictions, 4)

    def get_avg_latency(self) -> float:
        """Get average request latency."""
        if not self.request_latencies:
            return 0.0
        return round(np.mean([r['duration_ms'] for r in self.request_latencies]), 2)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        session_duration = time.time() - self.session_start

        # Calculate per-digit accuracy from confusion matrix
        per_digit_accuracy = {}
        for digit in range(10):
            total = self.confusion_matrix[digit].sum()
            correct = self.confusion_matrix[digit][digit]
            per_digit_accuracy[str(digit)] = round(correct / total, 4) if total > 0 else None

        # Recent performance (last 20 predictions)
        recent_preds = self.prediction_history[-20:]
        recent_correct = sum(1 for p in recent_preds if p.get('is_correct') == True)
        recent_accuracy = recent_correct / len(recent_preds) if recent_preds else None

        return {
            'session': {
                'id': self.session_id,
                'duration_seconds': round(session_duration, 1),
                'start_time': datetime.fromtimestamp(self.session_start).isoformat()
            },
            'training': {
                'total_samples': self.total_samples_trained,
                'digit_distribution': dict(self.digit_distribution),
                'session_samples': len(self.training_history)
            },
            'predictions': {
                'total': self.total_predictions,
                'correct': self.correct_predictions,
                'accuracy': self.get_accuracy(),
                'recent_accuracy': recent_accuracy,
                'per_digit_accuracy': per_digit_accuracy
            },
            'confidence': self._get_confidence_stats(),
            'performance': {
                'avg_latency_ms': self.get_avg_latency(),
                'total_requests': len(self.request_latencies)
            },
            'weights': self.weight_snapshots[-1] if self.weight_snapshots else None,
            'confusion_matrix': self.confusion_matrix.tolist()
        }

    def _get_confidence_stats(self) -> Dict:
        """Get statistics about prediction confidence."""
        if not self.prediction_history:
            return {}

        confidences = [p['confidence'] for p in self.prediction_history]
        entropies = [p['entropy'] for p in self.prediction_history]
        gaps = [p['confidence_gap'] for p in self.prediction_history]

        return {
            'mean_confidence': round(np.mean(confidences), 4),
            'std_confidence': round(np.std(confidences), 4),
            'mean_entropy': round(np.mean(entropies), 4),
            'mean_confidence_gap': round(np.mean(gaps), 4)
        }

    def get_training_chart_data(self, limit: int = 50) -> List[Dict]:
        """Get data formatted for training progress chart."""
        return self.training_history[-limit:]

    def get_prediction_chart_data(self, limit: int = 50) -> List[Dict]:
        """Get data formatted for prediction chart."""
        return self.prediction_history[-limit:]

    def reset_session_metrics(self):
        """Reset session-specific metrics (keeps historical totals)."""
        self.training_history = []
        self.prediction_history = []
        self.request_latencies = []
        self.session_start = time.time()
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    def evaluate_mnist(self, nn, sample_size: int = 1000) -> Dict:
        """Evaluate the neural network on MNIST test set.

        Args:
            nn: OCRNeuralNetwork instance
            sample_size: Number of test samples to use (default 1000)

        Returns:
            Dict with accuracy metrics and per-digit breakdown
        """
        print(f"Loading MNIST test data ({sample_size} samples)...")
        test_data = load_mnist_test(sample_size)

        correct = 0
        per_digit_correct = defaultdict(int)
        per_digit_total = defaultdict(int)
        confusion = np.zeros((10, 10), dtype=int)

        print("Evaluating model...")
        for item in test_data:
            prediction = nn.predict(item['pixels'])
            label = item['label']

            per_digit_total[label] += 1
            confusion[label][prediction] += 1

            if prediction == label:
                correct += 1
                per_digit_correct[label] += 1

        accuracy = correct / len(test_data)

        # Per-digit accuracy
        per_digit_accuracy = {}
        for digit in range(10):
            if per_digit_total[digit] > 0:
                per_digit_accuracy[str(digit)] = round(
                    per_digit_correct[digit] / per_digit_total[digit], 4
                )
            else:
                per_digit_accuracy[str(digit)] = None

        result = {
            'timestamp': datetime.now().isoformat(),
            'sample_size': len(test_data),
            'accuracy': round(accuracy, 4),
            'accuracy_percent': round(accuracy * 100, 2),
            'correct': correct,
            'per_digit_accuracy': per_digit_accuracy,
            'confusion_matrix': confusion.tolist()
        }

        # Store this evaluation
        self.mnist_evaluations = getattr(self, 'mnist_evaluations', [])
        self.mnist_evaluations.append(result)
        self.save_metrics()

        print(f"MNIST Test Accuracy: {result['accuracy_percent']}%")
        return result

    def get_network_architecture(self, nn) -> Dict:
        """Get network architecture and parameter count.

        Args:
            nn: OCRNeuralNetwork instance

        Returns:
            Dict with architecture details and parameter counts
        """
        # Layer dimensions
        input_size = nn.INPUT_SIZE  # 400
        hidden_size = nn.num_hidden_nodes
        output_size = nn.NUM_DIGITS  # 10

        # Parameter counts
        theta1_params = nn.theta1.shape[0] * nn.theta1.shape[1]  # hidden x input
        b1_params = nn.b1.shape[0]  # hidden
        theta2_params = nn.theta2.shape[0] * nn.theta2.shape[1]  # output x hidden
        b2_params = nn.b2.shape[0]  # output

        total_params = theta1_params + b1_params + theta2_params + b2_params

        return {
            'architecture': {
                'type': '3-layer feedforward neural network',
                'input_layer': {
                    'size': input_size,
                    'description': '20x20 pixel grayscale image'
                },
                'hidden_layer': {
                    'size': hidden_size,
                    'activation': 'sigmoid'
                },
                'output_layer': {
                    'size': output_size,
                    'activation': 'sigmoid',
                    'description': 'digit probabilities (0-9)'
                }
            },
            'parameters': {
                'theta1': {
                    'shape': list(nn.theta1.shape),
                    'count': theta1_params,
                    'description': 'input -> hidden weights'
                },
                'b1': {
                    'shape': list(nn.b1.shape),
                    'count': b1_params,
                    'description': 'hidden layer bias'
                },
                'theta2': {
                    'shape': list(nn.theta2.shape),
                    'count': theta2_params,
                    'description': 'hidden -> output weights'
                },
                'b2': {
                    'shape': list(nn.b2.shape),
                    'count': b2_params,
                    'description': 'output layer bias'
                },
                'total': total_params
            },
            'hyperparameters': {
                'learning_rate': nn.LEARNING_RATE
            }
        }


# Global metrics instance
metrics = MetricsCollector()
