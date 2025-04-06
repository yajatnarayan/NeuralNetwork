import numpy as np
import json
import os
from ocr import OCRNeuralNetwork

def load_data():
    """
    Load training data from a JSON file.
    In a real implementation, this would load actual training data.
    For this example, we'll create some dummy data.
    """
    # Create dummy data for demonstration
    data_matrix = []
    data_labels = []
    
    # Create 100 samples of each digit (0-9)
    for digit in range(10):
        for _ in range(100):
            # Create a random 20x20 pixel representation (400 values)
            sample = np.random.randint(0, 2, 400).tolist()
            data_matrix.append(sample)
            data_labels.append(digit)
    
    # Convert to numpy arrays
    data_matrix = np.array(data_matrix)
    data_labels = np.array(data_labels)
    
    # Split into training and test sets (75% training, 25% testing)
    indices = np.random.permutation(len(data_matrix))
    split = int(0.75 * len(data_matrix))
    train_indices = indices[:split]
    test_indices = indices[split:]
    
    return data_matrix, data_labels, train_indices, test_indices

def test(data_matrix, data_labels, test_indices, nn):
    """
    Test the neural network on the test data and return the accuracy.
    """
    avg_sum = 0
    for j in range(100):
        correct_guess_count = 0
        for i in test_indices:
            test = data_matrix[i]
            prediction = nn.predict(test)
            if data_labels[i] == prediction:
                correct_guess_count += 1

        avg_sum += (correct_guess_count / float(len(test_indices)))
    return avg_sum / 100

def main():
    print("Loading data...")
    data_matrix, data_labels, train_indices, test_indices = load_data()
    
    print("\nPERFORMANCE")
    print("-----------")
    
    # Try various number of hidden nodes and see what performs best
    for i in range(5, 50, 5):
        print(f"Training network with {i} hidden nodes...")
        nn = OCRNeuralNetwork(i, data_matrix, data_labels, train_indices, False)
        
        # Train the network
        for _ in range(10):  # Train for 10 epochs
            for idx in train_indices:
                nn.train([{"y0": data_matrix[idx], "label": data_labels[idx]}])
        
        # Test the network
        performance = test(data_matrix, data_labels, test_indices, nn)
        print(f"{i} Hidden Nodes: {performance:.4f}")

if __name__ == "__main__":
    main() 