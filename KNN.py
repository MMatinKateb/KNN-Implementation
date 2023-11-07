import numpy as np
from collections import Counter


"""
    Author: Mohammad Matin Kateb
    Email: matin.kateb.mk@gmail.com
    GitHub: https://www.github.com/MMatinKateb
"""


class KNN:
    """
    K-Nearest Neighbors (KNN) classification algorithm.

    Parameters:
    k (int): The number of nearest neighbors to use for classification.
    """
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        """
        Trains the KNN model on the given training data.

        Parameters:
        X (numpy.ndarray): An array of shape (n_samples, n_features) containing the training data.
        y (numpy.ndarray): An array of shape (n_samples,) containing the training labels.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Makes predictions on the given test data using the trained KNN model.

        Parameters:
        X (numpy.ndarray): An array of shape (n_samples, n_features) containing the test data.

        Returns:
        numpy.ndarray: An array of shape (n_samples,) containing the predicted class labels for the test data.
        """
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        """
        Helper method for making a single prediction.

        Parameters:
        x (numpy.ndarray): A single data point of shape (n_features,) containing the features of the data point.

        Returns:
        int: The predicted class label for the data point.
        """
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def _euclidean_distance(self, x1, x2):
        """
        Calculates the Euclidean distance between two points in n-dimensional space.

        Parameters:
        x1 (numpy.ndarray): A point of shape (n_features,) containing the coordinates of the first point.
        x2 (numpy.ndarray): A point of shape (n_features,) containing the coordinates of the second point.

        Returns:
        float: The Euclidean distance between the two points.
        """
        return np.sqrt(np.sum((x1 - x2)**2))