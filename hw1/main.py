"""
1. Complete the implementation for the `...` part
2. Feel free to take strategies to make faster convergence
3. You can add additional params to the Class/Function as you need. But the key print out should be kept.
4. Traps in the code. Fix common semantic/stylistic problems to pass the linting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger


class LinearRegressionBase:
    def __init__(self):
        self.weights = None
        self.intercept = None

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class LinearRegressionCloseform(LinearRegressionBase):
    def fit(self, X, y):
        """Question1
        Complete this function
        """
        n = X.shape[0]
        X_b = np.c_[np.ones((n, 1)), X]
        theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        self.intercept = theta[0]
        self.weights = theta[1:]

    def predict(self, X):
        """Question4
        Complete this function
        """
        return np.dot(X, self.weights) + self.intercept


class LinearRegressionGradientDescent(LinearRegressionBase):
    def __init__(self):
        self.weights = None
        self.intercept = None

    def fit(self, X, y, epochs: int, lr: float):
        y = y.ravel()
        n, d = X.shape
        self.weights = np.zeros(d)
        self.intercept = 0.0
        losses = []
        for epoch in range(epochs):
            y_pred = np.dot(X, self.weights) + self.intercept
            error = y_pred - y
            grad_w = (2 / n) * np.dot(X.T, error)
            grad_b = (2 / n) * np.sum(error)
            self.weights -= lr * grad_w
            self.intercept -= lr * grad_b
            loss = np.mean(error ** 2)
            losses.append(loss)
        return losses, [lr] * len(losses)

    def predict(self, X):
        """Question4
        Complete this
        """
        return np.dot(X, self.weights) + self.intercept


def compute_mse(prediction, ground_truth):
    mse = np.mean((prediction - ground_truth) ** 2)
    return mse


def main():
    train_df = pd.read_csv("./train.csv")  # Load training data
    test_df = pd.read_csv("./test.csv")  # Load test data
    train_x = train_df.drop(["Performance Index"], axis=1).to_numpy()
    train_y = train_df["Performance Index"].to_numpy()
    test_x = test_df.drop(["Performance Index"], axis=1).to_numpy()
    test_y = test_df["Performance Index"].to_numpy()

    lr_cf = LinearRegressionCloseform()
    lr_cf.fit(train_x, train_y)

    """This is the print out of question1"""
    logger.info(f"{lr_cf.weights=}")
    logger.info(f"{lr_cf.intercept=:.4f}")

    lr_gd = LinearRegressionGradientDescent()
    losses, lr_history = lr_gd.fit(train_x, train_y, epochs=1000000, lr=0.00018)

    """
    This is the print out of question2
    Note: You need to screenshot your hyper-parameters as well.
    """
    logger.info(f"GD original weights: {lr_gd.weights}")
    logger.info(f"GD original intercept: {lr_gd.intercept:.4f}")

    """
    Question3: Plot the learning curve.
    Implement here
    """
    if not losses:
        logger.error("No loss data available. Check if fit() executed correctly.")
    else:
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(losses)), losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("Learning Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

    """Question4"""
    y_preds_cf = lr_cf.predict(test_x)
    y_preds_gd = lr_gd.predict(test_x)
    y_preds_diff = np.abs(y_preds_gd - y_preds_cf).mean()
    logger.info(f"Prediction difference: {y_preds_diff:.8f}")

    mse_cf = compute_mse(y_preds_cf, test_y)
    mse_gd = compute_mse(y_preds_gd, test_y)
    diff = (np.abs(mse_gd - mse_cf) / mse_cf) * 100
    logger.info(f"{mse_cf=:.4f}, {mse_gd=:.4f}. Difference: {diff:.8f}%")


if __name__ == "__main__":
    main()
