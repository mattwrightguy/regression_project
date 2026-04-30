import numpy as np
from typing import Optional
from abc import ABC, abstractmethod

class BaseRegressor(ABC):
    """
    Base regressor model with regularization.

    Attributes:
        w: Shape(n,). A 1D array of model weights, corresponding to each feature.
        b: The model bias.
        J_history: A list of costs for each iteration.
    """
    w: Optional[np.ndarray]
    b: float
    J_history: list
    
    def __init__(self, alpha: float = 0.01, n_iterations: int = 1000, lambda_: float = 0) -> None:
        """
        Initialize the model with hyperparameters.

        Args:
            alpha: The learning rate.
            n_iterations: The number of iterations.
            lambda_: The regularization parameter.
        """
        self.alpha = alpha
        self.n_iterations= n_iterations
        self.lambda_ = lambda_
        self.w = None
        self.b = 0.0
        self.J_history = []

    @property
    def is_fitted(self):
        """Returns True if the model weights have been trained."""
        return self.w is not None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Complete batch gradient descent learning algorithm and update parameters w and b.
    
        Args:
            X: Shape(m,n). A 2D array of training features, where 'm' is the number of training examples and 'n' is the number of features.
            y: Shape(m,). A 1D array of target values.
        """
        n = X.shape[1]
        self.w = np.zeros(n)
        for i in range(self.n_iterations):
            dj_dw, dj_db = self._compute_gradient(X, y)
            self.w = self.w - self.alpha * dj_dw
            self.b = self.b - self.alpha * dj_db
            cost = self._compute_cost(X, y)
            if i < 100000:
                self.J_history.append(cost)
            if i % max(1, (self.n_iterations // 10)) == 0:
                print(f"Iteration {i:4d}: Cost {cost:8.2f}")

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate predictions from the regressor model.

        Args:
            X: Shape(m,n). A 2D array of training features, where 'm' is the number of training examples and 'n' is the number of features.

        Returns:
            pred: Shape(m,). A 1D array of predicted values from the model.
        """
        pass
    
    @abstractmethod
    def _compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the cost for the specific regressor. 
    
        Args:
            X: Shape(m,n). A 2D array of training features, where 'm' is the number of training examples, and n is the number of features.
            y: Shape(m,). A 1D array of target values.
    
        Returns:
            total_cost: The cost J(w,b).
        """
        pass
        
    @abstractmethod
    def _compute_gradient(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Compute the gradient dj_dw, dj_db of the cost function J(w,b) for the specific regressor.
    
        Args:
            X: Shape(m,n). A 2D array of training features, where 'm' is the number of training examples and 'n' is the number of features.
            y: Shape(m,). A 1D array of target values.
    
        Returns:
            dj_dw: Shape(n,). A 1D array of gradients w.r.t. the parameters w.
            dj_db: The gradient w.r.t the parameter b.
        """
        pass

class LinearRegressor(BaseRegressor):
    """
    Linear regressor, using mean least squares cost and regularization.

    Attributes:
        w: Shape(n,). A 1D array of model weights, corresponding to each feature.
        b: The model bias.
        J_history: A list of costs for each iteration.
    """

    def __init__(self, alpha: float = 0.01, n_iterations: int = 1000, lambda_: float = 0) -> None:
        """
        Initialize the model with hyperparameters.

        Args:
            alpha: The learning rate.
            n_iterations: The number of iterations.
            lambda_: The regularization parameter.
        """
        super().__init__(alpha=alpha, n_iterations=n_iterations, lambda_=lambda_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("The model is not fitted yet. Call 'fit' with appropriate argument before using this estimator.")
        pred = X @ self.w + self.b
        return pred

    def _compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        m = X.shape[0]
        pred = X @ self.w + self.b
        err = (pred - y)
        cost = np.mean(err**2) / 2
        reg = self.lambda_ / (2 * m) * np.sum(self.w**2)
        total_cost = cost + reg
        return total_cost

    def _compute_gradient(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        m = X.shape[0]
        pred = X @ self.w + self.b
        err = pred - y
        dj_db = np.mean(err)
        dj_dw = (X.T @ err) / m + self.lambda_ / m * self.w
        return dj_dw, dj_db

class LogisticRegressor(BaseRegressor):
    """
    Logistic regressor for binary classification, using log loss cost and regularization.

    Attributes:
        w: Shape(n,). A 1D array of model weights, corresponding to each feature.
        b: The model bias.
        J_history: A list of costs for each iteration.
    """
    threshold: float

    def __init__(self, alpha: float = 0.01, n_iterations: int = 1000, lambda_: float = 0, threshold: float = 0.5) -> None:
        """
        Initialize the model with hyperparameters.

        Args:
            alpha: The learning rate.
            n_iterations: The number of iterations.
            lambda_: The regularization parameter.
            threshold: Probability equal to or above which the model will predict a classification of 1.
        """
        super().__init__(alpha=alpha, n_iterations=n_iterations, lambda_=lambda_)
        self.threshold = threshold

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("The model is not fitted yet. Call 'fit' with appropriate argument before using this estimator.")
        z = X @ self.w + self.b
        g = 1 / (1 + np.exp(-z))
        return (g >= self.threshold).astype(int)

    def _compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        m = X.shape[0]
        epsilon = 1e-15
        z = X @ self.w + self.b
        g = 1 / (1 + np.exp(-z))
        g = np.clip(g, epsilon, 1 - epsilon)
        cost = -np.mean(y * np.log(g) + (1 - y) * np.log(1 - g))
        reg = self.lambda_ / (2 * m) * np.sum(self.w**2)
        total_cost = cost + reg
        return total_cost

    def _compute_gradient(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        m = X.shape[0]
        z = X @ self.w + self.b
        g = 1 / (1 + np.exp(-z))
        err = g - y
        dj_db = np.mean(err)
        dj_dw = (X.T @ err) / m + self.lambda_ / m * self.w
        return dj_dw, dj_db