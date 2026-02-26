# TODO: Implement model evaluation functions
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

def evaluate_model(y_true, y_pred):
    """
    Evaluate the performance of a regression model.

    Parameters:
    y_true (array-like): The true target values.
    y_pred (array-like): The predicted target values by the model.

    Returns:
    dict: A dictionary containing the evaluation metrics.
    """
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return {
        'Mean Squared Error': mse,
        'Root Mean Squared Error': np.sqrt(mse),
        'R^2 Score': r2,
        'Mean Absolute Error': mae
    }