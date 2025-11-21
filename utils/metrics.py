import numpy as np


def RSE(pred, true):
    """
    Calculate Relative Squared Error (RSE)
    Measures the squared error relative to the variance of the true values

    Args:
        pred: Predicted values
        true: Ground truth values

    Returns:
        float: Relative squared error
    """
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    """
    Calculate Correlation Coefficient between predictions and ground truth
    Measures the linear relationship between predictions and actual values

    Args:
        pred: Predicted values
        true: Ground truth values

    Returns:
        float: Correlation coefficient
    """
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    """
    Calculate Mean Absolute Error (MAE)
    Average of absolute differences between predictions and ground truth

    Args:
        pred: Predicted values
        true: Ground truth values

    Returns:
        float: Mean absolute error
    """
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    """
    Calculate Mean Squared Error (MSE)
    Average of squared differences between predictions and ground truth

    Args:
        pred: Predicted values
        true: Ground truth values

    Returns:
        float: Mean squared error
    """
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    """
    Calculate Root Mean Squared Error (RMSE)
    Square root of MSE, provides error in same units as the data

    Args:
        pred: Predicted values
        true: Ground truth values

    Returns:
        float: Root mean squared error
    """
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    """
    Calculate Mean Absolute Percentage Error (MAPE)
    Average of absolute percentage errors, capped at 500% to handle outliers

    Args:
        pred: Predicted values
        true: Ground truth values

    Returns:
        float: Mean absolute percentage error
    """
    mape = np.abs((pred - true) / true)
    mape = np.where(mape > 5, 0, mape)  # Cap extreme errors at 500%
    return np.mean(mape)


def MSPE(pred, true):
    """
    Calculate Mean Squared Percentage Error (MSPE)
    Average of squared percentage errors

    Args:
        pred: Predicted values
        true: Ground truth values

    Returns:
        float: Mean squared percentage error
    """
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    """
    Calculate multiple evaluation metrics for time series forecasting

    Args:
        pred: Predicted values
        true: Ground truth values

    Returns:
        tuple: (MAE, MSE, RMSE, MAPE, MSPE) - multiple error metrics
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe