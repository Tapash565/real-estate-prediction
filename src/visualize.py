"""Visualization functions for model performance and data analysis."""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def plot_predictions_vs_actual(y_true, y_pred, save_path=None, title="Predictions vs Actual"):
    """
    Create a scatter plot comparing predicted values to actual values.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    title : str, optional
        Title for the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
    
    # Add diagonal reference line (perfect predictions)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Labels and formatting
    ax.set_xlabel('Actual Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig


def plot_residuals(y_true, y_pred, save_path=None, title="Residual Plot"):
    """
    Create a residual plot to check model assumptions.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    title : str, optional
        Title for the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Create scatter plot
    ax.scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
    
    # Add zero reference line
    ax.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Residual')
    
    # Labels and formatting
    ax.set_xlabel('Predicted Values', fontsize=12)
    ax.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig


def plot_feature_importance(model, feature_names, save_path=None, title="Feature Importance", top_n=20):
    """
    Create a bar chart showing feature importance or coefficients.

    Parameters
    ----------
    model : estimator object
        Trained model with either feature_importances_ or coef_ attribute.
    feature_names : list
        List of feature names.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    title : str, optional
        Title for the plot.
    top_n : int, optional
        Number of top features to display. Default is 20.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract feature importance/coefficients
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        importance_type = "Importance"
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
        importance_type = "Absolute Coefficient"
    else:
        print("Model does not have feature_importances_ or coef_ attribute.")
        plt.close(fig)
        return None
    
    # Create DataFrame for sorting
    indices = np.argsort(importances)[::-1][:top_n]
    
    # Plot
    y_pos = np.arange(len(indices))
    ax.barh(y_pos, importances[indices], align='center', alpha=0.8, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] if i < len(feature_names) else f"Feature {i}" 
                        for i in indices])
    ax.invert_yaxis()  # Highest importance at top
    ax.set_xlabel(importance_type, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig


def plot_cv_scores(scores, save_path=None, title="Cross-Validation Scores"):
    """
    Visualize cross-validation scores.

    Parameters
    ----------
    scores : array-like
        Cross-validation scores for each fold.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    title : str, optional
        Title for the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    folds = np.arange(1, len(scores) + 1)
    mean_score = np.mean(scores)
    
    # Bar plot with mean line
    ax.bar(folds, scores, alpha=0.7, edgecolor='black', label='Fold Score')
    ax.axhline(y=mean_score, color='r', linestyle='--', lw=2, 
               label=f'Mean Score: {mean_score:.4f}')
    
    # Labels and formatting
    ax.set_xlabel('Fold Number', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(folds)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig


def plot_error_distribution(y_true, y_pred, save_path=None, title="Prediction Error Distribution"):
    """
    Create a histogram of prediction errors.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    title : str, optional
        Title for the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate errors
    errors = y_true - y_pred
    
    # Create histogram
    ax.hist(errors, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    
    # Add vertical line at zero
    ax.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
    
    # Add mean error line
    mean_error = np.mean(errors)
    ax.axvline(x=mean_error, color='green', linestyle='--', lw=2, 
               label=f'Mean Error: {mean_error:.2f}')
    
    # Labels and formatting
    ax.set_xlabel('Prediction Error (Actual - Predicted)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig


def plot_learning_curve(model, X, y, cv=5, save_path=None, title="Learning Curve"):
    """
    Plot learning curves to diagnose bias/variance.

    Parameters
    ----------
    model : estimator object
        Scikit-learn model to evaluate.
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    cv : int, optional
        Number of cross-validation folds.
    save_path : str, optional
        Path to save the figure.
    title : str, optional
        Title for the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    """
    from sklearn.model_selection import learning_curve
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get learning curve data
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='r2'
    )
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot
    ax.plot(train_sizes, train_mean, label='Training Score', color='blue', marker='o')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.15, color='blue')
    
    ax.plot(train_sizes, val_mean, label='Validation Score', color='red', marker='s')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.15, color='red')
    
    # Labels and formatting
    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig
