import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import umap
from tqdm import tqdm
import random
import copy

def evaluate_model(model, data, mask_percentage=0.15, device='cpu'):
    """
    Evaluate the model by masking a percentage of nodes and predicting their values.

    Parameters:
    - model (nn.Module): The trained PyTorch model.
    - data (torch_geometric.data.Data): The input data containing node features.
    - mask_percentage (float): The fraction of nodes to mask.
    - device (str): The device to perform computations on.

    Returns:
    - predictions (np.ndarray): The predicted values for masked nodes.
    - targets (np.ndarray): The actual values of masked nodes.
    """
    model.eval()
    with torch.no_grad():
        # Deep copy the data to avoid in-place modifications
        data = copy.deepcopy(data).to(device)
        
        # Apply masking
        num_nodes = data.x.size(0)
        mask = torch.rand(num_nodes, device=device) < mask_percentage
        target = data.x[mask].clone()

        modified_data_x = data.x.clone()
        masked_indices = torch.where(mask)[0]

        for idx in masked_indices:
            rand = random.random()
            if rand < 0.8:
                modified_data_x[idx] = 0  # [MASK]
            elif rand < 0.9:
                modified_data_x[idx] = torch.randn_like(data.x[idx])
            # else: leave it unchanged

        # Update data with masked features
        data.x = modified_data_x

        # Forward pass
        predictions = model(data, mask)

        # Move to CPU and convert to NumPy
        predictions = predictions.cpu().numpy()
        targets = target.cpu().numpy()

    return predictions, targets

def collect_predictions(model, data, num_evaluations=10, mask_percentage=0.15, device='cpu'):
    """
    Collect predictions and targets by evaluating the model multiple times.

    Parameters:
    - model (nn.Module): The trained PyTorch model.
    - data (torch_geometric.data.Data): The input data containing node features.
    - num_evaluations (int): Number of evaluation iterations.
    - mask_percentage (float): The fraction of nodes to mask in each evaluation.
    - device (str): The device to perform computations on.

    Returns:
    - all_predictions (np.ndarray): Concatenated predictions from all evaluations.
    - all_targets (np.ndarray): Concatenated targets from all evaluations.
    """
    model.eval()
    all_predictions = []
    all_targets = []

    for _ in tqdm(range(num_evaluations), desc="Evaluating"):
        preds, targs = evaluate_model(model, data, mask_percentage=mask_percentage, device=device)
        all_predictions.append(preds)
        all_targets.append(targs)

    # Concatenate all collected predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)  # Shape: [num_samples, feature_dim]
    all_targets = np.concatenate(all_targets, axis=0)          # Shape: [num_samples, feature_dim]

    return all_predictions, all_targets

def visualize_dimensionality_reduction(predictions, targets, pca_components=50, umap_components=2, random_state=42):
    """
    Perform PCA and UMAP dimensionality reduction and plot the UMAP projections separately.

    Parameters:
    - predictions (np.ndarray): Predicted values.
    - targets (np.ndarray): Actual target values.
    - pca_components (int): Number of PCA components.
    - umap_components (int): Number of UMAP components.
    - random_state (int): Random state for reproducibility.
    """
    # PCA for initial dimensionality reduction
    pca = PCA(n_components=pca_components, random_state=random_state)
    pred_pca = pca.fit_transform(predictions)
    target_pca = pca.fit_transform(targets)

    # UMAP for 2D visualization
    umap_reducer = umap.UMAP(n_components=umap_components, n_neighbors=15, min_dist=0.1,
                            metric='euclidean', random_state=random_state)
    pred_umap = umap_reducer.fit_transform(pred_pca)
    target_umap = umap_reducer.fit_transform(target_pca)

    # Side by Side Scatter Plots: UMAP Projection
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].scatter(target_umap[:, 0], target_umap[:, 1], alpha=0.5, label='Actual', s=10, color='blue')
    axes[0].set_xlabel('UMAP Component 1')
    axes[0].set_ylabel('UMAP Component 2')
    axes[0].set_title('UMAP Projection of Actual Values')
    axes[0].legend()

    axes[1].scatter(pred_umap[:, 0], pred_umap[:, 1], alpha=0.5, label='Predicted', s=10, color='orange')
    axes[1].set_xlabel('UMAP Component 1')
    axes[1].set_ylabel('UMAP Component 2')
    axes[1].set_title('UMAP Projection of Predicted Values')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def visualize_aggregate_metrics(predictions, targets):
    """
    Compute and plot histograms for MAE and RMSE per sample, and print overall metrics.

    Parameters:
    - predictions (np.ndarray): Predicted values.
    - targets (np.ndarray): Actual target values.
    """
    # Compute MAE and RMSE
    mae_per_sample = np.mean(np.abs(predictions - targets), axis=1)
    rmse_per_sample = np.sqrt(np.mean((predictions - targets) ** 2, axis=1))

    # Plot Histograms
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(mae_per_sample, bins=50, color='skyblue', edgecolor='black')
    plt.xlabel('Mean Absolute Error (MAE)')
    plt.ylabel('Frequency')
    plt.title('Histogram of MAE per Sample')

    plt.subplot(1, 2, 2)
    plt.hist(rmse_per_sample, bins=50, color='salmon', edgecolor='black')
    plt.xlabel('Root Mean Squared Error (RMSE)')
    plt.ylabel('Frequency')
    plt.title('Histogram of RMSE per Sample')

    plt.tight_layout()
    plt.show()

    # Overall Metrics
    overall_mae = mean_absolute_error(targets, predictions)
    overall_rmse = np.sqrt(mean_squared_error(targets, predictions))

    print(f"Overall MAE: {overall_mae:.4f}")
    print(f"Overall RMSE: {overall_rmse:.4f}")

def compute_feature_pcc(predictions, targets):
    """
    Compute Pearson Correlation Coefficient (PCC) for each feature.

    Parameters:
    - predictions (np.ndarray): Predicted values.
    - targets (np.ndarray): Actual target values.

    Returns:
    - pcc_values (np.ndarray): Array of PCC values for each feature.
    """
    num_features = predictions.shape[1]
    pcc_values = np.zeros(num_features)
    for i in range(num_features):
        if np.std(targets[:, i]) == 0 or np.std(predictions[:, i]) == 0:
            pcc_values[i] = 0  # Avoid division by zero
        else:
            pcc_values[i], _ = pearsonr(predictions[:, i], targets[:, i])
    return pcc_values

def visualize_pcc_histogram(predictions, targets, bins=50):
    """
    Compute and visualize the Pearson Correlation Coefficient (PCC) for each feature.

    Parameters:
    - predictions (np.ndarray): Predicted values.
    - targets (np.ndarray): Actual target values.
    - bins (int): Number of bins for the histogram.
    """
    # Compute PCC for each feature
    pcc_values = compute_feature_pcc(predictions, targets)

    # Sort PCC values from high to low
    sorted_pcc = np.sort(pcc_values)[::-1]

    # Plot Histogram of PCC Values
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_pcc)), sorted_pcc, color='green')
    plt.xlabel('Feature Index (sorted)')
    plt.ylabel('Pearson Correlation Coefficient (PCC)')
    plt.title('Histogram of PCC Values for Each Feature (High to Low)')
    plt.tight_layout()
    plt.show()

def visualize_feature_error_analysis(predictions, targets, top_n=20):
    """
    Analyze and visualize feature-wise MAE, highlighting top features with highest errors.

    Parameters:
    - predictions (np.ndarray): Predicted values.
    - targets (np.ndarray): Actual target values.
    - top_n (int): Number of top features to display based on MAE.
    """
    # Compute MAE for each feature
    feature_mae = np.mean(np.abs(predictions - targets), axis=0)

    # Identify top N features with highest MAE
    top_features = np.argsort(feature_mae)[-top_n:][::-1]  # Sorted descending

    # Plot MAE for Top N Features
    plt.figure(figsize=(12, 6))
    plt.bar(range(top_n), feature_mae[top_features], color='teal')
    plt.xlabel('Feature Index')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title(f'Top {top_n} Features by MAE')
    plt.xticks(range(top_n), top_features, rotation=45)
    plt.tight_layout()
    plt.show()

    # Scatter Plots for Top M Features
    visualize_top_feature_scatter(predictions, targets, top_features, num_top=5)

def visualize_top_feature_scatter(predictions, targets, top_features, num_top=5):
    """
    Create scatter plots for the top N features with the highest MAE.

    Parameters:
    - predictions (np.ndarray): Predicted values.
    - targets (np.ndarray): Actual target values.
    - top_features (np.ndarray): Array of feature indices sorted by MAE.
    - num_top (int): Number of top features to plot.
    """
    import seaborn as sns  # Importing here to avoid redundancy if already imported

    top_n = top_features[:num_top]

    plt.figure(figsize=(15, 12))
    for i, feature_idx in enumerate(top_n, 1):
        plt.subplot(3, 2, i)
        sns.scatterplot(x=targets[:, feature_idx], y=predictions[:, feature_idx], alpha=0.3)
        min_val = min(targets[:, feature_idx].min(), predictions[:, feature_idx].min())
        max_val = max(targets[:, feature_idx].max(), predictions[:, feature_idx].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')  # y=x line
        plt.xlabel(f'Actual Feature {feature_idx}')
        plt.ylabel(f'Predicted Feature {feature_idx}')
        plt.title(f'Feature {feature_idx}: Predicted vs. Actual')
        plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_cosine_similarity(predictions, targets):
    """
    Compute and visualize cosine similarity between predictions and targets.

    Parameters:
    - predictions (np.ndarray): Predicted values.
    - targets (np.ndarray): Actual target values.
    """
    # Normalize predictions and targets
    pred_norm = predictions / (np.linalg.norm(predictions, axis=1, keepdims=True) + 1e-8)
    target_norm = targets / (np.linalg.norm(targets, axis=1, keepdims=True) + 1e-8)

    # Compute cosine similarity
    cos_sim = np.sum(pred_norm * target_norm, axis=1)

    # Plot Histogram of Cosine Similarity
    plt.figure(figsize=(8, 6))
    plt.hist(cos_sim, bins=50, color='purple', edgecolor='black')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Histogram of Cosine Similarity between Predictions and Targets')
    plt.show()

    # Compute Overall Average Cosine Similarity
    average_cos_sim = np.mean(cos_sim)
    print(f"Average Cosine Similarity: {average_cos_sim:.4f}")

def feature_error_analysis(predictions, targets, top_n=20, num_top_scatter=5):
    """
    Perform feature-wise error analysis, including MAE computation and scatter plots.

    Parameters:
    - predictions (np.ndarray): Predicted values.
    - targets (np.ndarray): Actual target values.
    - top_n (int): Number of top features to analyze based on MAE.
    - num_top_scatter (int): Number of top features to visualize with scatter plots.
    """
    # Compute MAE for each feature
    feature_mae = np.mean(np.abs(predictions - targets), axis=0)

    # Identify top N features with highest MAE
    top_features = np.argsort(feature_mae)[-top_n:][::-1]  # Sorted descending

    # Plot MAE for Top N Features
    plt.figure(figsize=(12, 6))
    plt.bar(range(top_n), feature_mae[top_features], color='teal')
    plt.xlabel('Feature Index')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title(f'Top {top_n} Features by MAE')
    plt.xticks(range(top_n), top_features, rotation=45)
    plt.tight_layout()
    plt.show()

    # Scatter Plots for Top M Features
    visualize_top_feature_scatter(predictions, targets, top_features, num_top=num_top_scatter)

def visualize_pcc_histogram(predictions, targets, bins=50):
    """
    Compute and visualize the Pearson Correlation Coefficient (PCC) for each feature.

    Parameters:
    - predictions (np.ndarray): Predicted values.
    - targets (np.ndarray): Actual target values.
    - bins (int): Number of bins for the histogram.
    """
    # Compute PCC for each feature
    pcc_values = compute_feature_pcc(predictions, targets)

    # Sort PCC values from high to low
    sorted_pcc = np.sort(pcc_values)[::-1]

    # Plot Histogram of PCC Values
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(sorted_pcc)), sorted_pcc, color='green')
    plt.xlabel('Feature Index (sorted)')
    plt.ylabel('Pearson Correlation Coefficient (PCC)')
    plt.title('Histogram of PCC Values for Each Feature (High to Low)')
    plt.tight_layout()
    plt.show()

def visualize_all(predictions, targets):
    """
    Perform all visualization tasks.

    Parameters:
    - predictions (np.ndarray): Predicted values.
    - targets (np.ndarray): Actual target values.
    """
    visualize_dimensionality_reduction(predictions, targets)
    visualize_aggregate_metrics(predictions, targets)
    feature_error_analysis(predictions, targets, top_n=20, num_top_scatter=5)
    visualize_cosine_similarity(predictions, targets)
    visualize_pcc_histogram(predictions, targets)
