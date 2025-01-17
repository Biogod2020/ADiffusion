
import scanpy as sc
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import skimage.io as io
import random
from torch_geometric.data import Data
from tqdm import tqdm
from scipy.sparse import csr_matrix, diags

def extract_patches(image, cell_coords, patch_size=100):
    """
    Extract image patches centered on given cell coordinates.

    Parameters:
    image (numpy.ndarray): The input image from which patches are to be extracted.
    cell_coords (list of tuples): A list of (x, y) coordinates around which patches are to be extracted.
    patch_size (int, optional): The size of the square patches to be extracted. Default is 100.

    Returns:
    list of numpy.ndarray: A list of extracted image patches, all of the same size.

    Notes:
    - If a patch extends beyond the image border, it will be padded with zeros to match the specified patch size.
    """
    patches = []
    half_size = patch_size // 2
    h, w = image.shape[:2]
    different_shape_count = 0

    # Ensure the input image has 3 dimensions
    if len(image.shape) == 2:  # Grayscale image
        image = image[:, :, np.newaxis]

    for (x, y) in cell_coords:
        x, y = int(x), int(y)
        x_min, x_max = x - half_size, x + half_size
        y_min, y_max = y - half_size, y + half_size

        # Initialize the patch with zeros
        patch = np.zeros((patch_size, patch_size, image.shape[2]), dtype=image.dtype)

        # Calculate bounds for extracting the region from the image
        x_start = max(0, x_min)
        x_end = min(w, x_max)
        y_start = max(0, y_min)
        y_end = min(h, y_max)

        # Calculate where to place the extracted region in the patch
        patch_x_start = max(0, -x_min)
        patch_x_end = patch_x_start + (x_end - x_start)
        patch_y_start = max(0, -y_min)
        patch_y_end = patch_y_start + (y_end - y_start)

        # Assign the extracted region into the patch
        patch[patch_y_start:patch_y_end, patch_x_start:patch_x_end] = image[y_start:y_end, x_start:x_end]

        if patch.shape[:2] != (patch_size, patch_size):
            different_shape_count += 1

        patches.append(patch)

    print(f"Number of patches with different shapes before padding: {different_shape_count}")
    return patches


def create_graph_data_dict(adatas, areas, neighbors, cell_coords, embeddings=['X']):
    """
    Create a dictionary of PyTorch Geometric Data objects from AnnData objects,
    allowing the inclusion of multiple embeddings.

    Parameters:
    - adatas: dict of AnnData objects
    - areas: dict of patch areas
    - neighbors: dict of connectivity matrices (affinity matrices with weights)
    - cell_coords: dict of spatial coordinates
    - embeddings: list of strings specifying which embeddings to include as node features.
                 For example: ['X', 'X_pca', 'X_umap', 'X_spatial']

    Returns:
    - graph_data_dict: dict of PyTorch Geometric Data objects
    """
    graph_data_dict = {}

    for key in tqdm(adatas.keys(), desc="Creating graph data"):
        # Ensure the keys match between adata and patches
        if key not in areas:
            print(f"Warning: No patch area data for '{key}'. Skipping.")
            continue
        if key not in neighbors:
            print(f"Warning: No neighbors data for '{key}'. Skipping.")
            continue
        if key not in cell_coords:
            print(f"Warning: No cell coordinates data for '{key}'. Skipping.")
            continue

        adata = adatas[key]
        num_cells = adata.n_obs

        # Initialize a list to hold all feature arrays
        feature_list = []

        for embed in embeddings:
            if embed == 'X':
                # Primary gene expression data
                features = adata.X
            elif embed in adata.obsm.keys():
                # Embeddings stored in obsm (e.g., PCA, UMAP)
                features = adata.obsm[embed]
            elif embed in adata.layers.keys():
                # Embeddings stored in layers
                features = adata.layers[embed]
            else:
                print(f"Warning: Embedding '{embed}' not found in 'X', 'obsm', or 'layers' for '{key}'. Skipping this embedding.")
                continue

            # Convert sparse matrices to dense if necessary
            if scipy.sparse.issparse(features):
                features = features.toarray()

            # Convert to torch tensor
            features = torch.tensor(features, dtype=torch.float)

            # Ensure the number of cells matches
            if features.shape[0] != num_cells:
                print(f"Warning: Embedding '{embed}' has {features.shape[0]} cells, which does not match the number of cells ({num_cells}) for '{key}'. Trimming or skipping.")
                min_len = min(features.shape[0], num_cells)
                features = features[:min_len]
                num_cells = min_len  # Update num_cells accordingly

            feature_list.append(features)

        if not feature_list:
            print(f"Warning: No valid embeddings found for '{key}'. Skipping.")
            continue

        # Concatenate all features along the feature dimension
        features = torch.cat(feature_list, dim=1)

        # Labels: Patch areas
        label_areas = areas[key]
        if len(label_areas) != num_cells:
            print(f"Warning: Number of patches and cells do not match for '{key}'.")
            # Handle mismatch by trimming to the minimum length
            min_len = min(len(label_areas), num_cells)
            label_areas = label_areas[:min_len]
            features = features[:min_len]
            num_cells = min_len  # Update num_cells accordingly

        labels = torch.tensor(label_areas, dtype=torch.float).unsqueeze(1)  # Shape: [num_nodes, 1]

        # Edges: Connectivity matrix (Affinity matrix with weights)
        connectivity = neighbors[key].tocoo()

        # Extract edge indices
        edge_index = torch.tensor(
            np.vstack([connectivity.row, connectivity.col]),
            dtype=torch.long
        )

        # Extract edge weights (inverse distances or affinity weights)
        edge_weights = torch.tensor(connectivity.data, dtype=torch.float).unsqueeze(1)  # Shape: [num_edges, 1]

        # Create PyTorch Geometric Data object
        data = Data(
            x=features,
            edge_index=edge_index,
            edge_attr=edge_weights,
            y=labels
        )

        # Optionally, add additional information (e.g., spatial coordinates)
        spatial = torch.tensor(cell_coords[key], dtype=torch.float)
        data.pos = spatial  # 'pos' is a standard attribute in PyTorch Geometric for node positions

        # Add to the dictionary
        graph_data_dict[key] = data

    return graph_data_dict



import numpy as np
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import matplotlib.patches as mpatches

def construct_affinity_matrix(
    coordinates,
    mode='radius',
    cutoff=1.0,
    n_neighbors=5,
    metric='euclidean',
    add_self_loop=False  # New parameter: whether to add self loops
):
    """
    Constructs a neighbors affinity matrix based on either a radius cutoff or a fixed number of neighbors.
    
    Parameters:
    - coordinates: (N, D) array of cell coordinates.
    - mode: 'radius' or 'number' to choose the connectivity method.
    - cutoff: radius value if mode='radius'.
    - n_neighbors: number of neighbors if mode='number'.
    - metric: distance metric to use.
    - add_self_loop: Boolean flag to indicate whether to add self loop (diagonal entries) with weight 1.0.
    
    Returns:
    - affinity_matrix: (N, N) sparse matrix with inverse distance weights.
    """
    N = coordinates.shape[0]
    island_indices = []
    
    if mode == 'radius':
        # Build connectivity based on the radius
        adjacency = radius_neighbors_graph(coordinates, radius=cutoff, mode='connectivity', 
                                             metric=metric, include_self=False)
        
        # Get distances and indices within the radius cutoff
        nbrs = NearestNeighbors(radius=cutoff, metric=metric)
        nbrs.fit(coordinates)
        distances_list, indices_list = nbrs.radius_neighbors(coordinates, return_distance=True)
        
        # Construct the data for the sparse matrix with inverse distance weights
        row = []
        col = []
        data = []
        for i in range(N):
            # For each point, consider its neighbors (obtained from radius_neighbors)
            neighbors = indices_list[i]
            dists = distances_list[i]
            for j, dist in zip(neighbors, dists):
                if dist > 0:  # Exclude self (or potential zero distance cases)
                    row.append(i)
                    col.append(j)
                    data.append(1.0 / dist)
        
        affinity_matrix = csr_matrix((data, (row, col)), shape=(N, N))
        
        # Identify island points based solely on neighboring connectivity (without self loops)
        num_neighbors = adjacency.sum(axis=1).A1
        island_indices = np.where(num_neighbors == 0)[0]
        num_islands = len(island_indices)
        
        # Visualization: Distribution of number of neighbors
        plt.figure(figsize=(8,6))
        plt.hist(num_neighbors, bins=30, color='skyblue', edgecolor='black')
        plt.title('Distribution of Number of Neighbors (Radius Cutoff)')
        plt.xlabel('Number of Neighbors')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
    
    elif mode == 'number':
        if n_neighbors < 0:
            raise ValueError("Number of neighbors must be non-negative.")
        elif n_neighbors == 0:
            # All points are considered islands
            affinity_matrix = csr_matrix((N, N), dtype=float)
            island_indices = np.arange(N)
            num_islands = N
            # Visualization: All points have zero neighbors
            plt.figure(figsize=(8,6))
            plt.hist(np.zeros(N), bins=1, color='lightgreen', edgecolor='black')
            plt.title('All Points are Islands (n_neighbors=0)')
            plt.xlabel('Number of Neighbors')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.show()
            # Optionally add self loops even if points have no neighbors,
            # but typically self loops do not affect island detection.
            if add_self_loop:
                affinity_matrix = affinity_matrix + diags(np.ones(N))
            return affinity_matrix, island_indices
        
        # Ensure n_neighbors does not exceed N-1
        actual_n_neighbors = min(n_neighbors, N-1)
        
        # Use NearestNeighbors to obtain a fixed number of neighbors.
        # Note: n_neighbors+1 is used to include the point itself.
        nbrs = NearestNeighbors(n_neighbors=actual_n_neighbors + 1, metric=metric)
        nbrs.fit(coordinates)
        distances, indices = nbrs.kneighbors(coordinates)
        
        if add_self_loop:
            # If we want to include self loops, we keep the self index (first column) 
            # and compute a safe weight for self connections.
            # Here, we assign a fixed weight of 1.0 to self loops.
            pass  # We will add these self-loop weights later.
        else:
            # Exclude the first column (distance to self) if not adding self loop
            distances = distances[:, 1:]
            indices = indices[:, 1:]
        
        # In case some distances are zero, we avoid division by zero.
        valid_mask = distances > 0
        distances = distances * valid_mask  # Zero stays zero
        indices = indices * valid_mask
        
        # Prepare row indices for the sparse matrix
        row = np.repeat(np.arange(N), actual_n_neighbors if not add_self_loop else actual_n_neighbors + 1)
        col = indices.flatten()
        with np.errstate(divide='ignore'):
            inv_distances = 1.0 / distances.flatten()
        inv_distances[~np.isfinite(inv_distances)] = 0  # Replace infinities or NaNs with 0
        
        data = inv_distances
        affinity_matrix = csr_matrix((data, (row, col)), shape=(N, N))
        
        # Identify island points (excluding any potential self-loop contribution).
        num_neighbors = (affinity_matrix > 0).sum(axis=1).A1
        island_indices = np.where(num_neighbors == 0)[0]
        num_islands = len(island_indices)
        
        # Visualization: Distribution of distances to neighbors
        plt.figure(figsize=(8,6))
        plt.hist(distances.flatten(), bins=30, color='lightgreen', edgecolor='black')
        plt.title('Distribution of Distances to Neighbors (Fixed Number of Neighbors)')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
    
    else:
        raise ValueError("Mode must be 'radius' or 'number'")
    
    # Plot the coordinates and highlight island points
    plt.figure(figsize=(10,8))
    plt.scatter(coordinates[:,0], coordinates[:,1], c='blue', label='Points', alpha=0.6, edgecolor='k', s=50)
    if len(island_indices) > 0:
        plt.scatter(coordinates[island_indices,0], coordinates[island_indices,1], 
                    c='red', label='Island Points', edgecolor='k', s=100, marker='s')
    plt.title('Cell Coordinates with Island Points Highlighted')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"Number of island points: {num_islands}")
    if num_islands > 0:
        print(f"Island point indices: {island_indices}")
    
    # Finally, add self loop edges if requested.
    if add_self_loop:
        # Adding self-loops with weight 1.0 for every point.
        self_loops = diags(np.ones(N))
        affinity_matrix = affinity_matrix + self_loops
    
    return affinity_matrix