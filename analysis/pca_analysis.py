import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns


class PCAAnalyzer:
    """PCA analysis for cognitive pattern activations."""
    
    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.pca_models = {}
        self.scalers = {}
        self.transformed_data = {}
        
    def compute_pca(
        self, 
        activations: Dict[str, torch.Tensor], 
        pattern_name: str,
        standardize: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Compute PCA on activations for a cognitive pattern.
        
        Args:
            activations: Dictionary of activation tensors
            pattern_name: Name of the cognitive pattern
            standardize: Whether to standardize features before PCA
            
        Returns:
            Dictionary of PCA-transformed activations
        """
        transformed = {}
        
        for key, tensor in activations.items():
            # Convert to numpy and reshape if needed
            if tensor.dim() > 2:
                # Flatten extra dimensions
                data = tensor.view(tensor.size(0), -1).cpu().numpy()
            else:
                data = tensor.cpu().numpy()
            
            if data.shape[0] < 2:
                print(f"Warning: Not enough samples for PCA in {key}")
                continue
                
            # Standardize if requested
            scaler_key = f"{pattern_name}_{key}"
            if standardize:
                if scaler_key not in self.scalers:
                    self.scalers[scaler_key] = StandardScaler()
                    data = self.scalers[scaler_key].fit_transform(data)
                else:
                    data = self.scalers[scaler_key].transform(data)
            
            # Determine n_components
            n_comp = self.n_components
            if n_comp is None:
                n_comp = min(data.shape[0] - 1, data.shape[1], 50)  # Cap at 50
            else:
                n_comp = min(n_comp, data.shape[0] - 1, data.shape[1])
            
            # Fit PCA
            pca_key = f"{pattern_name}_{key}"
            if pca_key not in self.pca_models:
                self.pca_models[pca_key] = PCA(n_components=n_comp)
                transformed[key] = self.pca_models[pca_key].fit_transform(data)
            else:
                transformed[key] = self.pca_models[pca_key].transform(data)
        
        # Store transformed data
        if pattern_name not in self.transformed_data:
            self.transformed_data[pattern_name] = {}
        self.transformed_data[pattern_name].update(transformed)
        
        return transformed
    
    def get_explained_variance_ratio(self, pattern_name: str, layer_key: str) -> np.ndarray:
        """Get explained variance ratio for a specific pattern and layer."""
        pca_key = f"{pattern_name}_{layer_key}"
        if pca_key in self.pca_models:
            return self.pca_models[pca_key].explained_variance_ratio_
        return np.array([])
    
    def get_cumulative_explained_variance(self, pattern_name: str, layer_key: str) -> np.ndarray:
        """Get cumulative explained variance for a specific pattern and layer."""
        explained_var = self.get_explained_variance_ratio(pattern_name, layer_key)
        return np.cumsum(explained_var)
    
    def plot_explained_variance(
        self, 
        pattern_name: str, 
        layer_key: str, 
        save_path: Optional[str] = None
    ) -> None:
        """Plot explained variance ratio."""
        explained_var = self.get_explained_variance_ratio(pattern_name, layer_key)
        cumulative_var = self.get_cumulative_explained_variance(pattern_name, layer_key)
        
        if len(explained_var) == 0:
            print(f"No PCA data found for {pattern_name}_{layer_key}")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Individual explained variance
        ax1.bar(range(len(explained_var)), explained_var)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title(f'Explained Variance - {pattern_name} - {layer_key}')
        
        # Cumulative explained variance
        ax2.plot(range(len(cumulative_var)), cumulative_var, 'bo-')
        ax2.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title(f'Cumulative Explained Variance - {pattern_name} - {layer_key}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_pca_scatter(
        self, 
        pattern_names: List[str], 
        layer_key: str,
        components: Tuple[int, int] = (0, 1),
        save_path: Optional[str] = None
    ) -> None:
        """Plot PCA scatter plot comparing multiple patterns."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(pattern_names)))
        
        for i, pattern_name in enumerate(pattern_names):
            if (pattern_name in self.transformed_data and 
                layer_key in self.transformed_data[pattern_name]):
                
                data = self.transformed_data[pattern_name][layer_key]
                if data.shape[1] > max(components):
                    ax.scatter(
                        data[:, components[0]], 
                        data[:, components[1]], 
                        c=[colors[i]], 
                        label=pattern_name,
                        alpha=0.7
                    )
        
        ax.set_xlabel(f'PC{components[0] + 1}')
        ax.set_ylabel(f'PC{components[1] + 1}')
        ax.set_title(f'PCA Scatter Plot - {layer_key}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def get_principal_components(self, pattern_name: str, layer_key: str) -> np.ndarray:
        """Get the principal components (loadings) for interpretation."""
        pca_key = f"{pattern_name}_{layer_key}"
        if pca_key in self.pca_models:
            return self.pca_models[pca_key].components_
        return np.array([])
    
    def transform_new_data(
        self, 
        new_activations: torch.Tensor, 
        pattern_name: str, 
        layer_key: str
    ) -> np.ndarray:
        """Transform new activations using fitted PCA."""
        pca_key = f"{pattern_name}_{layer_key}"
        scaler_key = f"{pattern_name}_{layer_key}"
        
        if pca_key not in self.pca_models:
            raise ValueError(f"PCA not fitted for {pca_key}")
        
        # Convert and reshape
        if new_activations.dim() > 2:
            data = new_activations.view(new_activations.size(0), -1).cpu().numpy()
        else:
            data = new_activations.cpu().numpy()
        
        # Apply scaler if available
        if scaler_key in self.scalers:
            data = self.scalers[scaler_key].transform(data)
        
        # Transform with PCA
        return self.pca_models[pca_key].transform(data)
    
    def get_pattern_separation(
        self, 
        pattern1: str, 
        pattern2: str, 
        layer_key: str,
        metric: str = 'euclidean'
    ) -> float:
        """Compute separation between two patterns in PCA space."""
        if (pattern1 not in self.transformed_data or 
            pattern2 not in self.transformed_data or
            layer_key not in self.transformed_data[pattern1] or
            layer_key not in self.transformed_data[pattern2]):
            return 0.0
        
        data1 = self.transformed_data[pattern1][layer_key]
        data2 = self.transformed_data[pattern2][layer_key]
        
        # Compute centroids
        centroid1 = np.mean(data1, axis=0)
        centroid2 = np.mean(data2, axis=0)
        
        if metric == 'euclidean':
            return np.linalg.norm(centroid1 - centroid2)
        elif metric == 'cosine':
            dot_product = np.dot(centroid1, centroid2)
            norms = np.linalg.norm(centroid1) * np.linalg.norm(centroid2)
            return 1 - (dot_product / norms) if norms > 0 else 0.0
        
        return 0.0