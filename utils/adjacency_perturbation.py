import torch
import numpy as np
import random
from typing import Tuple, Optional


class AdjacencyPerturbation:
    """
    Adjacency Matrix Perturbation Data Augmentation Class
    Supports various perturbation strategies: random addition/deletion of edges, adjusting edge weights, symmetric perturbations, etc.
    """

    def __init__(self,
                 probability: float = 0.5,  # Perturbation probability
                 perturbation_type: str = "mixed",  # "add_delete", "weight_adjust", "symmetric", "mixed"
                 edge_change_ratio: float = 0.1,  # Proportion of edges to be changed
                 weight_noise_std: float = 0.1,  # Standard deviation for weight noise
                 min_edge_weight: float = 0.01,  # Minimum edge weight
                 max_edge_weight: float = 1.0,  # Maximum edge weight
                 preserve_self_loops: bool = True,  # Whether to preserve self-loops
                 ensure_connectivity: bool = True,  # Ensure connectivity of the graph after perturbation
                 device: str = 'cuda'):

        self.probability = probability
        self.perturbation_type = perturbation_type
        self.edge_change_ratio = edge_change_ratio
        self.weight_noise_std = weight_noise_std
        self.min_edge_weight = min_edge_weight
        self.max_edge_weight = max_edge_weight
        self.preserve_self_loops = preserve_self_loops
        self.ensure_connectivity = ensure_connectivity
        self.device = device

    def __call__(self, adj_matrix: torch.Tensor, apply_perturbation: bool = True) -> torch.Tensor:
        """
        Apply perturbation to the adjacency matrix

        Args:
            adj_matrix: Original adjacency matrix [n_nodes, n_nodes]
            apply_perturbation: Whether to apply perturbation (for controlling randomness during training)

        Returns:
            Perturbed adjacency matrix
        """
        if not apply_perturbation or random.random() > self.probability:
            return adj_matrix.clone()

        perturbed_adj = adj_matrix.clone()
        n_nodes = perturbed_adj.size(0)

        if self.perturbation_type == "add_delete":
            perturbed_adj = self._add_delete_edges(perturbed_adj)
        elif self.perturbation_type == "weight_adjust":
            perturbed_adj = self._adjust_edge_weights(perturbed_adj)
        elif self.perturbation_type == "symmetric":
            perturbed_adj = self._symmetric_perturbation(perturbed_adj)
        elif self.perturbation_type == "mixed":

            strategies = ["add_delete", "weight_adjust", "symmetric"]
            chosen_strategy = random.choice(strategies)
            if chosen_strategy == "add_delete":
                perturbed_adj = self._add_delete_edges(perturbed_adj)
            elif chosen_strategy == "weight_adjust":
                perturbed_adj = self._adjust_edge_weights(perturbed_adj)
            else:
                perturbed_adj = self._symmetric_perturbation(perturbed_adj)

        perturbed_adj = self._ensure_symmetry(perturbed_adj)  # Ensure symmetry

        if self.ensure_connectivity:  # Check if the perturbed graph is still connected, to prevent model collapse on disconnected graphs
            perturbed_adj = self._ensure_connectivity_preservation(perturbed_adj, adj_matrix)

        perturbed_adj = torch.clamp(perturbed_adj, self.min_edge_weight,
                                    self.max_edge_weight)  # Clamp the edge weight range to prevent out-of-bound weights after perturbation

        return perturbed_adj

    def _add_delete_edges(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        """Randomly add or delete edges"""
        n_nodes = adj_matrix.size(0)

        # Create a perturbation mask
        mask = ~torch.eye(n_nodes, dtype=torch.bool, device=adj_matrix.device)
        if not self.preserve_self_loops:  # Don't perturb self-loops
            mask = torch.ones(n_nodes, n_nodes, dtype=torch.bool, device=adj_matrix.device)

        total_possible_edges = mask.sum().item()  # Count the number of perturbable edges
        n_changes = int(total_possible_edges * self.edge_change_ratio)  # Calculate the number of edges to perturb

        # Get the indices of perturbable edges from the mask
        flat_indices = torch.where(mask.flatten())[0]
        if len(flat_indices) > n_changes:  # Randomly select edges to perturb
            selected_indices = torch.randperm(len(flat_indices))[:n_changes]
            selected_flat_indices = flat_indices[selected_indices]
        else:
            selected_flat_indices = flat_indices

        # Calculate the 2D indices of the selected edges by converting the flat indices to 2D (i, j)
        selected_i = torch.div(selected_flat_indices, n_nodes, rounding_mode='floor')
        selected_j = selected_flat_indices % n_nodes

        # Randomly add or delete edges
        for i, j in zip(selected_i, selected_j):
            if random.random() < 0.5:  # 50% chance to add an edge
                if adj_matrix[i, j] < self.min_edge_weight:
                    adj_matrix[i, j] = random.uniform(self.min_edge_weight, self.max_edge_weight)
            else:  # 50% chance to delete an edge
                if adj_matrix[i, j] > self.min_edge_weight:
                    adj_matrix[i, j] = self.min_edge_weight

        return adj_matrix

    def _adjust_edge_weights(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        """Adjust existing edge weights by Gaussian perturbation"""
        # Generate normal noise with the same shape as the adjacency matrix
        noise = torch.randn_like(adj_matrix) * self.weight_noise_std

        n_nodes = adj_matrix.size(0)
        edge_mask = adj_matrix > self.min_edge_weight  # Create an edge mask: select edges with weights greater than the threshold as "existing edges"
        if self.preserve_self_loops:  # Self-loops do not participate in perturbation
            diagonal_mask = ~torch.eye(n_nodes, dtype=torch.bool, device=adj_matrix.device)
            edge_mask = edge_mask & diagonal_mask

        adj_matrix[edge_mask] += noise[edge_mask]

        return adj_matrix

    def _symmetric_perturbation(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        """Symmetric perturbation: Maintain graph connectivity but break deterministic structure by adding noise symmetrically or adding edges symmetrically"""
        n_nodes = adj_matrix.size(0)

        # Generate symmetric noise matrix
        random_matrix = torch.randn(n_nodes, n_nodes, device=adj_matrix.device)
        symmetric_noise = (random_matrix + random_matrix.T) / 2 * self.weight_noise_std

        # Add symmetric noise to existing edges
        edge_mask = adj_matrix > self.min_edge_weight
        adj_matrix[edge_mask] += symmetric_noise[edge_mask]

        # Add symmetric edges
        n_new_edges = int(n_nodes * self.edge_change_ratio)
        for _ in range(n_new_edges):
            i, j = random.randint(0, n_nodes - 1), random.randint(0, n_nodes - 1)
            if i != j and adj_matrix[i, j] <= self.min_edge_weight:
                weight = random.uniform(self.min_edge_weight, self.max_edge_weight * 0.3)
                adj_matrix[i, j] = weight
                adj_matrix[j, i] = weight

        return adj_matrix

    def _ensure_symmetry(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        """Ensure the adjacency matrix is symmetric"""
        return (adj_matrix + adj_matrix.T) / 2

    def _ensure_connectivity_preservation(self, perturbed_adj: torch.Tensor,
                                          original_adj: torch.Tensor) -> torch.Tensor:
        """Ensure the perturbed graph preserves basic connectivity"""
        n_nodes = perturbed_adj.size(0)

        for i in range(n_nodes):
            if self.preserve_self_loops:  # Preserve self-loops

                connections = perturbed_adj[i, :].clone()
                connections[i] = 0
                if connections.sum() < self.min_edge_weight:  # Check if the node is unconnected based on weighted degree, if unconnected, connect to an original edge
                    original_connections = original_adj[i,
                                           :] > self.min_edge_weight  # Select connections based on edge threshold
                    if original_connections.sum() > 1:  # Ensure more than one connection besides self-loop
                        valid_indices = torch.where(original_connections)[0]
                        if len(valid_indices) > 0:  # Choose a valid connection and assign it symmetrically
                            chosen_idx = valid_indices[random.randint(0, len(valid_indices) - 1)]
                            perturbed_adj[i, chosen_idx] = original_adj[i, chosen_idx]
                            perturbed_adj[chosen_idx, i] = original_adj[chosen_idx, i]
            else:  # Do not preserve self-loops

                if perturbed_adj[i, :].sum() < self.min_edge_weight:
                    original_connections = original_adj[i, :] > self.min_edge_weight
                    if original_connections.sum() > 0:
                        valid_indices = torch.where(original_connections)[0]
                        if len(valid_indices) > 0:
                            chosen_idx = valid_indices[random.randint(0, len(valid_indices) - 1)]
                            perturbed_adj[i, chosen_idx] = original_adj[i, chosen_idx]
                            perturbed_adj[chosen_idx, i] = original_adj[chosen_idx, i]

        return perturbed_adj


class GraphAugmentationWrapper:  # Graph-level augmentation, no sample generation
    """
    Graph augmentation wrapper that integrates adjacency matrix perturbation into the existing data augmentation pipeline
    """

    def __init__(self,
                 base_transform=None,
                 adj_perturbation: Optional[AdjacencyPerturbation] = None,
                 apply_to_every_batch: bool = False):
        """
        Args:
            base_transform: Base data transformations (e.g., CutMix, RandomCrop, etc.)
            adj_perturbation: Adjacency matrix perturbator
            apply_to_every_batch: Whether to apply adjacency perturbation to every batch
        """
        self.base_transform = base_transform
        self.adj_perturbation = adj_perturbation
        self.apply_to_every_batch = apply_to_every_batch

    def __call__(self, x, y=None, adj_matrix=None):
        """
        Args:
            x: Input data
            y: Labels
            adj_matrix: Adjacency matrix (if perturbation is required)

        Returns:
            If adj_matrix is None, returns (x, y); otherwise returns (x, y, perturbed_adj)
        """

        if self.base_transform is not None:
            if y is not None:
                x = self.base_transform(x, y)
            else:
                x = self.base_transform(x)

        if adj_matrix is not None and self.adj_perturbation is not None:
            perturbed_adj = self.adj_perturbation(adj_matrix, apply_perturbation=True)
            return x, y, perturbed_adj

        return x, y if y is not None else x
