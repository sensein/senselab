"""This module provides the implementation of dimensionality reduction.

Dimensionality reduction is a crucial preprocessing step in many machine learning and data analysis pipelines.
It aims to project a high-dimensional dataset into a lower-dimensional subspace while preserving as much of the
significant structure and information as possible. This process is particularly useful for visualization,
noise reduction, and improving computational efficiency in downstream tasks.

This module specifically implements three popular dimensionality reduction techniques: Principal Component Analysis
(PCA), t-Distributed Stochastic Neighbor Embedding (t-SNE), and Uniform Manifold Approximation and Projection (UMAP).

### Task Overview
- Input: High-dimensional data (typically represented as a matrix or tensor), i.e. A tensor of shape (100, 10)
- Output: Lower-dimensional representation of the input data, i.e. A tensor of shape (100, 2)
- Goal: Preserve important structures or relationships in the data while reducing its dimensionality

### Evaluation Metrics
The effectiveness of dimensionality reduction techniques can be evaluated using several metrics, including but not
limited to:
- **Reconstruction Error**: Measures the difference between the original data and its reduced representation when
    projected back to the original space. Commonly used for techniques like PCA.
- **Trustworthiness**: Assesses how much the model preserves neighbor relationships between the high-dimensional
    space and the low-dimensional space.
- **Continuity**: Evaluates how well the local structure of the data is preserved in the reduced space.

### Supported models
- **Principal Component Analysis (PCA)**: PCA is a widely recognized technique in the field of data science
    and machine learning. It excels in linear dimensionality reduction, making it an invaluable tool for
    simplifying complex datasets. PCA's primary strength lies in its ability to capture the most significant
    variance within the data. By identifying the principal components that account for the majority of the variation,
    PCA allows researchers and analysts to reduce the dimensionality of their data while retaining its most important
    features. This makes PCA particularly useful for tasks like data visualization and feature extraction.

- **t-Distributed Stochastic Neighbor Embedding (t-SNE)**: t-SNE is primarily utilized as a visualization technique,
    setting it apart from more general dimensionality reduction methods. It is able to preserve local structures
    within data when projecting it into lower-dimensional spaces, typically two or three dimensions. This property
    makes t-SNE exceptionally good at creating visually interpretable embeddings that can reveal clusters and patterns
    in complex datasets. However, t-SNE is not suitable for tasks that require the preservation of global structures.

- **Uniform Manifold Approximation and Projection (UMAP)**: UMAP is a relatively newer technique that has gained
    significant attention due to its versatility and performance. UMAP is able to balance the preservation of both
    global and local structures within data. Like t-SNE, UMAP can be used visualizing high-dimensional data in two
    or three dimensions. However, UMAP is also effective as a general-purpose non-linear dimensionality reduction
    technique, making it suitable for a wide range of applications where preserving the overall structure of the
    data in lower dimensions is crucial.

#### Summary:
.. include:: ./dimensionality_reduction_table.md

References:
- PCA: https://www.nature.com/articles/s43586-022-00184-w
- t-SNE: https://jmlr.org/papers/v9/vandermaaten08a.html
- UAMP: https://arxiv.org/abs/1802.03426

### Notes:
This module leverages the implementations from scikit-learn for PCA and t-SNE, and the umap-learn library for UMAP.
The reduced data is returned as a PyTorch tensor, allowing seamless integration with PyTorch-based workflows.

Learn more:
- PCA: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
- t-SNE: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
- UMAP: https://umap-learn.readthedocs.io/
"""

from typing import Literal

import pydra
import torch
import umap
from sklearn.decomposition import PCA as sklearn_PCA
from sklearn.manifold import TSNE as sklearn_TSNE


def compute_dimensionality_reduction(
    data: torch.Tensor, model: Literal["pca", "tsne", "umap"] = "pca", n_components: int = 2, **kwargs: object
) -> torch.Tensor:
    """Reduce the dimensionality of the given data using the specified model.

    This function provides three methods: PCA, t-SNE, or UMAP. The reduced data is returned as a PyTorch tensor.

    Args:
        data (torch.Tensor): Input data tensor of shape (n_samples, n_features).
        model ({"pca", "tsne", "umap"}, optional): The dimensionality reduction model to use.
            Choices are:
                "pca" for Principal Component Analysis (default),
                "tsne" for t-Distributed Stochastic Neighbor Embedding, or
                "umap" for Uniform Manifold Approximation and Projection
        n_components (int, optional): Number of dimensions in the output.
            Must be less than or equal to the number of features in the input data.
            Defaults to 2.
        **kwargs: Additional keyword arguments to pass to the chosen model.
            (See below for links to each model's documentation for available parameters.)

    Returns:
        torch.Tensor: The reduced data tensor of shape (n_samples, n_components).

    Raises:
        ValueError: If an invalid model choice is provided or if n_components is
            greater than the number of features in the input data or less than 0.

    Examples:
        >>> data = torch.randn(100, 10)  # 100 samples, 10 features
        >>> data.shape
        torch.Size([100, 10])

        >>> reduced_data = compute_dimensionality_reduction(data)
        >>> print(reduced_data.shape)
        torch.Size([100, 2])

        >>> reduced_data = compute_dimensionality_reduction(data, model="pca", n_components=2, svd_solver="full")
        >>> print(reduced_data.shape)
        torch.Size([100, 2])

        >>> reduced_data = compute_dimensionality_reduction(data, model="tsne", n_components=3, perplexity=30)
        >>> print(reduced_data.shape)
        torch.Size([100, 3])

        >>> reduced_data = compute_dimensionality_reduction(data, model="umap", n_components=4,
            force_approximation_algorithm=False, init='spectral', learning_rate=1.0, n_neighbors=5)
        >>> print(reduced_data.shape)
        torch.Size([100, 4])

    Notes:
        This function uses implementations from scikit-learn for PCA and t-SNE, and the umap-learn
        library for UMAP. For detailed information about each method and its parameters, please
        refer to the following documentation:

        - PCA: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        - t-SNE: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
        - UMAP: https://umap-learn.readthedocs.io/en/latest/api.html
    """
    if n_components > data.shape[1]:
        raise ValueError("n_components must be less than or equal to the number of features in the input data")

    if n_components < 1:
        raise ValueError("n_components must be greater than 0")

    if data.numel() == 0:
        raise ValueError("Input data must not be empty")

    if model == "pca":
        # Perform PCA
        reducer = sklearn_PCA(n_components=n_components, **kwargs)
        reduced_data = reducer.fit_transform(data)

    elif model == "tsne":
        # Perform t-SNE
        reducer = sklearn_TSNE(n_components=n_components, **kwargs)
        reduced_data = reducer.fit_transform(data)

    elif model == "umap":
        # Perform UMAP
        reducer = umap.UMAP(n_components=n_components, **kwargs)
        reduced_data = reducer.fit_transform(data)

    else:
        raise ValueError(f"Invalid model choice: {model}")

    return torch.from_numpy(reduced_data).float()


compute_dimensionality_reduction_pt = pydra.mark.task(compute_dimensionality_reduction)
