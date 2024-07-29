| Feature             | PCA                           | t-SNE                                | UMAP                                        |
|---------------------|-------------------------------|--------------------------------------|---------------------------------------------|
| Type                | Linear                        | Non-linear                           | Non-linear                                  |
| Preserves           | Global structure (variance)   | Local structure, cluster patterns    | Local and global structure to some extent   |
| Filters out         | Noise, less important variance     | Global structure, distances between clusters | Less important variance, noise |
| Best for            | - Understanding main sources of variation<br>- Data compression<br>- Preprocessing<br>- Linear relationships | - Visualizing high-dimensional data<br>- Discovering clusters or patterns<br>- Exploratory data analysis | - General-purpose dimensionality reduction<br>- Balancing local and global structure preservation |
| Limitations         | - Assumes linear relationships<br>- May miss complex patterns | - Can distort global structure<br>- Non-deterministic | - Sensitive to parameters<br>- Does not completely preserve density |
