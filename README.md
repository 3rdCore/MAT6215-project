# MAT6215

Dimensionality reduction techniques are often used to visualize the underlying geometry of a high-dimensional dataset. These methods usually rely on specific similarity measures. In this project, we first approximate the geodesic distance using a diffusion process over the underlying manifold, then we use <a class="tog" href="https://en.wikipedia.org/wiki/Multidimensional_scaling" target="_blank">Multi-Dimentionnal Scaling</a> combined with our previously defined pairwise 'distances' to embed our Manifold in a lower dimensional space. We compare our model with popular algorithms such as PHATE, UMAP, and Isomap on toy datasets and RNA-seq dataset.

## Prerequisites

The external python libraries needed are:
 1. umap-learn
 2. pyDiffMap
 3. seaborn

However, you can simply run the attached Notebook Jupiter that will download everything for you :)

## Usage
Run the Notebook. 
