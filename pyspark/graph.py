from typing import List
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

IMAGES_DIR = './Images'


def generate_heatmap(df):
    # Calculate the correlation matrix
    corr = df.toPandas().corr()

    # Create a heatmap
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(corr, annot=True, cmap='YlGnBu', ax=ax)
    plt.title("Heat map of how each feature is correlated to each other")
    plt.savefig(f'{IMAGES_DIR}/heatmap.png')  # Save the plot as an image


def generate_corr_hist(df: pd.DataFrame, variables: List):
    # Calculate correlation for each variable

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 15))
    for i, variable in enumerate(variables):
        corr = df.toPandas().corr()[variable]

        # Plot correlation
        sns.barplot(x=corr.index, y=corr.values, ax=axes[i//3, i % 3])
        axes[i//3, i % 3].set_title(f'Correlation with {variable}')
        axes[i//3, i % 3].set_xlabel('Variables')
        axes[i//3, i % 3].set_ylabel('Correlation Coefficient')
        axes[i//3, i % 3].tick_params(axis='x', rotation=90)
        axes[i//3, i % 3].set_ylim([-1, 1])

    plt.tight_layout()
    # Save the plot as an image
    plt.savefig(f'{IMAGES_DIR}/correlation_histogram.png')


def plot_silhouette_score(silhouette_scores, initial_k, K):
    plt.clf()
    plt.plot(range(initial_k, K), silhouette_scores)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.savefig(f'{IMAGES_DIR}/silhouette_plot.png')


def plot_kmeans_clusters(clustered_data):
    clustered_data_pd = clustered_data.toPandas()
    # plt.scatter(clustered_data_pd[:, 0], clustered_data_pd[:,
    #             1], c=clustered_data_pd["cluster"], cmap='viridis')
    # plt.title("K-means Clustering with PySpark MLlib")
    # plt.colorbar().set_label("Cluster")
    # plt.savefig('result.png')  # Save the plot as an image

    features = ['Anxiety Score', 'Self Esteem Score',
                'Depression Score', 'ADHD Score']

    for i in range(len(features)):
        for j in range(i+1, len(features)):
            plt.figure()
            plt.scatter(clustered_data_pd[features[i]], clustered_data_pd[features[j]],
                        c=clustered_data_pd['cluster'], cmap='viridis')
            plt.xlabel(f'{features[i]} (Standardized)')
            plt.ylabel(f'{features[j]} (Standardized)')
            plt.title(
                f'Scatter Plot of {features[i]} and {features[j]} Scores')
            plt.colorbar(label='Cluster')
            plt.savefig(
                f'{IMAGES_DIR}/Scatter Plot of {features[i]} and {features[j]} Scores.png')
            plt.close()
