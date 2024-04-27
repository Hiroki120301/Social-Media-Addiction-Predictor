from typing import List
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from dowhy import CausalModel
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import HillClimbSearch, BicScore

IMAGES_DIR = './Images'


def causal_ineference(clustered_data):
    df = clustered_data.toPandas()
    model = CausalModel(
        data=df,
        treatment='Total Score',
        outcome='Hours Per Day',
        common_causes=['ADHD Score', 'Anxiety Score',
                       'Self Esteem Score', 'Depression Score']
    )

    # Identify causal effect
    identified_estimand = model.identify_effect()

    # Estimate causal effect
    estimate = model.estimate_effect(identified_estimand,
                                     method_name="backdoor.linear_regression")

    print(estimate)
    # Instantiate a HillClimbSearch object with the data
    hc = HillClimbSearch(df)

    # Use Hill climbing to learn the structure of the Bayesian Network
    best_model_structure = hc.estimate(scoring_method=BicScore(df))

    # Instantiate a BayesianModel with the learned structure
    best_model = BayesianModel(best_model_structure.edges())

    # Fit the model to the data using Bayesian parameter estimation
    best_model.fit(df, estimator=BayesianEstimator)

    # Print the learned graph structure
    print("Learned Causal Graph Structure:")
    print(best_model_structure.edges())

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges from the learned model structure
    G.add_edges_from(best_model_structure.edges())

    # Draw the graph
    plt.figure(figsize=(20, 15))  # Adjust width and height as needed
    plt.title("Learned Causal Graph Structure")
    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color="skyblue",
            font_size=12, font_weight="bold", arrows=True)

    # Show plot
    plt.savefig(f'{IMAGES_DIR}/causal_inference.png')


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
    plt.figure(figsize=(8, 6))
    plt.plot(range(initial_k, K), silhouette_scores)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.savefig(f'{IMAGES_DIR}/silhouette_plot.png')
    plt.close()


def plot_kmeans_clusters(clustered_data):
    clustered_data_pd = clustered_data.toPandas()
    features = ['Anxiety Score', 'Self Esteem Score',
                'Depression Score', 'ADHD Score']

    for i in range(len(features)):
        for j in range(i+1, len(features)):
            plt.figure(figsize=(8, 6))
            plt.scatter(clustered_data_pd[features[i]], clustered_data_pd[features[j]],
                        c=clustered_data_pd['cluster'], cmap='viridis')
            plt.xlabel(f'{features[i]} (Standardized)')
            plt.ylabel(f'{features[j]} (Standardized)')
            plt.title(
                f'K-means Clustering Plot of {features[i]} and {features[j]} Scores')
            plt.colorbar(label='Cluster')
            plt.savefig(
                f'{IMAGES_DIR}/Scatter Plot of {features[i]} and {features[j]} Scores.png')
            plt.close()
