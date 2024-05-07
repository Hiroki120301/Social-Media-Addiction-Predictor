from typing import List
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


def generate_corr_hist(df: pd.DataFrame, variables: List):
    # Calculate correlation for each variable
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 15))
    for i, variable in enumerate(variables):
        corr = df.corr()[variable]

        # Plot correlation
        sns.barplot(x=corr.index, y=corr.values, ax=axes[i//3, i % 3])
        axes[i//3, i % 3].set_title(f'Correlation with {variable}')
        axes[i//3, i % 3].set_xlabel('Variables')
        axes[i//3, i % 3].set_ylabel('Correlation Coefficient')
        axes[i//3, i % 3].tick_params(axis='x', rotation=90)
        axes[i//3, i % 3].set_ylim([-1, 1])

    plt.tight_layout()
    plt.savefig('correlation_histogram.png')  # Save the plot as an image


def generate_heatmap(df: pd.DataFrame):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(corr, annot=True, cmap='YlGnBu', ax=ax)
    plt.title("Heat map of how each feature is correlated to each other")
    plt.savefig('heatmap.png')  # Save the plot as an image


def generate_scatter(x, y, target, x_label, y_label, title, colorbar_label):
    plt.scatter(x, y,
                c=target, cmap='viridis')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.colorbar(label=colorbar_label)
    plt.show()
