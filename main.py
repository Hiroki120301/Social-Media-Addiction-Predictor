from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from dowhy import CausalModel
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
import networkx as nx
from pgmpy.estimators import HillClimbSearch, BicScore
import dowhy
from data_preprocessing import *


DATASET_PATH = './Data/dataset.csv'


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
    plt.show()


def generate_heatmap(df: pd.DataFrame):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(corr, annot=True, cmap='YlGnBu', ax=ax)
    plt.title("Heat map of how each feature is correlated to each other")
    plt.show()


if __name__ == '__main__':
    print("Reading the dataset...")
    df = pd.read_csv(DATASET_PATH)
    print("Dataset loaded.")

    print("Performing data cleanup...")
    df = data_cleanup(df=df)
    print("Data cleanup done.")

    print("Handling nominal data...")
    df = handle_nominal_data(df, 'Relationship Status')
    df = handle_nominal_data(df, 'Occupation')
    df = handle_nominal_data(df, 'Gender')
    print("Nominal data handled.")

    print("Generating histograms...")
    df.hist(figsize=(16, 12))
    print("Histograms generated.")

    print("Generating heatmap...")
    generate_heatmap(df=df)
    print("Heatmap generated.")

    print("Dropping unnecessary columns...")
    df.drop(columns=['Male', 'Non-Binary',
                     'There are others???', 'Trans', 'unsure '], inplace=True)
    print("Columns dropped.")

    print("Generating correlation histograms...")
    generate_corr_hist(df=df, variables=['Hours Per Day', 'ADHD Score', 'Anxiety Score',
                                         'Self Esteem Score', 'Depression Score', 'Total Score'])
    print("Correlation histograms generated.")
