import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from dowhy import CausalModel
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
import networkx as nx
from pgmpy.estimators import HillClimbSearch, BicScore
import dowhy
from data_preprocessing import *
from graph import *


DATASET_PATH = './Data/dataset.csv'


if __name__ == '__main__':
    # Loading dataset
    print("Reading the dataset...")
    df = pd.read_csv(DATASET_PATH)
    print("Dataset loaded.")

    # Clean up dataset
    print("Performing data cleanup...")
    df = data_cleanup(df=df)
    print("Data cleanup done.")

    # Data preprocessing
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

    print("Standaridize dataset...")
    df_std = standardize_data(df=df)
    print(df_std.head())
