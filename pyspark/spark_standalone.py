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
from pyspark.sql import SparkSession
import pyspark.pandas as ps
from pyspark_dist_explore import Histogram, hist
from matplotlib import pyplot as plt


DATASET_PATH = '../Data/dataset.csv'


if __name__ == '__main__':
    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName("SocialMediaAddictionPredictor") \
        .getOrCreate()

    # Loading dataset
    print("Reading the dataset...")
    df = spark.read.csv(DATASET_PATH, header=True)
    print("Dataset loaded.")

    # Clean up dataset
    print("Performing data cleanup...")
    df = data_cleanup(df=df)
    print("Data cleanup done.")

    # Data preprocessing
    print("Handling nominal data...")
    df = spark.createDataFrame(handle_nominal_data(df, 'Relationship Status'))
    df = spark.createDataFrame(handle_nominal_data(df, 'Occupation'))
    df = spark.createDataFrame(handle_nominal_data(df, 'Gender'))
    print("Nominal data handled.")
    print(df.show())
    # Plot histograms directly from PySpark DataFrame
    df.toPandas().hist(figsize=(16, 12))
    plt.savefig('data_dist_hist.png')  # Save the plot as an image

    print("Generating heatmap...")
    generate_heatmap(df=df)
    print("Heatmap generated.")

    print("Dropping unnecessary columns...")
    columns_to_drop = ['Male', 'Non-Binary', 'Non-binary']
    df = df.drop(*columns_to_drop)
    print("Columns dropped.")

    print("Generating correlation histograms...")
    generate_corr_hist(df=df, variables=['Hours Per Day', 'ADHD Score', 'Anxiety Score',
                                         'Self Esteem Score', 'Depression Score', 'Total Score'])
    print("Correlation histograms generated.")

    print("Standaridize dataset...")
    standardize_df = standardize_data(df)
    print(standardize_df.show())
