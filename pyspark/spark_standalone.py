from pyspark.sql import SparkSession
from matplotlib import pyplot as plt
import reina
from data_preprocessing import *
from graph import *
from model import *

DATASET_PATH = './Data/dataset.csv'
INITIAL_K = 2
K = 20

if __name__ == '__main__':
    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName("SocialMediaAddictionPredictor") \
        .getOrCreate()

    # Loading dataset
    print("Reading the dataset...")
    df = spark.read.csv(DATASET_PATH, header=True)
    print("Dataset loaded.")

    processed_df = preprocess(spark=spark, df=df)

    final_dataset = standardize_data(processed_df)
    print("finding the best k value for KMeans clustering...")
    best_k, silhouette_scores = find_k(final_dataset, INITIAL_K, K)
    print(silhouette_scores)
    plot_silhouette_score(silhouette_scores, INITIAL_K, K)
    spark.stop()
