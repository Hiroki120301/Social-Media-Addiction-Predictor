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
    causal_ineference(processed_df)
    final_dataset = standardize_data(processed_df)
    print("finding the best k value for KMeans clustering...")
    best_k, silhouette_scores = find_k(final_dataset, INITIAL_K, K)
    print(f"Best K value for KMeans Clustering is: {best_k}")
    plot_silhouette_score(silhouette_scores, INITIAL_K, K)
    wssse, output = train_fit(final_dataset)
    print(f"Within Set Sum of Squared Errors (WSSSE) = {wssse}")
    plot_kmeans_clusters(output)
    spark.stop()
