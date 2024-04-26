
# Importing the model
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


def find_k(df, initial_k, K):
    silhouette_score = dict()

    evaluator = ClusteringEvaluator(predictionCol='prediction',
                                    featuresCol='scaledFeatures',
                                    metricName='silhouette',
                                    distanceMeasure='squaredEuclidean')

    for i in range(initial_k, K):
        kmeans = KMeans(featuresCol='scaledFeatures', k=i)
        model = kmeans.fit(df)
        predictions = model.transform(df)
        score = evaluator.evaluate(predictions)
        silhouette_score[i] = score
        print('Silhouette Score for k =', i, 'is', score)

    return max(silhouette_score, key=silhouette_score.get), list(silhouette_score.values())


def train_fit(df, k=5):
    evaluator = ClusteringEvaluator(predictionCol='cluster',
                                    featuresCol='scaledFeatures',
                                    metricName='silhouette',
                                    distanceMeasure='squaredEuclidean')

    # Trains a k-means model.
    kmeans = KMeans(featuresCol='scaledFeatures', predictionCol='cluster', k=k)
    model = kmeans.fit(df)
    clustered_data = model.transform(df)

    # Printing cluster centers
    centers = model.clusterCenters()
    print("Cluster Centers: ")
    for center in centers:
        print(center)

    wssse = evaluator.evaluate(clustered_data)
    return wssse, clustered_data
