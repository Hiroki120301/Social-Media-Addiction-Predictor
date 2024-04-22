from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
//creating spark session
spark = sparkSession.builder \
    .appName("NominalDataHandling") \
    .getOrCreate()
//unsure if that is the correct path to the file, can edit what is in quotations
df = spark.read.csv("CSDS-312-Final-Project/pyspark/Data/dataset.csv", header=True, inferSchema=True) 
categoricalCols = ['Age','Gender','Relationship Status', 'Occupation','Hours Per Day']
indexers = [StringIndexer(inputCol = col, outputCol = col+"_index", handleInvalid = "keep") for col in categoricalCols]
encoders = [OneHotEncoder(inputCol = col+"_index", outputCol = col+"_encoded") for col in categoricalCols]
pipeline = Pipeline(stages = indexers + encoders)
model = pipeline.fit(df)
transformed_df = model.transform(df)
output_df = transformed_df.drop(*[col+"_index" for col in categoricalCols])
spark.stop()

