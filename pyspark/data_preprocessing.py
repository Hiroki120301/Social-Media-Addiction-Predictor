import pandas as pd
import pyspark.pandas as ps
from pyspark.sql.functions import when, col, trim
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from graph import *

COLUMNS_TO_DROP = [
    'Timestamp',
    'Affiliations',
    'Platforms Used',
    'Social Media User?'
]


def data_cleanup(df: ps.DataFrame):
    spark = SparkSession.builder.appName("DataCleaner").getOrCreate()
    df = df.withColumnRenamed('1. What is your age?', 'Age') \
        .withColumnRenamed('2. Gender', 'Gender') \
        .withColumnRenamed('3. Relationship Status', 'Relationship Status') \
        .withColumnRenamed('4. Occupation Status', 'Occupation') \
        .withColumnRenamed('5. What type of organizations are you affiliated with?', 'Affiliations') \
        .withColumnRenamed('6. Do you use social media?', 'Social Media User?') \
        .withColumnRenamed('7. What social media platforms do you commonly use?', 'Platforms Used') \
        .withColumnRenamed('8. What is the average time you spend on social media every day?', 'Hours Per Day') \
        .withColumnRenamed('9. How often do you find yourself using Social media without a specific purpose?', 'ADHD Q1') \
        .withColumnRenamed('10. How often do you get distracted by Social media when you are busy doing something?', 'ADHD Q2') \
        .withColumnRenamed("11. Do you feel restless if you haven't used Social media in a while?", 'Anxiety Q1') \
        .withColumnRenamed('12. On a scale of 1 to 5, how easily distracted are you?', 'ADHD Q3') \
        .withColumnRenamed('13. On a scale of 1 to 5, how much are you bothered by worries?', 'Anxiety Q2') \
        .withColumnRenamed('14. Do you find it difficult to concentrate on things?', 'ADHD Q4') \
        .withColumnRenamed('15. On a scale of 1-5, how often do you compare yourself to other successful people through the use of social media?', 'Self Esteem Q1') \
        .withColumnRenamed('16. Following the previous question, how do you feel about these comparisons, generally speaking?', 'Self Esteem Q2') \
        .withColumnRenamed('17. How often do you look to seek validation from features of social media?', 'Self Esteem Q3') \
        .withColumnRenamed('18. How often do you feel depressed or down?', 'Depression Q1') \
        .withColumnRenamed('19. On a scale of 1 to 5, how frequently does your interest in daily activities fluctuate?', 'Depression Q2') \
        .withColumnRenamed('20. On a scale of 1 to 5, how often do you face issues regarding sleep?', 'Depression Q3')
    df = df.drop(*COLUMNS_TO_DROP)
    # Replace values in 'Gender' column
    df = df.withColumn("Gender",
                       when(trim(col("Gender")) == "NB", "Non-Binary")
                       .when(trim(col("Gender")) == "Nonbinary", "Non-Binary")
                       .when(trim(col("Gender")) == "Non binary", "Non-Binary")
                       .when(trim(col("Gender")) == "Non-binary", "Non-Binary")
                       .otherwise(col("Gender")))

    # Replace values in 'Self Esteem Q2' column
    df = df.withColumn("Self Esteem Q2",
                       when(col("Self Esteem Q2") == 1, 5)
                       .when(col("Self Esteem Q2") == 2, 4)
                       .when(col("Self Esteem Q2") == 3, 3)
                       .when(col("Self Esteem Q2") == 4, 2)
                       .when(col("Self Esteem Q2") == 5, 1)
                       .otherwise(col("Self Esteem Q2")))
    # Define lists of column names
    ADHD = ['ADHD Q1', 'ADHD Q2', 'ADHD Q3', 'ADHD Q4']
    Anxiety = ['Anxiety Q1', 'Anxiety Q2']
    SelfEsteem = ['Self Esteem Q1', 'Self Esteem Q2', 'Self Esteem Q3']
    Depression = ['Depression Q1', 'Depression Q2', 'Depression Q3']

    # Calculate scores and add them as new columns
    df = df.withColumn("ADHD Score", sum(col(c) for c in ADHD))
    df = df.withColumn("Anxiety Score", sum(col(c) for c in Anxiety))
    df = df.withColumn("Self Esteem Score", sum(col(c) for c in SelfEsteem))
    df = df.withColumn("Depression Score", sum(col(c) for c in Depression))

    # Calculate total score and add it as a new column
    total_columns = ['ADHD Score', 'Anxiety Score',
                     'Self Esteem Score', 'Depression Score']
    df = df.withColumn("Total Score", sum(col(c) for c in total_columns))

    # Deleting question columns and timestamp columns as they are no longer used
    # Drop columns
    columns_to_drop = ADHD + Anxiety + SelfEsteem + Depression
    df = df.drop(*columns_to_drop)

    # Replace values in 'Hours Per Day' column
    df = df.withColumn("Hours Per Day", trim(df["Hours Per Day"]))
    df.select("Hours Per Day").distinct().show()
    df = df.withColumn("Hours Per Day",
                       when(df["Hours Per Day"] == "More than 5 hours", 5.5)
                       .when(df["Hours Per Day"] == "Between 2 and 3 hours", 2.5)
                       .when(df["Hours Per Day"] == "Between 3 and 4 hours", 3.5)
                       .when(df["Hours Per Day"] == "Between 1 and 2 hours", 1.5)
                       .when(df["Hours Per Day"] == "Between 4 and 5 hours", 4.5)
                       .when(df["Hours Per Day"] == "Less than an Hour", 0.5)
                       .otherwise(df["Hours Per Day"]))

    # Replace value in 'Age' column
    df = df.withColumn("Age", when(df["Age"] == 91, 19).otherwise(df["Age"]))
    df = df.withColumn("Age", col("Age").cast("bigint"))
    df = df.withColumn("Hours Per Day", col("Hours Per Day").cast("bigint"))
    return df


def handle_nominal_data(df, feature):
    pandas_df = df.toPandas()
    dummy_df = pd.get_dummies(pandas_df[feature], drop_first=True)

    # Convert True/False values in dummy_df to 0/1
    dummy_df = dummy_df.astype('int64')

    # Concatenate dummy variables with original DataFrame
    df = pd.concat([pandas_df, dummy_df], axis=1)

    # Drop the original 'Relationship Status' column
    df.drop(columns=[feature], inplace=True)

    return df


def standardize_data(df):

    vec_assembler = VectorAssembler(inputCols=df.columns,
                                    outputCol='features')

    final_data = vec_assembler.transform(df)

    scaler = StandardScaler(inputCol="features",
                            outputCol="scaledFeatures",
                            withStd=True,
                            withMean=False)

    # Compute summary statistics by fitting the StandardScaler
    scalerModel = scaler.fit(final_data)

    # Normalize each feature to have unit standard deviation.
    final_data = scalerModel.transform(final_data)

    return final_data


def preprocess(spark, df):
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
    print(df.show(5))

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

    # print("Standaridize dataset...")
    # final_df = standardize_data(df)
    # print(final_df.show(5))

    return df
