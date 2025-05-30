import pandas as pd
from pyspark import SparkConf
from pyspark.sql import SparkSession
import os

os.environ["PYSPARK_PYTHON"] = "python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python"

def get_spark_session():
    # Check if Spark session is already created, else create a new one
    return SparkSession.builder \
        .appName("Stroke Prediction2") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.network.timeout", "800s") \
        .config("spark.executor.heartbeatInterval", "60s") \
        .getOrCreate()

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df
