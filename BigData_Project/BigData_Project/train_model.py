from pyspark.sql import SparkSession
from pipeline import preprocess_and_train  

if __name__ == "__main__":
    
    spark = SparkSession.builder \
        .appName("Stroke Model Training") \
        .config("dfs.permissions.enabled", "false") \
        .getOrCreate()

    #
    df = spark.read.csv("C:\\BigData_Project\\Data\\Healthcare-stroke-data_after preprocess.csv", header=True, inferSchema=True)

    
    preprocess_and_train(df, model_path="C:/BigData_Project/Model")

    print("✅ Đã huấn luyện và lưu model.")
