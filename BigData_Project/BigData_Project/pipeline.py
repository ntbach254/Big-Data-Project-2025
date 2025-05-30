from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml import Pipeline
from pyspark.sql.types import DoubleType
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def preprocess_and_train(df_spark, model_path="C:/BigData_Project/Model_GBT"):
    
    df = df_spark.withColumn("hypertension", when(col("hypertension") == "Yes", 1).otherwise(0).cast("int")) \
                 .withColumn("heart_disease", when(col("heart_disease") == "Yes", 1).otherwise(0).cast("int")) \
                 .withColumn("stroke", when(col("stroke") == "Yes", 1).otherwise(0).cast("int")) \
                 .withColumn("age", col("age").cast("int")) \
                 .withColumn("avg_glucose_level", col("avg_glucose_level").cast("float"))

    print(f"⚠️ Số lượng bản ghi sau tiền xử lý: {df.count()}")
    df.select("hypertension", "heart_disease", "age", "avg_glucose_level", "stroke").show(5)

    
    features = ["hypertension", "heart_disease", "age", "avg_glucose_level"]

    
    assembler = VectorAssembler(inputCols=features, outputCol="features", handleInvalid="skip")
    gbt = GBTClassifier(featuresCol="features", labelCol="stroke", maxIter=100)  # GBT không có numTrees, dùng maxIter

    pipeline = Pipeline(stages=[assembler, gbt])

    
    pipeline_model = pipeline.fit(df)

    
    predictions = pipeline_model.transform(df)

    
    evaluator = MulticlassClassificationEvaluator(labelCol="stroke", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)

    print(f"✅ Accuracy của mô hình Gradient Boosted Trees trên tập huấn luyện: {accuracy:.4f}")

    
    pipeline_model.write().overwrite().save(model_path)
    print(f"✅ Mô hình Gradient Boosted Trees đã được lưu tại: {model_path}")
