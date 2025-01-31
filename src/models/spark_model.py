from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.config import Config

class SparkPredictiveMaintenanceModel:
    def __init__(self, config):
        """Initialize Spark model"""
        self.config = config
        # Initialize Spark session
        self.spark = SparkSession.builder \
            .appName("PredictiveMaintenance") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .getOrCreate()
            
        self.feature_assembler = None
        self.scaler = None
        self.type_indexer = None
        self.rf_model = None
        self.pipeline = None
        
    def prepare_data(self, data_path):
        df = self.spark.read.csv(data_path, header=True, inferSchema=True)
        
        self.feature_assembler = VectorAssembler(
            inputCols=self.config.NUMERICAL_FEATURES,
            outputCol="numerical_features"
        )
        
        self.scaler = StandardScaler(
            inputCol="numerical_features",
            outputCol="scaled_features",
            withStd=True,
            withMean=True
        )
        
        self.type_indexer = StringIndexer(
            inputCol="Type",
            outputCol="type_index"
        )
        
        self.feature_assembler_final = VectorAssembler(
            inputCols=["scaled_features", "type_index"],
            outputCol="features"
        )
        
        return df
    
    def train(self, data_path):
        print("Loading and preparing data...")
        df = self.prepare_data(data_path)
        
        
        self.rf_model = RandomForestClassifier(
            labelCol="Machine failure",
            featuresCol="features",
            numTrees=100,
            maxDepth=10,
            seed=42
        )
        
        
        self.pipeline = Pipeline(stages=[
            self.feature_assembler,
            self.scaler,
            self.type_indexer,
            self.feature_assembler_final,
            self.rf_model
        ])
        
        
        train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
        
        print("Training model...")
        self.model = self.pipeline.fit(train_data)
        
        predictions = self.model.transform(test_data)
        
        evaluator = MulticlassClassificationEvaluator(
            labelCol="Machine failure",
            predictionCol="prediction",
            metricName="accuracy"
        )
        
        accuracy = evaluator.evaluate(predictions)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        feature_importance = self.model.stages[-1].featureImportances
        print("\nFeature Importance:")
        for idx, importance in enumerate(feature_importance):
            print(f"Feature {idx}: {importance:.4f}")
        
        return self.model
    
    def save_model(self, path):
        self.model.write().overwrite().save(path)
    
    def load_model(self, path):
        from pyspark.ml import PipelineModel
        self.model = PipelineModel.load(path)
    
    def predict(self, data):
        if isinstance(data, dict):
            data = [data]
        spark_df = self.spark.createDataFrame(data)
        
        predictions = self.model.transform(spark_df)
        return predictions.select("prediction", "probability").collect()

    def stop_spark(self):
        self.spark.stop()