import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, ArrayType
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os
from typing import Dict

# Load environment variables
load_dotenv()

# Function to initialize a Spark session
def initialize_spark():
    return SparkSession.builder \
    .appName("UserProfileAnalysis") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.4,"\
    "org.elasticsearch:elasticsearch-spark-30_2.12:8.11.0,") \
    .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true") \
    .getOrCreate()

# Function to create an Elasticsearch index
def create_index( index_name: str, mapping: Dict):
    try:
        elastic_client = Elasticsearch(
          os.getenv("ELASTIC_URL"),
          api_key=(os.getenv("ELASTIC_API_KEY"))
        )
        # Use Elasticsearch.indices.create method
        elastic_client.indices.create(
            index=index_name,
            body=mapping,
            ignore=400  # Ignore 400 already exists code
        )
        print(f"Created index {index_name} successfully!")

        # close the Elasticsearch connection
        elastic_client.close()
        return True
    except Exception as e:
        print(f"Error creating index {index_name}: {str(e)}")
        return False


review_index_mapping = {
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0
  },
  "mappings": {
    "properties": {
      "movie": {
        "properties": {
          "genres": {
            "type": "keyword"
          },
          "movieId": {
            "type": "keyword"
          },
          "title": {
            "type": "text"
          }
        }
      },
      "rating": {
        "type": "float"
      },
      "timestamp": {
        "type": "date"
      },
      "user": {
        "properties": {
          "age": {
            "type": "integer"
          },
          "gender": {
            "type": "keyword"
          },
          "occupation": {
            "type": "keyword"
          },
          "userId": {
            "type": "keyword"
          }
        }
      }
    }
  }
}

# Function to read data from Kafka topic and return a DataFrame
def read_from_kafka(spark, kafka_bootstrap_servers, kafka_topic):
    return (
        spark
        .readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", kafka_bootstrap_servers)
        .option("subscribe", kafka_topic)
        .load()
    )

# Function to parse JSON data from Kafka message
def parse_kafka_message(data, schema):
    return (
        data
        .selectExpr("CAST(value AS STRING) as json")
        .select(from_json("json", schema).alias("data"))
        .select("data.*")
    )

# Function to write DataFrame to Elasticsearch index
def write_to_elasticsearch(data, index_name, elastic_settings):
    (
        data
        .writeStream
        .format("org.elasticsearch.spark.sql") \
        .outputMode("append") \
        .option("es.resource", "nested_movies_reviews") \
        .option("es.nodes", elastic_settings["url"]) \
        .option("es.port", "9243") \
        .option("es.net.http.auth.user", elastic_settings["user"]) \
        .option("es.net.http.auth.pass", elastic_settings["password"]) \
        .option("es.nodes.wan.only", "true") \
        .option("es.write.operation", "index") \
        .option("checkpointLocation", f"/tmp/{index_name}-checkpoint")
        .start()
        .awaitTermination()
    )

# Define the schema for parsing Kafka messages
kafka_message_schema = StructType([
    StructField("movie", StructType([
        StructField("genres", StringType(), True),
        StructField("movieId", StringType(), True),
        StructField("title", StringType(), True)
    ]), True),
    StructField("rating", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("user", StructType([
        StructField("age", StringType(), True),
        StructField("gender", StringType(), True),
        StructField("occupation", StringType(), True),
        StructField("userId", StringType(), True)
    ]), True)
])


# Example usage
if __name__ == "__main__":
    # Spark initialization
    spark = initialize_spark()

    # Kafka configuration
    kafka_bootstrap_servers = "localhost:9092"
    kafka_topic = "reviews"

    # Elasticsearch configuration
    elastic_settings = {
        "url": os.getenv("ELASTIC_URL"),
        "api_key": os.getenv("ELASTIC_API_KEY"),
        "user" : os.getenv("ELASTIC_USER"),
        "password" : os.getenv("ELASTIC_PASSWORD")
    }

    # Read data from Kafka
    kafka_data = read_from_kafka(spark, kafka_bootstrap_servers, kafka_topic)

    # Parse JSON data from Kafka message
    parsed_data = parse_kafka_message(kafka_data, kafka_message_schema)

    create_index("nested_movies_reviews", review_index_mapping)

    write_to_elasticsearch(parsed_data, "reviews", elastic_settings)