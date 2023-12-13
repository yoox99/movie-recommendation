import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from elasticsearch import Elasticsearch
from elastic_cloud import cloud_id,api_key


es = Elasticsearch(cloud_id=cloud_id, api_key=api_key)

def get_my_movie_detaile(es,title) :
    
    query = {
          "_source": ["movieId","genre", "movie_title", "rating","release_date"],
                "query": {
                    "match": {
                        "movie_title": title
              }
              }
            }

    response2 = es.search(index="movies-user_rating", body=query)

    return(response2['hits']['hits'][0]["_source"])

def get_user_ids(title,es) :
      
  query =  { "query": {
      "match": {"movie_title": str(title).strip()}}
      }

  response = es.search(index="movies-user_rating", body=query)

  userid_list = []

  if response['hits']['total']['value'] > 0 :
        movie_info = response['hits']['hits'] 

        for i in movie_info :
            userid_list.append(i['_source']['userId'])

        return userid_list
  else :
      print("Movie not found")

def get_moviesids(list_ids) :
    # Create a Spark session :
    spark = SparkSession.builder.appName("ModelUsage").getOrCreate()

    model_path= 'C:/Users/Youcode/Desktop/zed/film-recommender_with_ia/ALS/best_model'

    # Load the ALS model :
    loaded_model = ALSModel.load(model_path)

    # Create a DataFrame with the list of user IDs
    user_ids_df = spark.createDataFrame([(user_id,) for user_id in list_ids], ['userId'])

    user_recommendations = loaded_model.recommendForUserSubset(user_ids_df, 5)

    recommendations = user_recommendations.select('recommendations')

    recommendations_list = [list(map(lambda x: (x['movieId'], x['rating']), row['recommendations'])) for row in recommendations.collect()]

    all_recommendations = []
    for row in recommendations_list:
        all_recommendations.extend(row)

    all_recommendations_ids = [i[0] for i in all_recommendations]

    spark.stop()

    return all_recommendations_ids


def get_movies_information(movieId_list,es) :

        query = {
          "_source": ["movieId","genre", "movie_title", "rating","release_date"],
          "query": {
            "terms": {
              "movieId": movieId_list
            }
          }
        }

        response2 = es.search(index="movies-user_rating", body=query)

        movies_list = []
        for j in response2['hits']['hits'] :
            movies_list.append(j['_source'])

        return(movies_list)
