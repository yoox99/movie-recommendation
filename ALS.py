from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

# Initialisation de la session Spark
spark = SparkSession.builder.appName("MovieRecommendation").getOrCreate()

# Charger les données des utilisateurs (u.user)
users_data = spark.read.text("api/data/u.user")
users = users_data.rdd.map(lambda l: l.split('|')).toDF(['userId', 'age', 'gender', 'occupation', 'zipCode'])

# Charger les données des films (u.item)
movies_data = spark.read.text("api/data/u.item")
movies = movies_data.rdd.map(lambda l: l.split('|')).toDF(['movieId', 'title', 'release_date', '...'])  # Ajoutez les colonnes appropriées

# Charger les données de notation (u.data)
ratings_data = spark.read.text("api/data/u.data")
ratings = ratings_data.rdd.map(lambda l: l.split('\t')).toDF(['userId', 'movieId', 'rating', 'timestamp'])

# Fusionner les DataFrames pour former l'ensemble de données complet
movie_ratings = ratings.join(movies, 'movieId').join(users, 'userId')

# Diviser l'ensemble de données en ensembles de formation et de test
(training, test) = movie_ratings.randomSplit([0.8, 0.2])

# Création du modèle ALS
als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating")
model = als.fit(training)

# Évaluation du modèle sur l'ensemble de test
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) = " + str(rmse))

# Génération de recommandations pour tous les utilisateurs
userRecs = model.recommendForAllUsers(10)

# Affichage des recommandations pour un utilisateur spécifique (remplacez 'userId' par l'ID de l'utilisateur souhaité)
user_id = 1
userRecs.filter(userRecs['userId'] == user_id).show(truncate=False)

# Fermeture de la session Spark
spark.stop()
