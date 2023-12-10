from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

# Initialisation de la session Spark
spark = SparkSession.builder.appName("MovieRecommendation").getOrCreate()

# Charger l'ensemble de données MovieLens (assurez-vous que le chemin est correct)
data = spark.read.text("chemin/vers/votre/ensemble-de-donnees")

# Prétraitement des données
ratings = data.rdd.map(lambda l: l.split(','))\
    .map(lambda l: Row(userId=int(l[0]), movieId=int(l[1]),
                      rating=float(l[2]), timestamp=int(l[3])))
ratings_df = spark.createDataFrame(ratings)

# Division des données en ensembles de formation et de test
(training, test) = ratings_df.randomSplit([0.8, 0.2])

# Création du modèle ALS
als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating")

# Entraînement du modèle
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
