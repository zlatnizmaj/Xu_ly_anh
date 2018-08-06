from sklearn import datasets
import sklearn.datasets
from scikits.crab.metrics import pearson_correlation
from sklearn.base import BaseEstimator
from scikits.crab.models import MatrixPreferenceDataModel
from scikits.crab.metrics import pearson_correlation
from scikits.crab.similarities import UserSimilarity
from scikits.crab.recommenders.knn import UserBasedRecommender

movies = datasets.load_sample_movies() # Load the dataset
model = MatrixPreferenceDataModel(movies.data) # Build the model
similarity = UserSimilarity(model, pearson_correlation) # Build the similarity
recommender = UserBasedRecommender(model, similarity, with_preference=True) # Build the User based recommender

recommender.recommend(7) # Recommend items for the user 7