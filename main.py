import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the data.
movies = pd.read_csv('movies.csv')


# Create a TF-IDF Vectorizer object.
tfidf = TfidfVectorizer()

# Replace NaN with an empty string.
movies['genres'] = movies['genres'].fillna('')

# Generate the TF-IDF matrix.
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Compute the cosine similarity matrix.
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a reverse mapping of movie titles and DataFrame indices.
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def recommend_movies(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title.
    idx = indices[title]

    # Get the pairwise similarity scores of all movies with that movie.
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies.
    sim_scores = sim_scores[1:11]

    # Get the movie indices.
    movie_indices = [i[0] for i in sim_scores]

   # Return the top 10 most similar movies.
    return movies['title'].iloc[movie_indices].tolist()

# Test the function.
print(recommend_movies('Toy Story (1995)'))