# Movie Recommender Engine
This project is a simple Movie Recommender Engine that uses cosine similarity to recommend movies that are most similar based on their genres.

# Installation
To run this project, you will need Python and the following Python libraries installed:
    pandas
    scikit-learn
You can install these packages using pip:
    'pip install pandas sklearn'

# Project Structure
This project has the following structure:

main.py: The main Python script that you'll execute to run the recommender.
movies.csv: The movie dataset used for generating recommendations.

# Usage
Navigate to the directory containing the project files.

Run the main.py script.
    'python main.py'
The script will prompt you to input a movie title. The movie title must be one of the titles available in the movies.csv file.
The recommender will output the top 10 movies that are most similar to the movie you entered, based on their genres.

# Methodology
The recommender uses the following process:

1. Load the movie data from the movies.csv file into a pandas DataFrame.
2. Preprocess the data: replace any NaN values in the 'genres' column of the DataFrame with an empty string.
3. Generate a TF-IDF (Term Frequency-Inverse Document Frequency) matrix from the 'genres' column of the DataFrame.
4. Compute a cosine similarity matrix from the TF-IDF matrix. This is used to measure the similarity between the genres of different movies.
5. When a movie title is provided, the system uses the cosine similarity matrix to find the most similar movies.
6. The system then outputs the top 10 most similar movies.