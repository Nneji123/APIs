import difflib

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("movies.zip")
features = ["keywords", "cast", "genres", "director"]

for feature in features:
    df[feature] = df[feature].fillna("")


def combined_features(row):
    return (
        row["keywords"]
        + " "
        + row["cast"]
        + " "
        + row["genres"]
        + " "
        + row["director"]
    )


df["combined_features"] = df.apply(combined_features, axis=1)

Tfidf_vect = TfidfVectorizer()
vector_matrix = Tfidf_vect.fit_transform(df["combined_features"])
vector_matrix.toarray()

cosine_sim = cosine_similarity(vector_matrix)


def get_index_from_title(title):
    search = difflib.get_close_matches(title, df["title"])[0]
    return df[df.title == search]["index"].values[0]


def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]


def check_movie(m_name):
    movie_index = get_index_from_title(m_name)
    similar_movies = list(enumerate(cosine_sim[movie_index]))
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
    mv = get_suggestions(sorted_similar_movies)
    return mv


def get_suggestions(sorted_similar_movies):
    i = 0
    movies = ""
    for movie in sorted_similar_movies:
        t = get_title_from_index(movie[0])
        movies = movies + t + "\n"

        i = i + 1
        if i > 10:
            print(movies)
            return movies


def check(enter_movie_name):
    mvs = check_movie(enter_movie_name)
    return mvs
