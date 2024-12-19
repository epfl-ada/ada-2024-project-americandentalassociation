import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

movies_path_base = "data/movies_with_summaries.csv"

def load_data_movies(movies_path):
    """
    Load the data from the movies dataset and preprocess it.
    :param movies_path: path to the movies dataset
    :return: the preprocessed data
    """
    data = pd.read_table(movies_path, sep=",")

    data['Genres'] = data['Genres'].fillna("[]")
    data['Genres'] = data['Genres'].str.strip("[]").str.replace("'", "").str.split(", ")

    data['Genres_IMDb'] = data['Genres_IMDb'].fillna("[]")
    data['Genres_IMDb'] = data['Genres_IMDb'].str.strip("[]").str.replace("'", "").str.split(",")

    data['Countries'] = data['Countries'].fillna("[]")
    data['Countries'] = data['Countries'].str.strip("[]").str.replace("'", "").str.split(", ")

    return data

def get_genres(movies_df):
    """
    Get the genres from the movies dataset.
    :param movies_df: the movies dataset
    :return: the genres
    """
    all_genres = movies_df['Genres'].dropna().apply(lambda x: x if isinstance(x, list) else x.split(','))
    genres_count = Counter([genre.strip() for genres in all_genres for genre in genres])
    genres_count_sorted = dict(sorted(genres_count.items(), key=lambda x: x[1], reverse=True))
    return genres_count_sorted

def get_total_genres(genres_1, genres_2, n=30):
    """
    Get the total genres from two genres dictionaries.
    :param genres_1: the first genres dictionary
    :param genres_2: the second genres dictionary
    :param n: the number of genres to consider
    :return: the total genres
    """
    genres_1_r = dict(sorted(genres_1.items(), key=lambda item: item[1], reverse=True)[:n])
    genres_2_r = dict(sorted(genres_2.items(), key=lambda item: item[1], reverse=True)[:n])
    common_genres = list(set(genres_1_r).union(set(genres_2_r)))
    genres_1_new = dict(sorted({genre: genres_1.get(genre, 0) for genre in common_genres}.items(), key=lambda item: item[1], reverse=True))
    genres_2_new = dict(sorted({genre: genres_2.get(genre, 0) for genre in common_genres}.items(), key=lambda item: item[1], reverse=True))
    return genres_1_new, genres_2_new

def get_normalized_values(dict):
    """
    Get the normalized values of a dictionary.
    :param dict: the dictionary
    :return: the normalized values
    """
    total = sum(dict.values())
    vals = np.array(list(dict.values()))
    return vals/total
