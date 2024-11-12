'''
Question 1. How do global crises and significant world events shape the production, themes, and budget allocations of movies?
This question could explore whether events like environmental disasters, political turmoil, or economic recessions lead 
to shifts in movie content, such as an increase in dystopian themes, and how these events impact production budgets and 
genre popularity.
'''

import pandas as pd
from collections import Counter
import ast
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Import data
data = pd.read_csv("data/filtered_movie_data.csv")
df = pd.DataFrame(data)

# Function to count genres for a specific year and return total movies
def count_genres_by_year(dataframe, year):
    filtered_df = dataframe[dataframe['Year'] == year].copy()
    filtered_df.loc[:, 'Genres'] = filtered_df['Genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    all_genres = [genre for genres in filtered_df['Genres'] for genre in genres]
    genre_counts = Counter(all_genres)
    genre_counts_df = pd.DataFrame(genre_counts.items(), columns=['Genre', 'Count']).sort_values(by='Count', ascending=False).reset_index(drop=True)
    return genre_counts_df, len(all_genres)

# Function to calculate the difference in genre counts between two years
def calc_genre_differences(counts_year1, counts_year2, year1, year2):
    merged_counts = pd.merge(counts_year1, counts_year2, on='Genre', how='outer', suffixes=(f'_{year1}', f'_{year2}')).fillna(0)
    merged_counts['Difference'] = merged_counts[f'Count_{year1}'] - merged_counts[f'Count_{year2}']
    merged_counts = merged_counts.sort_values(by='Difference', key=abs, ascending=False).reset_index(drop=True)
    return merged_counts

# Function to calculate relative genre growth
def calc_genre_growth(counts_year1, counts_year2, year1, year2):
    merged_counts = pd.merge(counts_year1, counts_year2, on='Genre', how='outer', suffixes=(f'_{year1}', f'_{year2}')).fillna(0)
    merged_counts['Difference'] = merged_counts[f'Count_{year2}'] - merged_counts[f'Count_{year1}']
    merged_counts['Percentage_Increase'] = merged_counts.apply(
        lambda row: (row['Difference'] / row[f'Count_{year1}']) * 100 if row[f'Count_{year1}'] > 0 else float('inf'),
        axis=1
    )
    merged_counts = merged_counts.sort_values(by='Percentage_Increase', ascending=False).reset_index(drop=True)
    return merged_counts

# Function to calculate proportional difference of genres between two years
def calc_relative_growth(counts_year1, counts_year2, year1, year2, total_movies_year1, total_movies_year2):
    # Add proportions relative to the total number of movies for each year
    counts_year1['Proportion'] = counts_year1['Count'] / total_movies_year1
    counts_year2['Proportion'] = counts_year2['Count'] / total_movies_year2
    
    # Merge and keep the suffixes to distinguish each year's proportions
    merged_counts = pd.merge(counts_year1, counts_year2, on='Genre', how='outer', suffixes=(f'_{year1}', f'_{year2}')).fillna(0)
    merged_counts[f'Proportion_Difference'] = merged_counts[f'Proportion_{year2}'] - merged_counts[f'Proportion_{year1}']
    
    return merged_counts.sort_values(by='Proportion_Difference', ascending=False).reset_index(drop=True)

# Clustering function to categorize genres by growth patterns
def cluster_genres_by_growth(merged_counts):
    # Check that necessary columns are present and fill NaNs
    clustering_data = merged_counts[[f'Proportion_{year1}', f'Proportion_{year2}', 'Proportion_Difference']].fillna(0)
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    merged_counts['Cluster'] = kmeans.fit_predict(clustering_data)
    
    return merged_counts[['Genre', f'Proportion_{year1}', f'Proportion_{year2}', 'Proportion_Difference', 'Cluster']]

def plot_clusters(merged_counts, year1, year2, n_clusters=5):
    # Perform clustering with n_clusters
    clustering_data = merged_counts[[f'Proportion_{year1}', f'Proportion_{year2}', 'Proportion_Difference']].fillna(0)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    merged_counts['Cluster'] = kmeans.fit_predict(clustering_data)
    
    # Scatter plot of Proportion changes colored by cluster
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(merged_counts[f'Proportion_{year1}'], merged_counts[f'Proportion_{year2}'], 
                          c=merged_counts['Cluster'], cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(f'Proportion in {year1}')
    plt.ylabel(f'Proportion in {year2}')
    plt.title(f'Genre Clusters Based on Proportion Change ({year1} to {year2})')
    plt.show()

# Analysis for the start of Iraq War period (2003)
year1, year2 = 2002, 2003
counts_year1, total_movies_year1 = count_genres_by_year(df, year1)
counts_year2, total_movies_year2 = count_genres_by_year(df, year2)

# Calculating genre differences
print("Genre Counts:")
print(count_genres_by_year(df, year1)[0], count_genres_by_year(df, year2)[0])
print("Genre Differences:")
print(calc_genre_differences(counts_year1, counts_year2, year1, year2))
print("Genre Growth:")
print(calc_genre_growth(counts_year1, counts_year2, year1, year2).head(10))
relative_growth = calc_relative_growth(counts_year1, counts_year2, year1, year2, total_movies_year1, total_movies_year2)
print("Relative Growth:")
print(relative_growth)
print("Clustering by Growth Patterns:")
clustered_genres = cluster_genres_by_growth(relative_growth)
print(clustered_genres)
plot_clusters(relative_growth, year1, year2, n_clusters=5)