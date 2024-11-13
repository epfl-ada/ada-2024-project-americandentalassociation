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
import seaborn as sns

# Import data
data = pd.read_csv("data/filtered_movie_data.csv")
df = pd.DataFrame(data)


'''
Plotting function
'''
def name_plot(ylabel, title):
    plt.xticks(rotation=90)
    plt.ylabel(ylabel)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(title)
    plt.show()
    plt.close()

'''
Count of different genres released; change, relative change, proportion, clustering for relative change
'''
def count_genre_over_years(dataframe, genre):
    dataframe['Genres'] = dataframe['Genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    genre_counts_by_year = (
        dataframe[dataframe['Genres'].apply(lambda genres: genre in genres)]
        .groupby('Year')
        .size()
        .reset_index(name='Count')
    )
    return genre_counts_by_year

def top_years_for_genre(genre_counts_by_year, top_n=20):
    top_years_chronological = genre_counts_by_year.nlargest(top_n, 'Count')
    return top_years_chronological

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
    merged_counts = merged_counts.sort_values(by='Difference', ascending=False).reset_index(drop=True)
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

'''
Clustering Attempt
'''
# Clustering function to categorize genres by growth patterns
def cluster_genres_by_relative_change(merged_counts, n_clusters=5):
    # Replace infinite values in 'Percentage_Increase' with a large finite value for clustering purposes
    merged_counts['Percentage_Increase'] = merged_counts['Percentage_Increase'].replace([np.inf, -np.inf], 1000)

    # Use 'Percentage_Increase' for clustering
    clustering_data = merged_counts[['Percentage_Increase']].fillna(0)
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    merged_counts['Cluster'] = kmeans.fit_predict(clustering_data)
    
    return merged_counts[['Genre', 'Percentage_Increase', 'Cluster']]

# Visualization for Clustering based on Percentage Increase
def plot_clusters_relative_change(merged_counts, n_clusters=5):
    # Run the clustering based on relative change
    clustered_data = cluster_genres_by_relative_change(merged_counts, n_clusters)
    
    # Plot the clusters
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(clustered_data['Percentage_Increase'], clustered_data['Genre'], 
                          c=clustered_data['Cluster'], cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Percentage Increase')
    plt.ylabel('Genre')
    plt.title(f'Genre Clusters Based on Relative Change ({year1} to {year2})')
    plt.show()

def get_genres_in_cluster(clustered_data, cluster_label):
    # Filter the DataFrame for the specified cluster
    genres_in_cluster = clustered_data[clustered_data['Cluster'] == cluster_label]
    return genres_in_cluster[['Genre', 'Percentage_Increase']]


"""
End of functions definitions
Start of some analysis
"""

# Analysis for the Vietnam War
year1, year2, year3 = 1959, 1967, 1975
counts_year1, total_movies_year1 = count_genres_by_year(df, year1)
counts_year2, total_movies_year2 = count_genres_by_year(df, year2)
counts_year3, total_movies_year3 = count_genres_by_year(df, year3)

# Calculating genre differences
print("Genre Counts:")
print(count_genres_by_year(df, year1)[0], count_genres_by_year(df, year2)[0])



abs_difference_year1_year2 = calc_genre_differences(counts_year1, counts_year2, year1, year2)
abs_difference_year2_year3 = calc_genre_differences(counts_year2, counts_year3, year2, year3)

# Create subplots with two plots side by side
fig, axes = plt.subplots(1, 2, figsize=(20, 8))  # 1 row, 2 columns

# First plot: Absolute difference for year1 and year2
sns.barplot(data=abs_difference_year1_year2.head(30), x="Genre", y="Difference", ax=axes[0])
axes[0].set_title(f"Absolute Growth in Genres from {year1} to {year2} - Top 30")
axes[0].set_xlabel("Genre")
axes[0].set_ylabel("Absolute Difference")
axes[0].tick_params(axis='x', rotation=90)  # Rotate x-axis labels for readability

# Second plot: Absolute difference for year2 and year3
sns.barplot(data=abs_difference_year2_year3.head(30), x="Genre", y="Difference", ax=axes[1])
axes[1].set_title(f"Absolute Growth in Genres from {year2} to {year3} - Top 30")
axes[1].set_xlabel("Genre")
axes[1].tick_params(axis='x', rotation=90)  # Rotate x-axis labels for readability

# Adjust layout to prevent overlap and ensure both plots fit within the figure
plt.tight_layout()
plt.show()
plt.close()





print("Genre Differences:")
abs_difference = calc_genre_differences(counts_year1, counts_year2, year1, year2)
print(abs_difference)
sns.barplot(data = abs_difference.head(30), x = "Genre", y = "Difference")
name_plot("Absolute Difference", "Absolute Growth in Genres from 1959 to 1975 Top 30")

sns.barplot(data = abs_difference.tail(30), x = "Genre", y = "Difference")
name_plot("Absolute Difference", "Absolute Growth in Genres from 1959 to 1975 Bottom 30")


print("Relative Growth:")
relative_growth = calc_genre_growth(counts_year1, counts_year2, year1, year2)
relative_growth['Percentage_Increase'] = relative_growth['Percentage_Increase'].replace([np.inf, -np.inf], np.nan)
relative_growth = relative_growth.dropna(subset=['Percentage_Increase'])
print(relative_growth)
sns.barplot(data = relative_growth.head(30), x = "Genre", y = "Percentage_Increase")
name_plot("Relative Increase", "Relative Growth in Genres from 1959 to 1975")

'''
sns.barplot(data = relative_growth.tail(30), x = "Genre", y = "Percentage_Increase")
plt.xticks(rotation=90)
plt.ylabel("Relative Increase")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.suptitle("Relative Growth in Genres from 1959 to 1975", fontsize = 14)
plt.show()
'''

proportion_growth = calc_relative_growth(counts_year1, counts_year2, year1, year2, total_movies_year1, total_movies_year2)
print("Proportion Growth:")
print(proportion_growth)
sns.barplot(data = proportion_growth.head(30), x = "Genre", y = "Proportion_Difference")
name_plot("Proportion Increase", "Change in the Propotion of Genres from 1959 to 1975 Top 30")

sns.barplot(data = proportion_growth.tail(30), x = "Genre", y = "Proportion_Difference")
name_plot("Proportion Increase", "Change in the Propotion of Genres from 1959 to 1975 Bottom 30")

war_films = count_genre_over_years(df, "War film")
print(top_years_for_genre(war_films, top_n=20))
sns.lineplot(data = war_films, x = "Year", y = "Count")
name_plot("Number of Movies Released", "War Films Throughout the Years")

'''
relative_growth = calc_genre_growth(counts_year1, counts_year2, year1, year2)

# Plot clusters with 5 clusters
plot_clusters_relative_change(relative_growth, n_clusters=5)

# Example usage to get genres in a specific cluster (e.g., Cluster 0)
genres_in_cluster2 = get_genres_in_cluster(relative_growth, 2)
genres_in_cluster3 = get_genres_in_cluster(relative_growth, 3)
print(f"Genres in Cluster")
print(genres_in_cluster2)
print(genres_in_cluster3)
'''