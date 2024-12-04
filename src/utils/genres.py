import pandas as pd
from collections import Counter
import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Plot naming and showing
def name_plot(ylabel, title):
    plt.xticks(rotation=90)
    plt.ylabel(ylabel)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(title)
    plt.show()
    plt.close()

# Function to count the number of movies for each genre over the years
def count_genre_over_years(dataframe, genre):
    dataframe['Genres'] = dataframe['Genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    genre_counts_by_year = (
        dataframe[dataframe['Genres'].apply(lambda genres: genre in genres)]
        .groupby('Year')
        .size()
        .reset_index(name='Count')
    )
    return genre_counts_by_year

def get_event_years(df, war_name):
    war_rows = df[df['WarName'].str.contains(war_name, case=False, na=False)]
    if not war_rows.empty:
        year1 = war_rows.iloc[0]["StartYear"]
        year3 = war_rows.iloc[0]["EndYear"]
        year0 = year1 - 2
        year2 = (year1 + year3) // 2
        year4 = year3 + 2
        return year0, year1, year2, year3, year4
    else:
        print(f"No data found for {war_name}")
        return None

def count_genre_proportion(df, genre):
    # Get the count of the specified genre over the years
    genre_count = count_genre_over_years(df, genre)
    
    # Get the total count of all genres for each year
    total_count = df.groupby('Year')['Genres'].apply(lambda x: len([genre for genres in x for genre in genres])).reset_index()
    total_count.columns = ['Year', 'Total']
    
    # Merge the genre count with the total count by year
    genre_count = pd.merge(genre_count, total_count, on='Year')
    
    # Calculate the proportion of War Films to total films for each year
    genre_count['Proportion'] = genre_count['Count'] / genre_count['Total']
    
    return genre_count

def top_years_for_genre(genre_counts_by_year, n=20):
    top_years = genre_counts_by_year.nlargest(n, 'Count')
    return top_years

def bottom_years_for_genre(genre_counts_by_year, n=20):
    top_years = genre_counts_by_year.nsmallest(n, 'Count')
    return top_years

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
def calc_proportion_growth(counts_year1, counts_year2, year1, year2, total_movies_year1, total_movies_year2):
    # Add proportions relative to the total number of movies for each year
    counts_year1['Proportion'] = counts_year1['Count'] / total_movies_year1
    counts_year2['Proportion'] = counts_year2['Count'] / total_movies_year2
    
    # Merge and keep the suffixes to distinguish each year's proportions
    merged_counts = pd.merge(counts_year1, counts_year2, on='Genre', how='outer', suffixes=(f'_{year1}', f'_{year2}')).fillna(0)
    merged_counts[f'Proportion_Difference'] = merged_counts[f'Proportion_{year2}'] - merged_counts[f'Proportion_{year1}']
    
    return merged_counts.sort_values(by='Proportion_Difference', ascending=False).reset_index(drop=True)