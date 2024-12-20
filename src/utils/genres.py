import pandas as pd
from collections import Counter
import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import pearsonr
import statsmodels.api as sm

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
    # Make sure Year is an integer
    dataframe['Year'] = dataframe['Year'].astype(int)

    # Ensure 'Genres' is a list of genres
    dataframe['Genres'] = dataframe['Genres'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # Filter to only rows containing the chosen genre
    genre_df = dataframe[dataframe['Genres'].apply(lambda g: genre in g)]
    
    # Group by year and count
    genre_counts_by_year = genre_df.groupby('Year').size().reset_index(name='Count')

    # Determine the full range of years in the dataset
    min_year = dataframe['Year'].min()
    max_year = dataframe['Year'].max()

    # Create a complete list of all years in the range
    all_years = pd.DataFrame({'Year': range(min_year, max_year + 1)})

    # Merge the complete years with the genre counts
    genre_counts_by_year = pd.merge(all_years, genre_counts_by_year, on='Year', how='left')

    # Fill missing values with 0
    genre_counts_by_year['Count'] = genre_counts_by_year['Count'].fillna(0)

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


# Function to plot top n genres
def get_top_genres(data, top_n=10):
    # Format the genres
    data['Genres'] = data['Genres_IMDb'].str.strip("[]").str.replace("'", "").str.split(",")
    all_genres = data['Genres'].dropna().apply(lambda x: x if isinstance(x, list) else x.split(','))
    genres_count = Counter([genre.strip() for genres in all_genres for genre in genres])

    # Sort the genres by count
    genres = pd.DataFrame.from_dict(genres_count, orient='index').reset_index()
    genres.columns = ['Genre', 'Count']
    genres = genres.sort_values(by='Count', ascending=False)

    # Filter by the top n genres
    genres = genres.head(top_n)

    return genres

# Function to format the search keyword for a given genre
def format_keyword(genre):
    if "film" in genre.lower():
        keyword = genre
    else:
        keyword = genre + " movies"

    return keyword.lower()

def seasonal_decomposition(all_trends):
    genres = all_trends['Genre'].unique()
    genre_decompositions = {}

    # Perform seasonal decomposition for each genre
    for genre in genres:
        genre_data = all_trends[all_trends['Genre'] == genre].set_index('date')['value']
        
        # Apply seasonal decomposition
        decomposition = seasonal_decompose(genre_data, model='additive', period=12)
        genre_decompositions[genre] = decomposition

    return genre_decompositions

def genres_vs_interest_correlation(movies, all_trends, genres):
    # Filter movies dataset for the period 2004-2012
    movies_filtered = movies[(movies['Year'] >= 2004)]
    movies_filtered = movies_filtered.explode('Genres')
    movies_filtered.rename(columns={'Genres': 'Genre'}, inplace=True)

    # Count the number of movies released per top genres per year
    genre_counts = movies_filtered.groupby(['Year', 'Genre']).size().reset_index()
    genre_counts.columns = ['Year', 'Genre', 'Count']

    # Filter only the top genres
    genre_counts = genre_counts[genre_counts['Genre'].isin(genres)]

    # Filter trends for the period 2004-2012
    trends_filtered = all_trends[(all_trends['date'] >= '2004-01-01') & (all_trends['date'] <= '2012-12-31')]

    # Group the trends by year and genre
    trends_filtered['Year'] = trends_filtered['date'].dt.year
    trends_grouped = trends_filtered.groupby(['Year', 'Genre'])['value'].mean().reset_index()

    # Merge movie counts with search interest data
    merged_data = pd.merge(genre_counts, trends_grouped, on=['Year', 'Genre'], how='inner')

    # Compute pearson correlation and p-value
    results = []
    for genre in merged_data['Genre'].unique():
        genre_data = merged_data[merged_data['Genre'] == genre]
        corr, p_value = pearsonr(genre_data['Count'], genre_data['value'])
        results.append({'Genre': genre, 'Correlation': corr, 'P-Value': p_value})

    results_df = pd.DataFrame(results)
    return results_df

def regression_analysis(events_monthly, trends_monthly):
    genres = trends_monthly['Genre'].unique()
    event_types = events_monthly['event_type'].unique()

    # Prepare to collect regression results for plotting
    regression_results = []

    # Set up the plot grid
    num_genres = len(genres)
    num_event_types = len(event_types)
    fig, axes = plt.subplots(num_genres, num_event_types, figsize=(15, num_genres * 4), sharex=False, sharey=False)
    axes = axes.flatten()

    plot_index = 0
    for genre in genres:
        genre_data = trends_monthly[trends_monthly['Genre'] == genre]
        
        for event_type in event_types:
            # Event type data
            event_data = events_monthly[events_monthly['event_type'] == event_type]
            
            # Merge monthly data
            merged_data = pd.merge(genre_data, event_data, on='Month', how='outer').fillna(0)
            
            # Regression Analysis
            X = merged_data['event_count']
            y = merged_data['value']
            X_with_const = sm.add_constant(X)
            
            model = sm.OLS(y, X_with_const).fit()
            slope = model.params['event_count']
            p_value = model.pvalues['event_count']
            r_squared = model.rsquared
            
            # Store results for further analysis
            regression_results.append([event_type, genre, slope, p_value, r_squared])

            # Only plot more significant correlations
            if p_value > 0.05:
                continue
            
            # Scatterplot
            ax = axes[plot_index]
            ax.scatter(X, y, alpha=0.7)
            
            # Plot regression line
            line_x = np.linspace(X.min(), X.max(), 100)
            line_y = slope * line_x + model.params['const']
            ax.plot(line_x, line_y, color='red', label=f"y = {slope:.3f}x + {model.params['const']:.3f}\nR²: {r_squared:.3f}\nP-value: {p_value:.3f}")
            
            ax.set_title(f"{event_type} on {genre} movies", fontsize=10)
            ax.set_xlabel("Event Count")
            ax.set_ylabel("Search Interest")
            ax.legend()
            
            plot_index += 1

    # Remove empty subplots
    for i in range(plot_index, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.suptitle("Impact of Events on Movie Genre Interest (Monthly, 2004-2023)", y=1.01, fontsize=16)
    plt.show()

    regression_df = pd.DataFrame(regression_results, columns=['event_type', 'genre', 'slope', 'p_value', 'r_squared'])
    return regression_df
    