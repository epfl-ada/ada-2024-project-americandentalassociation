# Utils file for Q1
import pandas as pd
from collections import Counter
import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter("ignore")
from scipy.signal import find_peaks
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from itertools import combinations
from collections import Counter
import networkx as nx
from scipy.stats import chi2_contingency
from PIL import Image
import os


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

'''
Cleaning function for genres that are almost the same and have slight differences, like the word 'Film' included, 
change all genre_2 to genre_1 for future analysis.

Parameters:
- df_movies: movie data
- genre_1: genre to keep
- genre_2: similar genre to change for the first one
'''
def clean_genres(df_movies, genre_1, genre_2):
    
    df_movies['Genres'] = df_movies['Genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df_movies['Genres'] = df_movies['Genres'].apply(lambda genres: list(set([genre if genre != genre_2 else genre_1 for genre in genres])))
    return df_movies

def analyze_genre_wars(df_movies, df_wars, genre_name):
    '''
    Analyze the top 10 peak years for a given genre and find the relevant wars
    that occurred during +-2 years of those peak years.

    Parameters:
    - df_movies: DataFrame containing movie data 
    - df_wars: DataFrame containing war data 
    - genre_name: The genre of films to analyze 
    '''
    # Count genre over years
    genre_films = count_genre_over_years(df_movies, genre_name)
    # Count genre proportion over years
    genre_films_proportion = count_genre_proportion(df_movies, genre_name)

    # Find the peaks in the proportion vs. time figure
    peak_indices, _ = find_peaks(genre_films_proportion["Proportion"])
    # Extract the peak years and corresponding values
    peak_years = genre_films_proportion.iloc[peak_indices]["Year"]
    peak_values = genre_films_proportion.iloc[peak_indices]["Proportion"]


    peaks_df = pd.DataFrame({
        'Year': peak_years,
        'Proportion': peak_values
    })
    # Select top 10 peaks
    top_peaks = peaks_df.sort_values(by="Proportion", ascending=False).head(10)
    # Extract top 10 peak years
    top_peak_years = top_peaks['Year'].values

    # Define the +-2 time interval
    time_intervals = []
    for year in top_peak_years:
        start_year = year - 2
        end_year = year + 2
        time_intervals.append((start_year, end_year))

    # Find the wars that happened during those time intervals
    wars_in_intervals = []
    # Use a set to store unique war names for further intersection
    unique_wars = set()  

    for start_year, end_year in time_intervals:
        # Filter wars for the time interval
        relevant_wars = df_wars[
            (((start_year <= df_wars['StartYear']) & (df_wars['StartYear'] <= end_year)) |
             ((start_year <= df_wars['EndYear']) & (df_wars['EndYear'] <= end_year)) |
             ((df_wars['StartYear'] < start_year) & (df_wars["EndYear"] > end_year)))
        ]

        for _, war in relevant_wars.iterrows():
            if war['WarName'] not in unique_wars:
                unique_wars.add(war['WarName'])
                
                # Find the closest peak year that fits in the interval
                peak_year_for_war = None
                for peak_year in top_peak_years:
                    if peak_year - 2 <= war['StartYear'] <= peak_year + 2 or peak_year - 2 <= war['EndYear'] <= peak_year + 2 or\
                    ((war['StartYear'] < peak_year - 2) & (war['EndYear'] > peak_year + 2)):
                        peak_year_for_war = peak_year
                        break
 
                wars_in_intervals.append((war['StartYear'], war['EndYear'], war['WarName'], peak_year_for_war))

    # Results
    wars_in_intervals_df = pd.DataFrame(wars_in_intervals, columns=["Start Year", "End Year", "War Name", "Peak Year"])

    return genre_films_proportion['Year'], genre_films_proportion['Proportion'], wars_in_intervals_df

'''
Function that calculates the number of ongoing wars for each year
'''
def wars_per_year(df_wars):
    '''
    Calculate how many wars were ongoing each year.
    '''
    # Unifying the 2 rows for each war into one
    df_agg = df_wars.groupby(['WarName', 'StartYear', 'EndYear'], as_index=False)['BatDeath'].sum()
    min_year = df_agg['StartYear'].min()
    max_year = df_agg['EndYear'].max()

    all_years = range(min_year, max_year + 1)
    ongoing_counts = []
    
    for year in all_years:
        count = ((df_agg['StartYear'] <= year) & (df_agg['EndYear'] >= year)).sum()
        ongoing_counts.append(count)

    ongoing_df = pd.DataFrame({
        'Year': all_years,
        'Count': ongoing_counts
    })

    return ongoing_df

def inspect_peaks_events(df_movies, df_wars, genre_list, ylim, name, filename):
    '''
    For identified lists of genres sum up the releases of all genres for each year and plot vs. year. Plot the number of ongoing wars
    to make observations. Find the events for the peaks.
    
    Parameters:
    - genre_list: list of genres to analyze
    - ylim: y-axis limit for proportions
    - name: the name of the inspected list
    '''
    fig = go.Figure()

    peak_wars = []
    combined_data = pd.DataFrame()

    # Use the analyze_genre_wars function for all genres in the list and combine the data
    for genre in genre_list:
        genre_data = analyze_genre_wars(df_movies, df_wars, genre)  # Retrieve data
        
        # Combine data into a single DataFrame: Years as index, proportions as values
        temp_df = pd.DataFrame({'Year': genre_data[0], 'Proportion': genre_data[1]})
        if combined_data.empty:
            combined_data = temp_df
        else:
            combined_data = combined_data.merge(temp_df, on="Year", how="outer", suffixes=("", f"_{genre}"))

        peak_wars.append(genre_data[2][["War Name", "Peak Year"]])

    # Sum proportions across all genres for each year and replace Null values with 0
    combined_data = combined_data.fillna(0) 
    combined_data['Total Proportion'] = combined_data.drop(columns='Year').sum(axis=1)

    # Plot the summed proportions as a single line
    fig.add_trace(go.Scatter(
        x=combined_data['Year'], 
        y=combined_data['Total Proportion'], 
        mode='lines', 
        name=f"Total Proportion ({name})",
        line=dict(width=3, color='blue')
    ))

    # Calculate intersection of "War Names" across all genres in the list
    intersection = set(peak_wars[0]['War Name'])
    for genre_data in peak_wars[1:]:
        intersection = intersection.intersection(set(genre_data['War Name']))

    final_df = pd.DataFrame(list(intersection), columns=["War Name"])

    # Update the layout
    fig.update_layout(
        title=f"{name} Production Trends",
        xaxis=dict(title="Year", gridcolor='lightgray'),
        yaxis=dict(title=f"Proportion", color='blue', range=[0, ylim]),
        legend=dict(
            x=1,
            y=1,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor="Black",
            borderwidth=1
        ),
        template="plotly_white",
        width=600,
        height=500
    )

    # fig.show()
    # fig.write_html(f"images/{filename}.html")
    fig.write_image("pic1.png", engine="orca", width=1600, height=800, scale=6)    
    img = Image.open("pic1.png")
    plt.figure(figsize=(16, 8))    
    plt.imshow(img)
    plt.axis('off')    
    plt.show()
    os.remove("pic1.png")
    # Print the intersected wars for inspection
    print(final_df)
    return final_df

'''
Calculate total battle death in a year for all years
'''
def calc_bat_deaths(df_wars):
    # Adding duration column. For a war we consider that the Battle Death for the war was evenly distributed across the years of the war
    df_aggregated = df_wars.groupby(['WarName', 'StartYear', 'EndYear'], as_index=False)['BatDeath'].sum()
    df_aggregated['Duration'] = df_aggregated['EndYear'] - df_aggregated['StartYear'] + 1

    expanded_df = df_aggregated.loc[df_aggregated.index.repeat(df_aggregated['Duration'])].copy()
    expanded_df['Year'] = expanded_df.groupby(level=0).cumcount() + expanded_df['StartYear']
    expanded_df['Count'] = expanded_df['BatDeath'] / expanded_df['Duration']
    yearly_deaths = expanded_df.groupby('Year')['Count'].sum().reset_index()

    min_year = df_aggregated['StartYear'].min()
    max_year = df_aggregated['EndYear'].max()
    all_years = pd.DataFrame({'Year': range(min_year, max_year + 1)})

    final_df = pd.merge(all_years, yearly_deaths, on='Year', how='left').fillna(0)
    return final_df


def correlation_genre_cut(genredf, otherdf, year_cut1, year_cut2):
    '''
    Filter both DataFrames to the given year range and calculate Pearson correlation.
    '''
    otherdf_filtered = otherdf[(otherdf["Year"] >= year_cut1) & (otherdf["Year"] <= year_cut2)]
    genredf_filtered = genredf[(genredf["Year"] >= year_cut1) & (genredf["Year"] <= year_cut2)]

    if len(genredf_filtered) < 2 or len(otherdf_filtered) < 2:
        print("Not enough data points for correlation in given year range.")
        return np.nan, np.nan

    return stats.pearsonr(genredf_filtered["Proportion"], otherdf_filtered["Count"])

def plot_proportion_vs_number_wars(genre_df, wars_df, genre_name, year_cut1, year_cut2, filename):
    '''
    Plots the proportion of a genre's movies vs. the number of wars and the OLS.
    Filters data by year_cut1 and year_cut2 and calculates Pearson correlation.
    '''
    # Filter by year range the passed genre_df and wars_df and merge together
    genre_filtered = genre_df[(genre_df["Year"] >= year_cut1) & (genre_df["Year"] <= year_cut2)][["Year", "Proportion"]]
    wars_filtered = wars_df[(wars_df["Year"] >= year_cut1) & (wars_df["Year"] <= year_cut2)][["Year", "Count"]]

    merged_df = pd.merge(genre_filtered, wars_filtered, on="Year", how="inner")
    
    if len(merged_df) < 2:
        print(f"Not enough data to plot correlation for {genre_name} in {year_cut1}-{year_cut2}.")
        return

    correlation, p_value = correlation_genre_cut(genre_df, wars_df, year_cut1, year_cut2)

    fig = px.scatter(
        merged_df,
        x="Count",
        y="Proportion",
        trendline="ols",
        labels={"Count": "Number of Wars", "Proportion": f"Proportion of {genre_name} Movies"},
        title=f"{genre_name} Movies and Ongoing Wars Correlation ({year_cut1}-{year_cut2})<br>"
              f"<sup>Pearson Correlation: {correlation:.2f}, p={p_value:.5f}</sup>"
    )

    fig.update_traces(marker=dict(size=8, opacity=0.7, color="blue"))
    fig.update_layout(
        xaxis_title="Number of Wars",
        yaxis_title=f"Proportion of {genre_name} Movies",
        template="plotly_white",
        width=700,
        height=500
    )

    # fig.show()
    # fig.write_html(f"images/{filename}.html")
    fig.write_image("pic2.png", engine="orca", width=1600, height=800, scale=6)    
    img = Image.open("pic2.png")
    plt.figure(figsize=(16, 8))    
    plt.imshow(img)
    plt.axis('off')    
    plt.show()
    os.remove("pic2.png")

def sum_genres_proportions(df_movies, genres):
    '''
    Sums up the yearly proportions of multiple genres in order to calculate the correlation for a group of genres
    '''
    combined_df = pd.DataFrame()

    for genre in genres:
        genre_yearly = count_genre_proportion(df_movies, genre)
        genre_yearly = genre_yearly[["Year", "Proportion"]].rename(columns={"Proportion": f"Proportion_{genre}"})

        if combined_df.empty:
            combined_df = genre_yearly
        else:
            combined_df = combined_df.merge(genre_yearly, on="Year", how="outer")
    if combined_df.empty:
        print(f"Error: No data available for genres {genres}. Returning None.")
        return None

    combined_df = combined_df.fillna(0)
    proportion_cols = [col for col in combined_df.columns if col.startswith("Proportion_")]
    combined_df["Proportion"] = combined_df[proportion_cols].sum(axis=1)
    
    return combined_df[["Year", "Proportion"]]


def plot_proportion_vs_battle_deaths(genre_df, bat_deaths_df, genre_name, year_cut1, year_cut2, filename):
    """
    Plots the relationship between a genre's proportion of movies and battle deaths and OLS.
    Filters data by the specified year range, calculates Pearson correlation.
    """
    # Filter by year range
    genre_filtered = genre_df[(genre_df["Year"] >= year_cut1) & (genre_df["Year"] <= year_cut2)][["Year", "Proportion"]]
    deaths_filtered = bat_deaths_df[(bat_deaths_df["Year"] >= year_cut1) & (bat_deaths_df["Year"] <= year_cut2)][["Year", "Count"]]

    merged_df = pd.merge(genre_filtered, deaths_filtered, on="Year", how="inner")

    if len(merged_df) < 2:
        print(f"Not enough data to plot correlation for {genre_name} in {year_cut1}-{year_cut2}.")
        return

    correlation, p_value = correlation_genre_cut(genre_df, bat_deaths_df, year_cut1, year_cut2)

    fig = px.scatter(
        merged_df,
        x="Count",
        y="Proportion",
        trendline="ols",
        labels={"Count": "Battle Deaths", "Proportion": f"Proportion of {genre_name} Movies"},
        title=f"{genre_name} Movies and Battle Deaths Correlation ({year_cut1}-{year_cut2})<br>"
              f"<sup>Pearson Correlation: {correlation:.2f}, p={p_value:.5f}</sup>"
    )

    fig.update_traces(marker=dict(size=8, opacity=0.7, color="blue"))
    fig.update_layout(
        xaxis_title="Battle Deaths",
        yaxis_title=f"Proportion of {genre_name} Movies",
        template="plotly_white",
        width=700,
        height=500
    )

    # fig.show()
    # fig.write_html(f"images/{filename}.html")
    fig.write_image("pic3.png", engine="orca", width=1600, height=800, scale=6)    
    img = Image.open("pic3.png")
    plt.figure(figsize=(16, 8))    
    plt.imshow(img)
    plt.axis('off')    
    plt.show()
    os.remove("pic3.png")

def genre_co_occurrence_network_plotly(df_movies, genres, min_weight):
    '''
    Build and visualize a simplified co-occurrence network of genres using Plotly.
    
    Parameters:
    - df_movies: DataFrame containing movie data.
    - genres: List of genres to include in the network.
    - min_weight: Minimum co-occurrence count to include an edge.
    '''
    # Extract genre pairs from each movie
    genre_combinations = df_movies["Genres"].apply(lambda x: list(combinations(x, 2)))
    co_occurrence = Counter([pair for sublist in genre_combinations for pair in sublist])

    # Convert to a DataFrame
    co_occurrence_df = pd.DataFrame(co_occurrence.items(), columns=["Pair", "Count"])
    co_occurrence_df["Genre1"] = co_occurrence_df["Pair"].apply(lambda x: x[0])
    co_occurrence_df["Genre2"] = co_occurrence_df["Pair"].apply(lambda x: x[1])

    # Filter by weight and chosen genres
    filtered_df = co_occurrence_df[
        ((co_occurrence_df["Genre1"].isin(genres)) | (co_occurrence_df["Genre2"].isin(genres))) &
        (co_occurrence_df["Count"] >= min_weight)
    ]

    # Build a NetworkX graph
    G = nx.Graph()
    for _, row in filtered_df.iterrows():
        G.add_edge(row["Genre1"], row["Genre2"], weight=row["Count"])

    # Get positions for Plotly (spring layout)
    pos = nx.spring_layout(G, seed=42)

    # Prepare edges for Plotly
    edge_x = []
    edge_y = []
    edge_weights = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append([x0, x1, None])  # None for line breaks
        edge_y.append([y0, y1, None])
        edge_weights.append(edge[2]["weight"])

    edge_trace = go.Scatter(
        x=[x for segment in edge_x for x in segment],  # Flatten x-coordinates
        y=[y for segment in edge_y for y in segment],  # Flatten y-coordinates
        line=dict(width=0.5, color="gray"),
        hoverinfo="none",
        mode="lines"
    )

    # Prepare nodes for Plotly
    node_x = []
    node_y = []
    node_labels = []
    node_sizes = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_labels.append(node)
        node_sizes.append(G.degree(node))  # Size proportional to the degree

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(
            size=[size * 10 for size in node_sizes],  # Scale node size
            color="skyblue",
            opacity=0.8,
            line=dict(width=1, color="black")
        ),
        text=node_labels,
        textposition="top center",
        hoverinfo="text"
    )

    # Create the Plotly figure
    fig = go.Figure(data=[edge_trace, node_trace])

    fig.update_layout(
        title=f"Genre Co-Occurrence Network (Weight >= {min_weight})",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_white",
        width=800,
        height=800
    )

    # fig.show()
    # fig.write_html("images/network.html")
    fig.write_image("pic4.png", engine="orca", width=1600, height=800, scale=6)    
    img = Image.open("pic4.png")
    plt.figure(figsize=(16, 8))    
    plt.imshow(img)
    plt.axis('off')    
    plt.show()
    os.remove("pic4.png")


def genre_shifts_analysis(df_movies, war_start_year, war_end_year, name, filename, window=5):
    '''
    Analyzes genre shifts before, during, and after a war using chi-square test.
    '''
    # Time window for "before" and "after"
    before_period = df_movies[(df_movies["Year"] >= war_start_year - window) & (df_movies["Year"] < war_start_year)]
    during_period = df_movies[(df_movies["Year"] >= war_start_year) & (df_movies["Year"] <= war_end_year)]
    after_period = df_movies[(df_movies["Year"] > war_end_year) & (df_movies["Year"] <= war_end_year + window)]

    # Count genres in each period
    before = before_period["Genres"].explode().value_counts()
    during = during_period["Genres"].explode().value_counts()
    after = after_period["Genres"].explode().value_counts()

    # Contingency table
    contingency_table = pd.concat([before, during, after], axis=1, keys=["Before", "During", "After"]).fillna(0)
    # Chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    # Calculate residuals
    residuals = (contingency_table - expected) / np.sqrt(expected)
    # Find genres with the largest absolute residuals 
    top_residual_genres = residuals.abs().max(axis=1).sort_values(ascending=False).head(10).index
    # Filter the contingency table to include only top genres
    filtered_table = contingency_table.loc[top_residual_genres]
    # Normalize for proportions
    filtered_proportions = filtered_table.div(filtered_table.sum(axis=0), axis=1)

    # Plot filtered proportions
    fig = go.Figure()

    # Add traces for each time period (Before, During, After)
    for time_period in filtered_proportions.columns:
        fig.add_trace(go.Bar(
            x=filtered_proportions.index,
            y=filtered_proportions[time_period],
            name=time_period
        ))

    # Layout
    fig.update_layout(
        barmode='stack', 
        title=f"Significant Genre Changes: {name} War",
        xaxis=dict(title="Genres"),
        yaxis=dict(title="Proportion"),
        legend_title="Time Periods",
        template="plotly_white",
        width=600,
        height=500
    )
    # fig.show()
    # fig.write_html(f"images/{filename}.html")
    fig.write_image("pic5.png", engine="orca", width=1600, height=800, scale=6)    
    img = Image.open("pic5.png")
    plt.figure(figsize=(16, 8))    
    plt.imshow(img)
    plt.axis('off')    
    plt.show()
    os.remove("pic5.png")



def war_analysis(df_movies, init_year, fin_year, war_name):
    '''
    Analyzes movie genres' proportions during and around (+-2) a specific war and plots.
    
    Parameters:
    - df_movies 
    - init_year: Starting year for the plot (war start year - 2).
    - fin_year: Ending year for the plot (war end year + 2).
    - war_name: name of the war
    '''
    # Identified the genres to inspect, grouped the genres into groups of similar theme for further analysis
    military_genres = ["War film", "Spy", "Superhero"]
    military_and_antiwar = ["War film", "Spy", "Superhero", "Antiwar"]
    reflecting_genres = ["Political satire", "Political thriller", "Political cinema", "Political drama"]
    dystopian_genres = ["Dystopia", "Apocalyptic and postapocalyptic fiction"]
    positive_genres = ["Family Film", "Romance Film", "Comedy", "Romantic comedy", "Fantasy"]

    def stacked_bar_plot_by_genres(df_movies, genre_list, init_year, fin_year, title, filename):
        '''
        Creates an interactive stacked bar plot using Plotly for specified genres over a year range.

        Parameters:
        - df_movies
        - genre_list: group of genres 
        - init_year
        - fin_year
        - title: Title of the plot.
        '''

        stacked_data = pd.DataFrame()
        
        for genre in genre_list:
            # Use count_genre_proportion to calculate proportions for the genre over time
            films_genre = count_genre_proportion(df_movies, genre)
            # Filter by the year range
            films_genre = films_genre[(films_genre["Year"] >= init_year) & (films_genre["Year"] <= fin_year)]
            # Add to stacked_data 
            stacked_data[genre] = films_genre.set_index("Year")["Proportion"]
        
        # Fill null values with 0 
        stacked_data = stacked_data.fillna(0)

        # Ensure all years are present
        all_years = pd.Series(range(init_year, fin_year + 1), name="Year")
        stacked_data = stacked_data.reindex(all_years, fill_value=0).reset_index()
        stacked_data.rename(columns={"index": "Year"}, inplace=True)

        # Create Plotly stacked bar plot
        fig = go.Figure()

        for genre in genre_list:
            fig.add_trace(go.Bar(
                x=stacked_data['Year'],
                y=stacked_data[genre],
                name=genre
            ))

        # Update layout
        fig.update_layout(
            barmode='stack',
        title=title,
        xaxis=dict(title="Year"),
        yaxis=dict(title="Proportion"),
        legend=dict(
        orientation="h",  # Horizontal orientation
        yanchor="bottom",  # Anchor to the bottom
        y=-0.3,            # Move the legend further down
        xanchor="center",  # Center horizontally
        x=0.5              # Center the legend horizontally
        ),
        template="plotly_white",
        margin=dict(
            l=40,  # Left margin
            r=40,  # Right margin
            t=80,  # Top margin
            b=100  # Bottom margin increased to give space for legend
        ),
        width=600,  # Adjust width if needed
        height=500
        )

        # fig.show()
        # fig.write_html(f"images/{war_name}_{filename}.html")
        fig.write_image("pic10.png", engine="orca", width=1600, height=800, scale=6)    
        img = Image.open("pic10.png")
        plt.figure(figsize=(16, 8))    
        plt.imshow(img)
        plt.axis('off')    
        plt.show()
        os.remove("pic10.png")

    # Run the function for each group of genres
    stacked_bar_plot_by_genres(
        df_movies=df_movies,
        genre_list=reflecting_genres,
        init_year=init_year,
        fin_year=fin_year,
        title=f"{war_name}: Proportions of Political Genres",
        filename="barplot_political"
    )

    stacked_bar_plot_by_genres(
        df_movies=df_movies,
        genre_list=military_and_antiwar,
        init_year=init_year,
        fin_year=fin_year,
        title=f"{war_name}: Proportions of Military Genres",
        filename="barplot_militaryandantiwar"
    )

    stacked_bar_plot_by_genres(
        df_movies=df_movies,
        genre_list=dystopian_genres,
        init_year=init_year,
        fin_year=fin_year,
        title=f"{war_name}: Proportions of Dystopian Genres",
        filename="barplot_dystopian"
    )

    positive_genres = ["Family Film", "Romance Film", "Romantic comedy", "Comedy", "Fantasy"]
    stacked_bar_plot_by_genres(
        df_movies=df_movies,
        genre_list=positive_genres,
        init_year=init_year,
        fin_year=fin_year,
        title=f"{war_name}: Proportions of Positive Genres",
        filename = "barplot_positive"
    )

def pick_years(chosen_wars, i):
    '''
    Calculate init_year and fin_year for each chosen war
    
    Parameters:
    - chosen_wars: DataFrame containing war data with 'StartYear', 'EndYear', and 'WarName' for the chosen wars
    - i: Index of the chosen war in the DataFrame.
    '''
    startyear = chosen_wars.iloc[i]["StartYear"]
    endyear = chosen_wars.iloc[i]["EndYear"]
    war_name = chosen_wars.iloc[i]["WarName"]
    init_year = startyear - 2
    fin_year = endyear + 2
    return init_year, fin_year, war_name