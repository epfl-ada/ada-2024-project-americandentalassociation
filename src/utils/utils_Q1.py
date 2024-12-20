# Utils file for Q1
import * from utils/genres.py

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
def wars_per_year():
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

def inspect_peaks_events(genre_list, ylim, name, filename):
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

    # Ongoing Wars. Calculate and add to the plot with a second y-axis
    # ongoing_wars_df = wars_per_year()
    # fig.add_trace(go.Scatter(
    #     x=ongoing_wars_df['Year'], 
    #     y=ongoing_wars_df['Count'],
    #     mode='lines+markers',
    #     name='Ongoing Wars',
    #     yaxis='y2',
    #     line=dict(color='purple', width=2, dash='dot'),
    #     marker=dict(size=5, color='purple')
    # ))

    # Update the layout
    fig.update_layout(
        title=f"{name} Production Trends",
        xaxis=dict(title="Year", gridcolor='lightgray'),
        yaxis=dict(title=f"Proportion", color='blue', range=[0, ylim]),
        # yaxis2=dict(
        #     title="Number of Ongoing Wars",
        #     overlaying='y',
        #     side='right',
        #     color='red',
        #     range=[0, 10]
        # ),
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

    fig.show()
    fig.write_html(f"images/{filename}.html")
    # Print the intersected wars for inspection
    print(final_df)
    return final_df