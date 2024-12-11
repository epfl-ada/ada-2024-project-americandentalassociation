from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

from utils_movie import *
from utils_wars import *


def get_genres_year_side(wars, movies, war, mode="summary"):

    war_data = wars[wars.index==war]
    side_1, side_2 = find_sides(war_data)
    start_year, end_year = find_years(war_data)

    movies_war = find_movies_year(movies, start_year, end_year)
    movies_war_1 = find_movies_side(movies_war, side_1.copy(), mode=mode) 
    movies_war_2 = find_movies_side(movies_war, side_2.copy(), mode=mode)
    movies_side_1 = find_movies_side(movies, side_1.copy(), mode=mode)
    movies_side_2 = find_movies_side(movies, side_2.copy(), mode=mode)
    
    if movies_war_1 is None or movies_war_2 is None or movies_side_1 is None or movies_side_2 is None:
        return None, None, None, None, None

    if war == "Korean": war = "Korean War"
    if war == "Vietnam War, Phase 2": war = "Vietnam War"
    
    genres_war_1 = get_genres(movies_war_1)
    genres_war_2 = get_genres(movies_war_2)
    genres_side_1 = get_genres(movies_side_1)
    genres_side_2 = get_genres(movies_side_2)

    return war, genres_war_1, genres_war_2, genres_side_1, genres_side_2

def plot_genres_year_side(war, genres_war_1, genres_war_2, genres_side_1, genres_side_2):

    if war is None:
        print("Not Enough Data")
        return

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Side 1", "Side 2"])
    genres_1_r_war, genres_1_r_tot = get_total_genres(genres_war_1, genres_side_1, n=15)
    genres_2_r_war, genres_2_r_tot = get_total_genres(genres_war_2, genres_side_2, n=15)

    fig.add_trace(
        go.Bar(
            x=list(genres_1_r_war.keys()),
            y=list(get_normalized_values(genres_1_r_war)),
            name=f"Genres on side 1 during {war}",
            marker_color="indianred",
        ),
        row=1, col=1)
    fig.add_trace(
        go.Bar(
            x=list(genres_1_r_tot.keys()),
            y=list(get_normalized_values(genres_1_r_tot)),
            name=f"Genres on side 1 in total",
            marker_color="lightsalmon",
        ),
        row=1, col=1)

    fig.add_trace(
        go.Bar(
            x=list(genres_2_r_war.keys()),
            y=list(get_normalized_values(genres_2_r_war)),
            name=f"Genres on side 2 during {war}",
            marker_color="cornflowerblue",
        ),
        row=1, col=2)
    fig.add_trace(
        go.Bar(
            x=list(genres_2_r_tot.keys()),
            y=list(get_normalized_values(genres_2_r_tot)),
            name=f"Genres on side 2 in total",
            marker_color="lightblue",
        ),
        row=1, col=2)

    fig.update_xaxes(title_text="Genres", row=1, col=1)
    fig.update_xaxes(title_text="Genres", row=1, col=2)
    fig.update_layout(
        title=f"Genres during {war}",
        barmode="group",
        yaxis_title="Normalized count"
    )

    fig.show()

def plot_countries(side_1, side_2, war):
    countries = side_1 + side_2
    colors = ["indianred"] * len(side_1) + ["cornflowerblue"] * len(side_2)

    df = pd.DataFrame({"country": countries, "color": colors})

    fig = go.Figure()

    fig.add_trace(
        go.Choropleth(
            locations=side_1,  # List of country names
            locationmode="country names",  # Match by country names
            z=[1] * len(side_1),  # Dummy values for color
            colorscale=[[0, "indianred"], [1, "indianred"]],
            showscale=False,  # Hide color scale
            marker_line_color="black",  # Add border color
            name="Side 1"
        )
    )

    fig.add_trace(
        go.Choropleth(
            locations=side_2,  # List of country names
            locationmode="country names",
            z=[1] * len(side_2),  # Dummy values for color
            colorscale=[[0, "cornflowerblue"], [1, "cornflowerblue"]],
            showscale=False,
            marker_line_color="black",
            name="Side 2"
        )
    )

    fig.update_layout(
        title=f"Countries in Side 1 and Side 2 during {war}",
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type="equirectangular",
        ),
        legend=dict(
            yanchor="top",
            y=0.9,
            xanchor="left",
            x=1.05
        )
    )

    fig.show()

def get_genres_summary_side(wars, movies, war, mode="country"):
    war_data = wars[wars.index==war]
    side_1, side_2 = find_sides(war_data)

    if war == "Korean": war = "Korean War"
    if war == "Vietnam War, Phase 2": war = "Vietnam War"

    movies_war = find_movies_summary(movies, war)
    movies_war_1 = find_movies_side(movies_war, side_1.copy(), mode=mode, threshold=1)
    movies_war_2 = find_movies_side(movies_war, side_2.copy(), mode=mode, threshold=1)
    movies_side_1 = find_movies_side(movies, side_1.copy(), mode=mode, threshold=1)
    movies_side_2 = find_movies_side(movies, side_2.copy(), mode=mode, threshold=1)

    if movies_war_1 is None or movies_war_2 is None or movies_side_1 is None or movies_side_2 is None:
        return None, None, None, None, None
    
    genres_war_1 = get_genres(movies_war_1)
    genres_war_2 = get_genres(movies_war_2)
    genres_side_1 = get_genres(movies_side_1)
    genres_side_2 = get_genres(movies_side_2)

    return war, genres_war_1, genres_war_2, genres_side_1, genres_side_2

def plot_genres_summary_side(war, genres_war_1, genres_war_2, genres_side_1, genres_side_2):

    if war is None:
        print("Not Enough Data")
        return

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Side 1", "Side 2"])
    genres_1_r_war, genres_1_r_tot = get_total_genres(genres_war_1, genres_side_1, n=15)
    genres_2_r_war, genres_2_r_tot = get_total_genres(genres_war_2, genres_side_2, n=15)

    fig.add_trace(
        go.Bar(
            x=list(genres_1_r_war.keys()),
            y=list(get_normalized_values(genres_1_r_war)),
            name=f"Genres on side 1 on movies about {war}",
            marker_color="indianred",
        ),
        row=1, col=1)
    fig.add_trace(
        go.Bar(
            x=list(genres_1_r_tot.keys()),
            y=list(get_normalized_values(genres_1_r_tot)),
            name=f"Genres on side 1 in total",
            marker_color="lightsalmon",
        ),
        row=1, col=1)

    fig.add_trace(
        go.Bar(
            x=list(genres_2_r_war.keys()),
            y=list(get_normalized_values(genres_2_r_war)),
            name=f"Genres on side 2 on movies about {war}",
            marker_color="cornflowerblue",
        ),
        row=1, col=2)
    fig.add_trace(
        go.Bar(
            x=list(genres_2_r_tot.keys()),
            y=list(get_normalized_values(genres_2_r_tot)),
            name=f"Genres on side 2 in total",
            marker_color="lightblue",
        ),
        row=1, col=2)

    fig.update_xaxes(title_text="Genres", row=1, col=1)
    fig.update_xaxes(title_text="Genres", row=1, col=2)
    fig.update_layout(
        title=f"Genres for movies about {war}",
        barmode="group",
        yaxis_title="Normalized count"
    )

    fig.show()

def get_movies_year_side(wars, movies, war, mode="summary"):
    war_data = wars[wars.index==war]
    side_1, side_2 = find_sides(war_data)

    movies_war = find_movies_summary(movies, war, threshold=1)
    movies_war_1 = find_movies_side(movies_war, side_1.copy(), mode=mode)
    movies_war_2 = find_movies_side(movies_war, side_2.copy(), mode=mode)
    movies_side_1 = find_movies_side(movies, side_1.copy(), mode=mode)
    movies_side_2 = find_movies_side(movies, side_2.copy(), mode=mode)

    if movies_war_1 is None or movies_war_2 is None or movies_side_1 is None or movies_side_2 is None:
        return None, None, None, None, None

    return war, movies_war_1, movies_war_2, movies_side_1, movies_side_2

def plot_movies_timeline(wars, movies, war_list, war_line_flag=True):
    war_timelines = []
    for war in war_list:
        war_data = wars[wars.index == war]
        start_year, end_year = find_years(war_data)
        war_timelines.append((start_year, end_year))

    wars_large = clean_wars_name(war_list)

    # Initialize an empty DataFrame to store aggregated data
    data_aggregated = pd.DataFrame()

    # Iterate through the wars and prepare the aggregated data
    wars_copy = wars_large.copy()
    war_timelines_2 = []
    for i in range(len(wars_large)):
        war = wars_large[i]
        # print(f"Processing {war}...")
        war_data = find_movies_summary(movies, war)
        if war_data is None or war_data.empty:
            # print(f"No movies found for {war}. Removing from the list of wars.")
            wars_copy.remove(war)
            continue

        war_counts = war_data.groupby('Year').size().reset_index(name=war)
        # print(f"War: {war}, Number of Movies: {war_counts.shape[0]}")
        if war_counts.empty:
            wars_copy.remove(war)
            continue

        if data_aggregated.empty:
            data_aggregated = war_counts
            war_timelines_2.append(war_timelines[i])
        else:
            data_aggregated = pd.merge(data_aggregated, war_counts, on='Year', how='outer')
            war_timelines_2.append(war_timelines[i])
            # print(data_aggregated.columns)
        

    # Fill NaN values with 0 (for years with no movies for a specific war)
    data_aggregated = data_aggregated.fillna(0)
    data_aggregated = data_aggregated.sort_values('Year')

    data_aggregated['Total'] = data_aggregated.sum(axis=1)-data_aggregated['Year']
    data_max = data_aggregated['Total'].max()

    # Define a custom color sequence
    color_sequence = px.colors.qualitative.Plotly  # Or any other color sequence you like

    # Create a stacked area chart using Plotly
    fig = go.Figure()

    # Add traces for each war and their timelines
    for i, war in enumerate(wars_copy):
        # Explicitly set the color for the trace
        war_color = color_sequence[i % len(color_sequence)]
        
        # Add the stacked area trace
        fig.add_trace(go.Scatter(
            x=data_aggregated['Year'],
            y=data_aggregated[war],
            mode='lines',
            stackgroup='one',  # Stack traces
            name=war,
            line=dict(color=war_color)  # Set color explicitly
        ))
        
        # Add vertical lines for the start and end years of the war
        start_year, end_year = war_timelines_2[i]
        
        # Add a vertical line for the start year
        if war_line_flag:
            fig.add_shape(
                type='line',
                x0=start_year,
                x1=start_year,
                y0=0,
                y1=data_max,  # Set y1 to the max value for this war
                line=dict(color=war_color, dash='dash'),
                name=f"{war} Start"
            )
            
            # Add a vertical line for the end year
            fig.add_shape(
                type='line',
                x0=end_year,
                x1=end_year,
                y0=0,
                y1=data_max,  # Set y1 to the max value for this war
                line=dict(color=war_color, dash='dash'),
                name=f"{war} End"
            )

    # Customize the layout
    fig.update_layout(
        title='Number of Movies About Each War Produced Over Time',
        xaxis_title='Year',
        yaxis_title='Number of Movies',
        hovermode='x unified'
    )

    fig.show()

def plot_country_piechart(wars, movies, war):
    side_1, side_2 = find_sides(wars[wars.index==war])

    if war == "Korean": war = "Korean War"
    if war == "Vietnam War, Phase 2": war = "Vietnam War"
    movies_war = find_movies_summary(movies, war)
    if side_1 is None or side_2 is None or movies_war is None:
        return
    movies_exploded = movies_war.explode('Countries', ignore_index=True)
    movies_exploded = add_side_col(movies_exploded, side_1, side_2)
    movies_exploded["Count"] = np.ones(movies_exploded.shape[0])
    fig = px.sunburst(movies_exploded, path=['Side', 'Countries'], values="Count")
    fig.update_layout(
        title=f'Movies about {war} by Country',
        hovermode='x unified'
    )
    fig.show()