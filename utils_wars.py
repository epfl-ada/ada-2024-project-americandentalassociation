import pandas as pd

wars_path_base = "data/wars_filtered_clean.csv"

def clean_wars_name(wars):
    """
    Clean the names of the wars.
    :param wars: the wars
    :return: the cleaned wars
    """
    wars = [war.split(",")[0] for war in wars]
    for i in range(len(wars)):
        if wars[i] == "Korean": wars[i] = "Korean War"
    return wars

def load_data_wars(wars_path):
    """
    Load the data from the wars dataset and preprocess it.
    :param wars_path: path to the wars dataset
    :return: the preprocessed data
    """
    data = pd.read_table(wars_path, sep=",")
    data.set_index("WarName", inplace=True)
    return data

def find_sides (war_data):
    """
    Find the sides of the war.
    :param war_data: the war data
    :return: the sides of the war
    """
    side_1 = [side for side in war_data[war_data["Side"]==1]["StateName"]]
    side_2 = [side for side in war_data[war_data["Side"]==2]["StateName"]]
    return side_1, side_2

def find_years(war_data):
    """
    Find the years of the war.
    :param war_data: the war data
    :return: the years of the war
    """
    start_year = war_data["StartYear"].min()
    end_year = war_data["EndYear"].max()
    return start_year, end_year

def process_side (side):
    """
    Process the side of the war.
    :param side: the side of the war
    :return: the processed side of the war
    """
    if "United States of America" in side:
        side.append("USA")
    if "United Kingdom" in side:
        side.append("UK")
        side.append("England")
    if "England" in side:
        side.append("UK")
        side.append("United Kingdom")
    if "Russia" in side:
        side.append("Soviet Union")
        side.append("USSR")
    if "Soviet Union" in side:
        side.append("Russia")
        side.append("USSR")
    if "USSR" in side:
        side.append("Russia")
        side.append("Soviet Union")
    if "Germany" in side:
        side.append("West Germany")
        side.append("East Germany")
    if "West Germany" in side:
        side.append("Germany")
    if "East Germany" in side:
        side.append("Germany")
    return side

def add_side_col(movies_war, side_1, side_2):
    """
    Add the side column to the movies dataset.
    :param movies_war: the movies dataset
    :param side_1: the first side of the war
    :param side_2: the second side of the war
    :return: the movies dataset with the side column
    """
    side_1 = process_side(side_1)
    side_2 = process_side(side_2)
    # 1 if side_1, 2 if side_2, 3 if both, 0 if neither
    side = []
    for c in movies_war["Countries"]:
        if c in side_1 and c in side_2:
            side.append("Changed sides")
        elif c in side_1:
            side.append("Side 1")
        elif c in side_2:
            side.append("Side 2")
        else:
            side.append("Neither side")
    movies_war["Side"] = side
    return movies_war      
    
def find_movies_year(movies, start_year, end_year, range=2):
    """
    Find the movies from a certain year range.
    :param movies: the movies dataset
    :param start_year: the start year
    :param end_year: the end year
    :param range: the range of years to consider
    :return: the movies from the year range
    """
    movies = movies[movies['Year'].between(start_year-range, end_year+range)]
    return movies

def find_movies_side(movies, side, mode="summary", threshold=10):
    """
    Find the movies from a certain side.
    :param movies: the movies dataset
    :param side: the side of the war
    :param mode: the mode to consider based on the column on which to filter
    :param threshold: the minimum number of movies to consider the search valid
    :return: the movies from the side
    """
    side = process_side(side)

    if mode == "summary":
        col = "summary"
    elif mode == "country":
        col = "Countries"
    else:
        return None
    
    movies_war = pd.DataFrame()
    for c in side:
        m = movies[movies[col].apply(lambda x: c in x)]
        movies_war = pd.concat([movies_war, m])
    movies = movies_war.drop_duplicates(subset="Title")

    if len(movies)<threshold:
        return None
    
    return movies

def find_movies_summary(movies, war, threshold=10):
    """
    Find the movies from a certain war.
    :param movies: the movies dataset
    :param war: the war
    :param threshold: the minimum number of movies to consider the search valid
    :return: the movies from the war
    """
    movies_war = movies[movies["summary"].apply(lambda x: war in x)]
    if len(movies_war)<threshold:
        return None
    return movies_war