import pandas as pd

wars_path_base = "data/wars_filtered_clean.csv"

def clean_wars_name(wars):
    wars = [war.split(",")[0] for war in wars]
    return wars

def load_data_wars(wars_path):
    data = pd.read_table(wars_path, sep=",")
    data.set_index("WarName", inplace=True)
    return data

def find_sides (war_data):
    side_1 = [side for side in war_data[war_data["Side"]==1]["StateName"]]
    side_2 = [side for side in war_data[war_data["Side"]==2]["StateName"]]
    return side_1, side_2

def find_years(war_data):
    start_year = war_data["StartYear"].min()
    end_year = war_data["EndYear"].max()
    return start_year, end_year

def process_side (side):
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
    movies = movies[movies['Year'].between(start_year-range, end_year+range)]
    return movies

def find_movies_side(movies, side, mode="summary", threshold=10):

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
    movies_war = movies[movies["summary"].apply(lambda x: war in x)]
    if len(movies_war)<threshold:
        return None
    return movies_war