import pandas as pd
from utils_movie import load_data_movies
import warnings
warnings.filterwarnings("ignore")

# Load the data
df = pd.read_csv('data/wars.csv')

# Get StartYear and EndYear 
df["StartYear"] = df[["StartYear1", "StartYear2"]].max(axis=1)
df["EndYear"] = df[["EndYear1", "EndYear2"]].max(axis=1)

# Discard unnecessary columns
df = df[["WarName", "StateName", "Side", "StartYear", "EndYear", "WhereFought", "Outcome", "BatDeath"]]

# Filter out wars that started before 1895
df = df[df["EndYear"]>1894]

# Add additional data
df.loc[0,:] = ["Gaza-Israel conflict","Israel", 1, 2006, 2024, 6, 0, 1700]
df.loc[1,:] = ["Gaza-Israel conflict","Palestine", 2, 2006, 2024, 6, 0, 44500]
df.loc[2,:] = ["Russo-Georgian War", "Russia", 1, 2008, 2008, 12, 0, 70]
df.loc[3,:] = ["Russo-Georgian War", "South Ossetia", 1, 2008, 2008, 12, 0, 70]
df.loc[3,:] = ["Russo-Georgian War", "South Ossetia", 1, 2008, 2008, 12, 0, 365]
df.loc[4,:] = ["Russo-Georgian War", "Abkhazia", 1, 2008, 2008, 12, 0, 70]
df.loc[5,:] = ["Russo-Georgian War", "Georgia", 2, 2008, 2008, 12, 0, None]
df.loc[6,:] = ["First Libyan Civil War", 'NATO', 1, 2011, 2011, 4, 0, None]
df.loc[7,:] = ["First Libyan Civil War", 'Qatar', 1, 2011, 2011, 4, 0, None]
df.loc[8,:] = ["First Libyan Civil War", 'Sweden', 1, 2011, 2011, 4, 0, None]
df.loc[9,:] = ["First Libyan Civil War", 'United Arab Emirates', 1, 2011, 2011, 4, 0, None]
df.loc[10,:] = ["First Libyan Civil War", "Libya", 2, 2011, 2011, 4, 0, 3000]
df.loc[11,:] = ["Heglig Crisis", "Sudan", 1, 2012, 2012, 4, 0, 1200]
df.loc[12,:] = ["Heglig Crisis", "South Sudan", 2, 2012, 2012, 4, 0, 280]
df.loc[13, :] = ["Cold War", "United States of America", 1, 1947, 1991, 12, 0, 0]
df.loc[14, :] = ["Cold War", "Soviet Union", 2, 1947, 1991, 12, 0, 0]

# Set the index
df.set_index("WarName", inplace=True)

# Set correct data types
df["Side"] = df["Side"].astype(int)
df["StartYear"] = df["StartYear"].astype(int)
df["EndYear"] = df["EndYear"].astype(int)
df["WhereFought"] = df["WhereFought"].astype(int)
df["Outcome"] = df["Outcome"].astype(int)
df["BatDeath"] = df["BatDeath"].astype(float)

# Load the movies data
movies_path = "data/movies_with_summaries.csv"
movies = load_data_movies(movies_path)

# Filter wars based on same-time movies
counts = []
for war in df.index:
    start_year = df.loc[war, "StartYear"].min()
    end_year = df.loc[war, "EndYear"].max()
    movies_war = movies[movies['Year'].between(start_year-2, end_year+2)]
    counts.append(len(movies_war))
df["Movies"] = counts

# Filter out wars with less than 500 movies
df=df[df["Movies"]>500]

# Save the data
df.to_csv("data/wars_filtered_clean.csv")