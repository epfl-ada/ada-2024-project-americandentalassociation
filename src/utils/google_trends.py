import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
from dotenv import load_dotenv
import os
import time
from src.utils.genres import format_keyword

# Fetch data by defining a query and date range
def fetch_trend_data(query, date_from, date_to, max_retries=5, timeout=100):
    load_dotenv()

    # Retrieve credentials
    USERNAME = os.getenv("USERNAME")
    PASSWORD = os.getenv("PASSWORD")
    URL = "https://realtime.oxylabs.io/v1/queries"

    payload = {
        "source": "google_trends_explore",
        "query": query,
        "context": [
            {'key': 'date_from', 'value': date_from},
            {'key': 'date_to', 'value': date_to},
        ],
    }
    for attempt in range(max_retries):
        try:
            # Attempt the POST request
            response = requests.post(URL, auth=(USERNAME, PASSWORD), json=payload, timeout=timeout)
            response.raise_for_status()

            # Parse and return the desired result
            data = response.json()
            return json.loads(data["results"][0]["content"])
        except requests.exceptions.RequestException as e:
            # Log the error and retry if possible
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(2)
            else:
                print("Max retries reached. Abandoning request.")
                return None

# Query full historical interest of a given keyword
def query_full_interest(query):
    # Fetch data
    result = fetch_trend_data(query, '2004-01-01', '2024-10-01')

    if result is None:
        return None
    
    # Convert to DataFrame and structurize
    df = pd.DataFrame(result["interest_over_time"][0]["items"])
    df["keyword"] = result["interest_over_time"][0]["keyword"]
    df["date"] = pd.to_datetime(df["time"], format="%b %Y")

    # Convert interest value to [0,1] scale
    df["value"] = df["value"] / 100

    return df[["date", "value", "keyword"]]

# Query historical interest of a given keyword around a 20-day interval of a given date
def query_interest_around_date(query, date):
    # Define the date range
    date_from = pd.to_datetime(date) - pd.Timedelta(days=10)
    date_to = pd.to_datetime(date) + pd.Timedelta(days=10)

    # Fetch data
    result = fetch_trend_data(query, date_from.strftime('%Y-%m-%d'), date_to.strftime('%Y-%m-%d'))

    if result is None:
        return None

    # Convert to DataFrame and structurize
    df = pd.DataFrame(result["interest_over_time"][0]["items"])
    df["keyword"] = result["interest_over_time"][0]["keyword"]
    df["date"] = pd.to_datetime(df["time"], format="%b %d, %Y")

    # Convert interest value to [0,1] scale
    df["value"] = df["value"] / 100

    return df[["date", "value", "keyword"]]

# Query interest breakdown by a list of regions
def query_interest_by_region(query):

    # Fetch data
    result = fetch_trend_data(query, '2004-01-01', '2024-10-01')

    if result is None:
        return None

    # Convert to DataFrame and structurize
    df = pd.DataFrame(result["breakdown_by_region"][0]["items"])
    df["keyword"] = result["breakdown_by_region"][0]["keyword"]
    df["date"] = pd.to_datetime(df["time"], format="%b %Y")

    # Convert interest value to [0,1] scale
    df["value"] = df["value"] / 100

    return df[["date", "value", "keyword"]]

# Plot historical changes in interest
def plot_interest_time_series(df, keyword, highlight_date=None, highlight_name=None):
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df, x='date', y='value', label=keyword)
    plt.title(f'Historical Interest for "{keyword}"')
    plt.xlabel('Date')
    plt.ylabel('Interest Level')
    plt.legend()
    
    # Highlight a specific date if provided
    if highlight_date:
        plt.axvline(pd.to_datetime(highlight_date), color='red', linestyle='--', linewidth=1.5,label=highlight_name)
        plt.legend()

    plt.show()

# Fetch historical interest for a list of genres
def fetch_historical_interest_for_genres(genres):
    # Define the genres to search for
    genres['Keyword'] = genres['Genre'].apply(format_keyword)

    # Construct the relative path to the data file
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    file_path = os.path.join(base_dir, 'data', 'genre_trends.csv')
    
    if pd.read_csv(file_path).empty:
        trends = []
        for _, genre in genres.iterrows():
            print(f"Fetching historical search interest for: {genre['Keyword']}")
            trend = query_full_interest(genre["Keyword"])
            trend["Genre"] = genre["Genre"]
            trends.append(trend)
        all_trends = pd.concat(trends)
        all_trends.to_csv('data/genre_trends.csv', index=False)
    
    all_trends = pd.read_csv(file_path)
    all_trends['date'] = pd.to_datetime(all_trends['date'])
    return all_trends

def get_events_trends_monthly(events, all_trends):
    # Filter events for the period 2004-2023
    events['date'] = pd.to_datetime(events['date'])
    events_filtered = events[(events['date'] >= '2004-01-01') & (events['date'] <= '2023-12-31')]
    events_filtered['Month'] = events_filtered['date'].dt.to_period('M').astype(str)

    # Filter trends for the period 2004-2023
    all_trends['date'] = pd.to_datetime(all_trends['date'])
    trends_filtered = all_trends[(all_trends['date'] >= '2004-01-01') & (all_trends['date'] <= '2023-12-31')]
    trends_filtered['Month'] = trends_filtered['date'].dt.to_period('M').astype(str)

    # Generate a range of months
    all_months = pd.date_range(start="2004-01-01", end="2023-12-31", freq='MS').strftime('%Y-%m').astype(str).tolist()

    # List all unique event types
    all_event_types = events_filtered['event_type'].unique()

    # Create all combinations
    all_combinations = pd.DataFrame([(month, event) for month in all_months for event in all_event_types], columns=['Month', 'event_type'])

    # Group actual data
    events_grouped = events_filtered.groupby(['Month', 'event_type']).size().reset_index(name='Count')

    # Merge with all combinations to include missing months and event types
    events_grouped = all_combinations.merge(events_grouped, on=['Month', 'event_type'], how='left')
    events_grouped['Count'] = events_grouped['Count'].fillna(0)

    # event_types = events_grouped['event_type'].unique()

    # trends_grouped = trends_filtered.groupby(['Month', 'Genre'])['value'].mean().reset_index()


    # Aggregate counts of events by type per month
    events_monthly = events_filtered.groupby(['Month', 'event_type']).size().reset_index(name='event_count')

    # Aggregate search interest by genre per month
    trends_monthly = trends_filtered.groupby(['Month', 'Genre'])['value'].sum().reset_index()

    return events_monthly, trends_monthly
    
