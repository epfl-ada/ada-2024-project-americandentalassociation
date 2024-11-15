import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
from dotenv import load_dotenv
import os

# Fetch data by defining a query and date range
def fetch_trend_data(query, date_from, date_to):
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
    try:
        response = requests.post(URL, auth=(USERNAME, PASSWORD), json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        return json.loads(data["results"][0]["content"])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

# Query full historical interest of a given keyword
def query_full_interest(query):
    # Fetch data
    result = fetch_trend_data(query, '2004-01-01', '2024-10-01')

    # Convert to DataFrame and structurize
    df = pd.DataFrame(result["interest_over_time"][0]["items"])
    df["keyword"] = result["interest_over_time"][0]["keyword"]
    df["date"] = pd.to_datetime(df["time"], format="%b %Y")

    # Convert interest value to [0,1] scale
    df["value"] = df["value"] / 100

    return df[["date", "value", "keyword"]]

# Query historical interest of a given keyword around a 10-day interval of a given date
def query_interest_around_date(query, date):
    # Define the date range
    date_from = pd.to_datetime(date) - pd.Timedelta(days=5)
    date_to = pd.to_datetime(date) + pd.Timedelta(days=5)

    # Fetch data
    result = fetch_trend_data(query, date_from.strftime('%Y-%m-%d'), date_to.strftime('%Y-%m-%d'))

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

    # Convert to DataFrame and structurize
    df = pd.DataFrame(trend_data["breakdown_by_region"][0]["items"])
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
