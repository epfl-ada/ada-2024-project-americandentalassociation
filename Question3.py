#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import plotly.express as px
from collections import Counter
import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.genres import *
from src.utils.genres import name_plot, count_genre_over_years, top_years_for_genre, bottom_years_for_genre, count_genres_by_year, calc_genre_differences, calc_genre_growth, calc_proportion_growth
import warnings
import csv
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
from data.Cleaning import final_df
warnings.simplefilter("ignore")
from ast import literal_eval
import plotly.graph_objects as go 


# While our previous analysis revealed distinct patterns in genre production across different countries, a deeper examination of narrative construction within the same genre warrants investigation. Specifically, this section explores how war films, despite sharing a common genre classification, exhibit significant variations in their storytelling approaches based on their country of origin.

# In[2]:


#dowload the movie dataset
df = pd.read_csv('data/movies_with_summaries.csv')
df.head()


# In[3]:


#dowload the war dataset
wars_df = pd.read_csv('data/wars_filtered_clean.csv')
wars_df.head()


# In[4]:


# Download necessary NLTK packages for text processing and sentiment analysis

nltk.download('punkt')  # Tokenizer for splitting text into words
nltk.download('averaged_perceptron_tagger')  # POS tagger for grammatical structure
nltk.download('maxent_ne_chunker')  # NER chunker for extracting named entities
nltk.download('words')  # Word corpus for NER
nltk.download('stopwords')  # Common words to exclude from analysis
nltk.download('wordnet')  # Lexical database for English
nltk.download('vader_lexicon')  # Lexicon for sentiment analysis

# Define the set of English stopwords to remove common words that don't add meaning
stop_words = set(stopwords.words("english"))


# Based on the previous analyses, I will reproduce the segregation of parties involved in the conflicts of interest

# In[5]:


#WWII
# WWII
wwii_bell = wars_df[wars_df['WarName'] == 'World War II'][['StateName', 'Side', 'Outcome']]
wwii_bell_side1 = wwii_bell[wwii_bell['Side'] == 1]['StateName'].values.tolist()
wwii_bell_side2 = wwii_bell[wwii_bell['Side'] == 2]['StateName'].values.tolist()

print("wwii -  Side 1 countries:", wwii_bell_side1)
print("wwii - Side 2 countries:", wwii_bell_side2)

#Cold War 
cold_bell = wars_df[wars_df['WarName']=='Cold War'][['StateName', 'Side', 'Outcome']]
cold_bell_side1 = cold_bell[cold_bell['Side'] == 1]['StateName'].values.tolist()
cold_bell_side2 = cold_bell[cold_bell['Side'] == 2]['StateName'].values.tolist()
soviet_equivalents = [
    'Soviet Union', 'USSR', 'Union of Soviet Socialist Republics', 
    'Russia', 'Russian Federation', 'Belarus', 'Ukraine', 'Kazakhstan',
    'Estonia', 'Latvia', 'Lithuania', 'Uzbekistan', 
    'Turkmenistan', 'Kyrgyzstan', 'Tajikistan',"Germany", "Albania",
    "East Germany","GDR","Bulgaria","Hungary","Poland","Romania",
    "Czechoslovakia","Yugoslavia"
]
cold_bell_side2.extend([country for country in soviet_equivalents if country not in cold_bell_side2])
print("Cold War - Side 1 countries:", cold_bell_side1)
print("Cold War - Side 2 countries:", cold_bell_side2)


#Korean
korean_bell = wars_df[wars_df['WarName']=='Korean'][['StateName', 'Side', 'Outcome']]
korean_bell_side1 = korean_bell[korean_bell['Side'] == 1]['StateName'].values.tolist()
korean_bell_side2 = korean_bell[korean_bell['Side'] == 2]['StateName'].values.tolist()
# print(korean_bell_side1)
print("Korean War - Side 1 countries:", korean_bell_side1)
print("Korean War - Side 2 countries:", korean_bell_side2)

#Vietnam War
viet_bell = wars_df[wars_df['WarName']=='Vietnam War, Phase 2'][['StateName', 'Side', 'Outcome']]
viet_bell_side1 = viet_bell[viet_bell['Side'] == 1]['StateName'].values.tolist()
viet_bell_side2 = viet_bell[viet_bell['Side'] == 2]['StateName'].values.tolist()
print(viet_bell_side1)
print("Vietnam War - Side 1 countries:", viet_bell_side1)
print("Vietnam War - Side 2 countries:", viet_bell_side2)


# Our analysis will focus exclusively on war cinema as it provides a unique window into how different nations process, interpret, and memorialize shared historical events through film. War movies are particularly revealing because they often reflect not only a country's historical perspective but also its contemporary values, national identity, and relationship with military conflict

# In[6]:


# Filter rows where 'Genres' contains "WarMovie"
df_war_movies = df[df['Genres'].str.contains("War film", case=False, na=False)].reset_index(drop=True)
df_war_movies.shape


# Building upon our colleague's comprehensive analysis of genre distribution across four major conflicts (World War II, the Korean War, the Vietnam War, and the Cold War), we will delve deeper into those conflicts. To ensure precise identification of films depicting these specific conflicts, we conducted a systematic keyword analysis of movie summaries, allowing us to isolate relevant war films from our dataset

# In[7]:


# Defining the function that detects specific keywords related to conflicts 
def label_event_regex(summary):
    if re.search(r"(World\sWar\sII|WWII|WW2|Hitler|Nazis|Hiroshima|Holocaust|D-Day|Axis|Allies|Pearl\sHarbor|Third\sReich|Blitzkrieg)", summary, re.IGNORECASE):
        return "World War II"
    elif re.search(r"(Vietnam\sWar|Viet\sCong|Saigon|Ho\sChi\sMinh|Tet\sOffensive|Agent\sOrange|Hanoi|Domino\sTheory)", summary, re.IGNORECASE):
        return "Vietnam War"
    elif re.search(r"(Cold\sWar|Soviet\sUnion|USSR|communism|nuclear|Iron\sCurtain|Berlin\sWall|Cuban\sMissile\sCrisis|Space\sRace|Reagan|Stalin|KGB|Eastern\sBloc|Gorbachev|Perestroika|Glasnost)", summary, re.IGNORECASE):
        return "Cold War"
    elif re.search(r"(Korean\sWar|Kim|Korea|Korean|NKPA|Demilitarized\sZone|Pyongyang|Seoul|Joint\sChiefs\sof\sStaff|Inchon|38th\sParallel|MacArthur)", summary, re.IGNORECASE):
        return "Korean"
    else:
        return "Other"

# Apply the function to the "summary" column 
df_war_movies['event'] = df_war_movies['summary'].apply(label_event_regex)

count_other = df_war_movies['event'].value_counts()['Other']
print(count_other)


# Our keyword-based filtering approach retained approximately half of the war films in our dataset, which appears to be a reasonable outcome. This proportion aligns with expectations, as many war films in our database likely focus on other historical conflicts, fictional wars, or explore warfare in a more general context rather than specifically addressing these four major conflicts.

# In[8]:


# Display movies labeled as "Cold War" for verification
df_war_movies[df_war_movies['event'] == "Cold War"].head()


# Having isolated war films specifically depicting these four conflicts and identified their respective sides of production, we can now analyze the distribution of films across opposing sides for each conflict.

# In[9]:


# World War II
wwii_movies = df_war_movies[
    (df_war_movies['event'] == "World War II") &
    (df_war_movies['Countries'].apply(lambda loc: any(country in str(loc) for country in (wwii_bell_side1 + wwii_bell_side2))))
]

wwii_movies_side1 = df_war_movies[
    (df_war_movies['event'] == "World War II") &
    (df_war_movies['Countries'].apply(lambda loc: any(country in str(loc) for country in wwii_bell_side1)))
]

wwii_movies_side2 = df_war_movies[
    (df_war_movies['event'] == "World War II") &
    (df_war_movies['Countries'].apply(lambda loc: any(country in str(loc) for country in wwii_bell_side2)))
]

#sample_wwii_movies_side1 = wwii_movies_side1.sample(n=64, random_state=42)

print("-" * 40)
print(f"Shape of wwii_movies: {wwii_movies.shape}")
print(f"Shape of wwii_movies_side1: {wwii_movies_side1.shape}")
print(f"Shape of wwii_movies_side2: {wwii_movies_side2.shape}")
#print(f"Shape of sample_wwii_movies_side1 for verification: {sample_wwii_movies_side1.shape}")
print("-" * 40)

# Cold War
cold_war_movies = df_war_movies[
    (df_war_movies['event'] == "Cold War") &
    (df_war_movies['Countries'].apply(lambda loc: any(country in str(loc) for country in (cold_bell_side1 + cold_bell_side2))))
]


cold_war_movies_side1 = df_war_movies[
    (df_war_movies['event'] == "Cold War") &
    (df_war_movies['Countries'].apply(lambda loc: any(country in str(loc) for country in cold_bell_side1)))
]

cold_war_movies_side2 = df_war_movies[
    (df_war_movies['event'] == "Cold War") &
    (df_war_movies['Countries'].apply(lambda loc: any(country in str(loc) for country in cold_bell_side2)))
]

#sample_cold_war_movies_side1 = cold_war_movies_side1.sample(n=12, random_state=42)

print("-" * 40)
print(f"Shape of cold_war_movies: {cold_war_movies.shape}")
print(f"Shape of cold_war_movies_side1: {cold_war_movies_side1.shape}")
print(f"Shape of cold_war_movies_side2: {cold_war_movies_side2.shape}")
#print(f"Shape of sample_cold_war_movies_side1 for verification: {sample_cold_war_movies_side1.shape}")
print("-" * 40)

# Vietnam War
vietnam_war_movies = df_war_movies[
    (df_war_movies['event'] == "Vietnam War") &
    (df_war_movies['Countries'].apply(lambda loc: any(country in str(loc) for country in (viet_bell_side1 + viet_bell_side2))))
]

vietnam_war_movies_side1 = df_war_movies[
    (df_war_movies['event'] == "Vietnam War") &
    (df_war_movies['Countries'].apply(lambda loc: any(country in str(loc) for country in viet_bell_side1)))
]

vietnam_war_movies_side2 = df_war_movies[
    (df_war_movies['event'] == "Vietnam War") &
    (df_war_movies['Countries'].apply(lambda loc: any(country in str(loc) for country in viet_bell_side2)))
]

print("-" * 40)
print(f"Shape of vietnam_war_movies: {vietnam_war_movies.shape}")
print(f"Shape of vietnam_war_movies_side1: {vietnam_war_movies_side1.shape}")
print(f"Shape of vietnam_war_movies_side2: {vietnam_war_movies_side2.shape}")
print("-" * 40)

# Korean War
korean_war_movies = df_war_movies[
    (df_war_movies['event'] == "Korean") &
    (df_war_movies['Countries'].apply(lambda loc: any(country in str(loc) for country in (korean_bell_side1 + korean_bell_side2))))
]

korean_war_movies_side1 = df_war_movies[
    (df_war_movies['event'] == "Korean") &
    (df_war_movies['Countries'].apply(lambda loc: any(country in str(loc) for country in korean_bell_side1)))
]

korean_war_movies_side2 = df_war_movies[
    (df_war_movies['event'] == "Korean") &
    (df_war_movies['Countries'].apply(lambda loc: any(country in str(loc) for country in korean_bell_side2)))
]

print("-" * 40)
print(f"Shape of korean_war_movies: {korean_war_movies.shape}")
print(f"Shape of korean_war_movies_side1: {korean_war_movies_side1.shape}")
print(f"Shape of korean_war_movies_side2: {korean_war_movies_side2.shape}")
print("-" * 40)


# In[ ]:





# In[10]:


import plotly.express as px
import pandas as pd

# Data for visualization
data = {
    'Wars': ['World War II', 'Vietnam War', 'Korean War', 'Cold War'],
    'Side1': [
        wwii_movies_side1.shape[0], 
        vietnam_war_movies_side1.shape[0], 
        korean_war_movies_side1.shape[0], 
        cold_war_movies_side1.shape[0]
    ],
    'Side2': [
        wwii_movies_side2.shape[0], 
        vietnam_war_movies_side2.shape[0], 
        korean_war_movies_side2.shape[0], 
        cold_war_movies_side2.shape[0]
    ]
}

df_visual = pd.DataFrame(data)

# Melt the dataframe for a grouped bar chart
df_melted = df_visual.melt(id_vars="Wars", var_name="Side", value_name="Number of Movies")

# Create the interactive bar chart with plotly express
fig = px.bar(
    df_melted, 
    x="Wars", 
    y="Number of Movies", 
    color="Side", 
    barmode="group",
    text="Number of Movies",
    title="Number of War Movies by Conflict and Side"
)

fig.update_layout(
    xaxis_title="Conflicts",
    yaxis_title="Number of Movies",
    title_font_size=16,
    xaxis_tickangle=45,
    legend_title="Side",
    font=dict(size=12),
    height=600
)


fig.update_traces(textposition="outside")
fig.write_html("number_of_war_movies.html")
fig.show()


# Given the limited number of movies produced about the Vietnam War and the Korean War, we have decided to focus exclusively on the Cold War and World War II.

# # Sentiment Analysis
# 
# We will perform a sentiment analysis on the summaries of movies identified. To achieve this, we will use the NLTK (Natural Language Toolkit) library. This analysis will help identify the emotional tone of the summaries, determining whether they are primarily positive, negative, or neutral depending of the countries of production.
# 

# In[11]:


# Extract the summaries from the sampled movies
synopses1=wwii_movies_side1["summary"].tolist()
synopses2=wwii_movies_side2["summary"].tolist()

synopses3=cold_war_movies_side1["summary"].tolist()
synopses4=cold_war_movies_side2["summary"].tolist()


# In[12]:


# Preprocess the summaries: tokenize and clean the text and Remove stopwords

#WORD WAR II
processed_docs_1 = []
processed_docs_2 = []
for synopsis in synopses1:
    tokens = word_tokenize(synopsis.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    processed_docs_1.append(filtered_tokens)

for synopsis in synopses2:
    tokens = word_tokenize(synopsis.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    processed_docs_2.append(filtered_tokens)

#COLD WAR II

processed_docs_3 = []
processed_docs_4 = []
for synopsis in synopses3:
    tokens = word_tokenize(synopsis.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    processed_docs_3.append(filtered_tokens)

for synopsis in synopses4:
    tokens = word_tokenize(synopsis.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    processed_docs_4.append(filtered_tokens)


# In[15]:


## Conduct the Sentiment Analysis

analyzer = SentimentIntensityAnalyzer()
sentiment_scores = []

####### WORD WAR II ##############

# Analyze the sentiment of each processed summary
for tokens in processed_docs_1:
    text = ' '.join(tokens)
    score = analyzer.polarity_scores(text)
    sentiment_scores.append(score)

sentiment_df_1 = pd.DataFrame(sentiment_scores)
test_sample_1 = wwii_movies_side1.reset_index(drop=True)
test_sample_1 = pd.concat([test_sample_1, sentiment_df_1], axis=1)


analyzer_2 = SentimentIntensityAnalyzer()
sentiment_scores_2= []

# Analyze the sentiment of each processed summary
for tokens in processed_docs_2:
    text = ' '.join(tokens)
    score = analyzer_2.polarity_scores(text)
    sentiment_scores_2.append(score)

sentiment_df_2 = pd.DataFrame(sentiment_scores_2)
test_sample_2 = wwii_movies_side2.reset_index(drop=True)
test_sample_2 = pd.concat([test_sample_2, sentiment_df_2], axis=1)


##########COLD WAR################

analyzer_3 = SentimentIntensityAnalyzer()
sentiment_scores_3= []

# Analyze the sentiment of each processed summary
for tokens in processed_docs_3:
    text = ' '.join(tokens)
    score = analyzer.polarity_scores(text)
    sentiment_scores_3.append(score)

sentiment_df_3 = pd.DataFrame(sentiment_scores_3)
test_sample_3 = cold_war_movies_side1.reset_index(drop=True)
test_sample_3 = pd.concat([test_sample_3, sentiment_df_3], axis=1)


analyzer_4 = SentimentIntensityAnalyzer()
sentiment_scores_4= []

# Analyze the sentiment of each processed summary
for tokens in processed_docs_4:
    text = ' '.join(tokens)
    score = analyzer_4.polarity_scores(text)
    sentiment_scores_4.append(score)

sentiment_df_4 = pd.DataFrame(sentiment_scores_4)
test_sample_4 = wwii_movies_side2.reset_index(drop=True)
test_sample_4 = pd.concat([test_sample_4, sentiment_df_4], axis=1)

# Display the summaries along with their sentiment scores
print(test_sample_1[['summary', 'neg', 'neu', 'pos', 'compound']])
print(test_sample_2[['summary', 'neg', 'neu', 'pos', 'compound']])
print(test_sample_3[['summary', 'neg', 'neu', 'pos', 'compound']])
print(test_sample_4[['summary', 'neg', 'neu', 'pos', 'compound']])



# In[17]:


# Mean and CI for each set 

mean_scores_1 = sentiment_df_1.mean()
mean_scores_2 = sentiment_df_2.mean()
mean_scores_3 = sentiment_df_3.mean()
mean_scores_4 = sentiment_df_4.mean()

from scipy import stats

def calculate_confidence_intervals(sentiment_df, confidence=0.95):
    results = []
    
    for column in ['neg', 'neu', 'pos', 'compound']:
        data = sentiment_df[column]
        mean = np.mean(data)
        std_err = stats.sem(data)
        ci = stats.t.interval(confidence, len(data)-1, mean, std_err)
        
        results.append({
            'mesure': column,
            'moyenne': mean,
            'ci_lower': ci[0],
            'ci_upper': ci[1]
        })
    
    return pd.DataFrame(results)

ci_wwii_side1 = calculate_confidence_intervals(sentiment_df_1)
ci_wwii_side2 = calculate_confidence_intervals(sentiment_df_2)
ci_cold_side1 = calculate_confidence_intervals(sentiment_df_3)
ci_cold_side2 = calculate_confidence_intervals(sentiment_df_4)


# In[29]:


#WWII VISUALISATION

# Visualisation 1 : mean compearison
labels = ['Negative', 'Neutral', 'Positive', 'Compound']
x = np.arange(len(labels))

#Confiance Intervall

ci_lower_1 = mean_scores_1 - ci_wwii_side1['ci_lower'].values
ci_upper_1 = ci_wwii_side1['ci_upper'].values - mean_scores_1

ci_lower_2 = mean_scores_2 - ci_wwii_side2['ci_lower'].values
ci_upper_2 = ci_wwii_side2['ci_upper'].values - mean_scores_2


fig1 = go.Figure()
fig1.add_trace(go.Bar(
    x=labels,
    y=mean_scores_1,
    name='Side 1',
    error_y=dict(
        type='data',
        symmetric=False,
        array=ci_upper_1,
        arrayminus=ci_lower_1
    ),
    marker_color='blue',
    opacity=0.7
))
fig1.add_trace(go.Bar(
    x=labels,
    y=mean_scores_2,
    name='Side 2',
    error_y=dict(
        type='data',
        symmetric=False,
        array=ci_upper_2,
        arrayminus=ci_lower_2
    ),
    marker_color='orange',
    opacity=0.7
))

fig1.update_layout(
    title="Comparison of Mean Sentiment Scores for WWII",
    xaxis_title="Sentiment",
    yaxis_title="Mean Sentiment Score",
    barmode='group',
    legend_title="Sides", 
    height = 500,
    autosize=True,
    yaxis=dict(
        tick0=0,
        dtick=0.1, 
        scaleratio=2
    )
)

fig1.show()
fig1.write_html("wwii_sentiment_mini.html")


# We do observe a higher negative score which isn't surprizing, however the overlapping confidence intervals across all categories indicate that the differences between the Allied and Axis films' sentiment scores are likely not statistically significant. However, it is surpriing to observe that the confidence intervall is much higher for the compound result. Let's observe the distribution. 

# In[16]:


# Visualisation 2 : Distribution of Compound Sentiment Scores
fig2 = px.box(
    x=['Side 1'] * len(sentiment_df_1['compound']) + ['Side 2'] * len(sentiment_df_2['compound']),
    y=np.concatenate([sentiment_df_1['compound'], sentiment_df_2['compound']]),
    labels={"x": "Sides", "y": "Compound Sentiment Score"},
    color=['Side 1'] * len(sentiment_df_1['compound']) + ['Side 2'] * len(sentiment_df_2['compound']),
    title="Distribution of Compound Sentiment Scores",
    boxmode="group"
)

fig2.update_traces(
    boxpoints="all",  
    jitter=0.3,       
    pointpos=-1.8,    
    marker=dict(opacity=0.6)
)

fig2.update_layout(
    title="Distribution of Compound Sentiment Scores",
    xaxis_title="Sides",
    yaxis_title="Compound Sentiment Score",
    height = 500,
    autosize=True
)



fig2.show()
fig2.write_html("wwii_distribution_mini.html")



# The distribution seems pretty similar among the two samples, however the sample sizes (n=295 for Side 1 and n=64 for Side 2)are imbalanced, which could contribute to the variability observed, particularly for Side 2.
# 
# Let's still conduct a T-test to confirm our observations. 
# 

# In[17]:


def perform_t_tests(df1, df2):
    """
    Perform independent t-tests to compare sentiment scores between two DataFrames.
    """
    results = []
    
    for column in ['neg', 'neu', 'pos', 'compound']:
        t_stat, p_value = stats.ttest_ind(df1[column], df2[column], equal_var=False)
        results.append({
            'measure': column,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05  # True if p-value is less than 0.05
        })
    
    return pd.DataFrame(results)


# Perform t-tests for Side 1 and Side 2
t_test_results = perform_t_tests(sentiment_df_1, sentiment_df_2)


print("\nT-Test Results:")
print(t_test_results)



# We conducted a Student's t-test, and the results confirm that there is absolutely no statistical significance.
# 
# However, it remains difficult to draw strong conclusions on this matter. Is our sample size too small? Are the movie plots not representative of the actual content of the films? Before delving deeper into this, let’s carry out the same analysis for Cold War films.

# In[18]:


###COLD WAR###

labels = ['Negative', 'Neutral', 'Positive', 'Compound']
x = np.arange(len(labels))

# Confidence Interval Calculation for Cold War
ci_lower_3 = mean_scores_3 - ci_cold_side1['ci_lower'].values
ci_upper_3 = ci_cold_side1['ci_upper'].values - mean_scores_3

ci_lower_4 = mean_scores_4 - ci_cold_side2['ci_lower'].values
ci_upper_4 = ci_cold_side2['ci_upper'].values - mean_scores_4

# Create a bar chart with error bars for Cold War
fig3 = go.Figure()
fig3.add_trace(go.Bar(
    x=labels,
    y=mean_scores_3,
    name='Cold Side 1',
    error_y=dict(
        type='data',
        symmetric=False,
        array=ci_upper_3,
        arrayminus=ci_lower_3
    ),
    marker_color='green',
    opacity=0.7
))
fig3.add_trace(go.Bar(
    x=labels,
    y=mean_scores_4,
    name='Cold Side 2',
    error_y=dict(
        type='data',
        symmetric=False,
        array=ci_upper_4,
        arrayminus=ci_lower_4
    ),
    marker_color='purple',
    opacity=0.7
))

# Add sample size annotations
for i, (mean, lower, upper) in enumerate(zip(mean_scores_3, ci_lower_3, ci_upper_3)):
    fig3.add_annotation(
        x=labels[i],
        y=mean + upper + 0.05,
        text=f'n={sentiment_df_3.shape[0]}',
        showarrow=False,
        font=dict(size=10)
    )

for i, (mean, lower, upper) in enumerate(zip(mean_scores_4, ci_lower_4, ci_upper_4)):
    fig3.add_annotation(
        x=labels[i],
        y=mean + upper + 0.05,
        text=f'n={sentiment_df_4.shape[0]}',
        showarrow=False,
        font=dict(size=10)
    )

fig3.update_layout(
    title="Comparison of Mean Sentiment Scores for Cold War",
    xaxis_title="Sentiment",
    yaxis_title="Mean Sentiment Score",
    barmode='group',
    legend_title="Sides",
    height=800
)
fig3.show()

# Visualisation 4 : Distribution of Compound Sentiment Scores for Cold War
fig4 = px.box(
    x=['Cold Side 1'] * len(sentiment_df_3['compound']) + ['Cold Side 2'] * len(sentiment_df_4['compound']),
    y=np.concatenate([sentiment_df_3['compound'], sentiment_df_4['compound']]),
    labels={"x": "Sides", "y": "Compound Sentiment Score"},
    color=['Cold Side 1'] * len(sentiment_df_3['compound']) + ['Cold Side 2'] * len(sentiment_df_4['compound']),
    title="Distribution of Compound Sentiment Scores for Cold War",
    boxmode="group"
)


fig4.update_traces(
    boxpoints="all",
    jitter=0.3,       
    pointpos=-1.8,    
    marker=dict(opacity=0.6)
)

fig4.update_layout(
    title="Distribution of Compound Sentiment Scores for Cold War",
    xaxis_title="Sides",
    yaxis_title="Compound Sentiment Score",
    height=800
)

fig4.show()


fig3.write_html("comparison_mean_sentiment_cold_war.html")
fig4.write_html("comparison_mean_sentiment_cold_war.html")


# In[19]:


# Perform t-tests for Side 1 and Side 2 for cold war
t_test_results = perform_t_tests(sentiment_df_3, sentiment_df_4)


print("\nT-Test Results:")
print(t_test_results)


# # Conclusion
# 
# fdsfds
# 
# 

# In[43]:


from plotly.subplots import make_subplots

# Create a figure with subplots
fig = make_subplots(
    rows=2, cols=2, 
    subplot_titles=(
        "Scores for WWII",
        "Scores for Cold War",
        "Distribution for WWII (compound)",
        "Distribution forCold War (compound)"
    ), 
    vertical_spacing = 0.13
)

# Add the first bar plot (WWII mean sentiment scores)
fig.add_trace(
    go.Bar(
        x=['Negative', 'Neutral', 'Positive', 'Compound'],
        y=mean_scores_1,
        name='WWII Side 1',
        error_y=dict(
            type='data',
            symmetric=False,
            array=ci_wwii_side1['ci_upper'].values - mean_scores_1,
            arrayminus=mean_scores_1 - ci_wwii_side1['ci_lower'].values
        ),
        marker_color='blue',
        opacity=0.7
    ),
    row=1, col=1
)
fig.add_trace(
    go.Bar(
        x=['Negative', 'Neutral', 'Positive', 'Compound'],
        y=mean_scores_2,
        name='WWII Side 2',
        error_y=dict(
            type='data',
            symmetric=False,
            array=ci_wwii_side2['ci_upper'].values - mean_scores_2,
            arrayminus=mean_scores_2 - ci_wwii_side2['ci_lower'].values
        ),
        marker_color='orange',
        opacity=0.7
    ),
    row=1, col=1
)

# Add the second bar plot (Cold War mean sentiment scores)
fig.add_trace(
    go.Bar(
        x=['Negative', 'Neutral', 'Positive', 'Compound'],
        y=mean_scores_3,
        name='Cold War Side 1',
        error_y=dict(
            type='data',
            symmetric=False,
            array=ci_cold_side1['ci_upper'].values - mean_scores_3,
            arrayminus=mean_scores_3 - ci_cold_side1['ci_lower'].values
        ),
        marker_color='green',
        opacity=0.7
    ),
    row=1, col=2
)
fig.add_trace(
    go.Bar(
        x=['Negative', 'Neutral', 'Positive', 'Compound'],
        y=mean_scores_4,
        name='Cold War Side 2',
        error_y=dict(
            type='data',
            symmetric=False,
            array=ci_cold_side2['ci_upper'].values - mean_scores_4,
            arrayminus=mean_scores_4 - ci_cold_side2['ci_lower'].values
        ),
        marker_color='purple',
        opacity=0.7
    ),
    row=1, col=2
)

# Add the third box plot (WWII distribution)
fig.add_trace(
    go.Box(
        x=['WWII Side 1'] * len(sentiment_df_1['compound']) + ['WWII Side 2'] * len(sentiment_df_2['compound']),
        y=np.concatenate([sentiment_df_1['compound'], sentiment_df_2['compound']]),
        name='WWII Distribution',
        marker=dict(opacity=0.6)
    ),
    row=2, col=1
)

# Add the fourth box plot (Cold War distribution)
fig.add_trace(
    go.Box(
        x=['Cold Side 1'] * len(sentiment_df_3['compound']) + ['Cold Side 2'] * len(sentiment_df_4['compound']),
        y=np.concatenate([sentiment_df_3['compound'], sentiment_df_4['compound']]),
        name='Cold War Distribution',
        marker=dict(opacity=0.6)
    ),
    row=2, col=2
)

# Update the global layout
fig.update_layout(
    title_text="Sentiment Analysis Comparison for WWII and Cold War",
    height=700,
    showlegend=True, 
    autosize=True
)

fig.update_yaxes(range=[-1.1, 1.0], row=1, col=1)
fig.update_yaxes(range=[-1.1, 1.0], row=1, col=2) 

fig.show()



fig.write_html("global_sentiment_mini.html")


# In[ ]:





# In[ ]:




