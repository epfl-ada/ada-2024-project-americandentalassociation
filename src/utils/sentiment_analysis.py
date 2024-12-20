import pandas as pd
import plotly.express as px
import warnings
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
from src.utils import google_trends as gt
warnings.simplefilter("ignore")
from ast import literal_eval
import plotly.graph_objects as go 
import seaborn as sns
import matplotlib.pyplot as plt


# Defining the function that detects specific keywords related to conflicts 
def label_event_regex(summary):
    if re.search(r"(World\sWar\sII|WWII|Hitler|Nazis|Hiroshima|Holocaust)", summary, re.IGNORECASE):
        return "World War II"
    elif re.search(r"(Vietnam\sWar|Viet\sCong|Saigon)", summary, re.IGNORECASE):
        return "Vietnam War"
    elif re.search(r"(Cold\sWar|Soviet\sUnion|USSR|communism|nuclear|Iron\sCurtain|Berlin\sWall|Cuban\sMissile\sCrisis|Space\sRace|Reagan|Stalin|KGB|Eastern\sBloc|Gorbachev|Perestroika|Glasnost)", summary, re.IGNORECASE):
        return "Cold War"
    elif re.search(r"(Kim|Korea|Korean|NKPA|Demilitarized\sZone|Pyongyang|Seoul|Joint\sChiefs\sof\sStaff)", summary, re.IGNORECASE):
        return "Korean"
    else:
        return "Other"

def extracting_side(wars_df, df):
    #WWII
    wwii_bell = wars_df[wars_df['WarName']=='World War II'][['StateName', 'Side', 'Outcome']]
    wwii_bell_side1 = wwii_bell[wwii_bell['Side'] == 1]['StateName'].values.tolist()
    wwii_bell_side2 = wwii_bell[wwii_bell['Side'] == 2]['StateName'].values.tolist()

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


    #Korean
    korean_bell = wars_df[wars_df['WarName']=='Korean'][['StateName', 'Side', 'Outcome']]
    korean_bell_side1 = korean_bell[korean_bell['Side'] == 1]['StateName'].values.tolist()
    korean_bell_side2 = korean_bell[korean_bell['Side'] == 2]['StateName'].values.tolist()


    #Vietnam War
    viet_bell = wars_df[wars_df['WarName']=='Vietnam War, Phase 2'][['StateName', 'Side', 'Outcome']]
    viet_bell_side1 = viet_bell[viet_bell['Side'] == 1]['StateName'].values.tolist()
    viet_bell_side2 = viet_bell[viet_bell['Side'] == 2]['StateName'].values.tolist()


    # Filter rows where 'Genres' contains "WarMovie"
    df_war_movies = df[df['Genres'].str.contains("War film", case=False, na=False)].reset_index(drop=True)

    # Apply the function to the "summary" column 
    df_war_movies['event'] = df_war_movies['summary'].apply(label_event_regex)



    # Display movies labeled as "Cold War" for verification
    df_war_movies[df_war_movies['event'] == "Cold War"].head()

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
    return wwii_movies_side1, wwii_movies_side2, cold_war_movies_side1, cold_war_movies_side2, korean_war_movies_side1, korean_war_movies_side2, vietnam_war_movies_side1


# Extracting entities corresponding to organizations from the NER tags
def extracting_entities(tree):
    # Initializing the entities 
    entities = []
    for subtree in tree.subtrees():
        # If the label of subtree corresponds to an organization, append its leaves (the names) to the entities list
        if (subtree.label() == 'ORGANIZATION'):
            entities.append(" ".join(word for word, tag in subtree.leaves()))
    return entities



# Defining a function to extract sentiment scores per entity
def entity_sentiment_analysis(summary, entities, country):
    analyzer = SentimentIntensityAnalyzer()

    # Initialization of the variables used 
    global average_sentiment
    entities_sentiment = []
    # Tokenizing the summary by sentences 
    sent = sent_tokenize(summary)
    
    for entity in entities:
        # Extracting the sentences that contain a specific entity, and obtaining the sentiment score of all of these sentences
        entity_sentences = [sentence for sentence in sent if entity in sentence]
        sentiment_scores = [analyzer.polarity_scores(sentence) for sentence in entity_sentences]

        if sentiment_scores:
            #Computing the average compound score for the entity, and creating 
            average_sentiment = {
                'Entity' : entity,
                'Country': country,
                'Compound' : sum([score['compound'] for score in sentiment_scores])/len(sentiment_scores),
            }
        entities_sentiment.append(average_sentiment)

    return entities_sentiment


# Only keeping relevant countries
def named_entity_recognition(wars_df, df, lemmatizer, stop_words, df_war_movies):
    # Changing the sides
    wwii_bell_side1 = ['United States of America', 'United Kingdom', 'USSR', 'Poland', 'France']
    wwii_bell_side2 = ['Italy', 'Germany', 'Japan']


    # Selecting movies 

    # Changing the sides
    wwii_bell_side1 = ['United States of America', 'United Kingdom', 'USSR', 'Poland', 'France']
    wwii_bell_side2 = ['Italy', 'Germany', 'Japan']
    cold_bell = wars_df[wars_df['WarName']=='Cold War'][['StateName', 'Side', 'Outcome']]
    cold_bell_side1 = ['France', 'United States of America', 'Belgium']
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

    # Selecting movies 

    wwii_movies_side1 = wwii_movies_side1[
        (df_war_movies['event'] == "World War II") &
        (df_war_movies['Countries'].apply(lambda loc: any(country in str(loc) for country in wwii_bell_side1)))
    ]

    wwii_movies_side2 = df_war_movies[
        (df_war_movies['event'] == "World War II") &
        (df_war_movies['Countries'].apply(lambda loc: any(country in str(loc) for country in wwii_bell_side2)))
    ]

    cold_war_movies_side1 = df_war_movies[
        (df_war_movies['event'] == "Cold War") &
        (df_war_movies['Countries'].apply(lambda loc: any(country in str(loc) for country in cold_bell_side1)))
    ]

    cold_war_movies_side2 = df_war_movies[
        (df_war_movies['event'] == "Cold War") &
        (df_war_movies['Countries'].apply(lambda loc: any(country in str(loc) for country in cold_bell_side2)))
    ]



    # Tokenize the summaries for NER processing
    wwii_movies_side1['tokenized_summaries'] = wwii_movies_side1['summary'].apply(word_tokenize)
    wwii_movies_side2['tokenized_summaries'] = wwii_movies_side2['summary'].apply(word_tokenize)

    cold_war_movies_side1['tokenized_summaries'] = cold_war_movies_side1['summary'].apply(word_tokenize)
    cold_war_movies_side2['tokenized_summaries'] = cold_war_movies_side2['summary'].apply(word_tokenize)

    # Separating before and after conflicts
    wwii_movies_side1_before = wwii_movies_side1[wwii_movies_side1['Year'] <= 1945]
    wwii_movies_side1_after = wwii_movies_side1[wwii_movies_side1['Year'] > 1945]

    wwii_movies_side2_before = wwii_movies_side2[wwii_movies_side2['Year'] <= 1945]
    wwii_movies_side2_after = wwii_movies_side2[wwii_movies_side2['Year'] > 1945]

    cold_war_movies_side1_before = cold_war_movies_side1[cold_war_movies_side1['Year'] <= 1989]
    cold_war_movies_side1_after = cold_war_movies_side1[cold_war_movies_side1['Year'] > 1989]

    cold_war_movies_side2_before = cold_war_movies_side2[cold_war_movies_side2['Year'] <= 1989]
    cold_war_movies_side2_after = cold_war_movies_side2[cold_war_movies_side2['Year'] > 1989]



    # Initializing variables necessary for iteration over all of the conflicts
    wars_iterator = [wwii_movies_side1_before, wwii_movies_side1_after, wwii_movies_side2, cold_war_movies_side1_before, cold_war_movies_side1_after, cold_war_movies_side2_before, cold_war_movies_side2_after]
    war_names = ['World War II Allied Side Before','World War II Allied Side After', "World War II Axis Side", 'Cold War Western Side Before', 'Cold War Western Side After', 'Cold War Sovietic Side Before', 'Cold War Sovietic Side After']# "Cold War", "Korean war", "Vietnan war"]
    side_iterator = [wwii_bell_side1, wwii_bell_side1, wwii_bell_side2, cold_bell_side1, cold_bell_side1, cold_bell_side2, cold_bell_side2]

    # Extracting the NER tags from each  summary

    full_NER_TAGS = [[], [], [], [], [], [], []] 
    for i, test_sample in enumerate(wars_iterator):
        # Preprocess the text,tokenize, clean and remove stopwords
        tokens_summ = test_sample['tokenized_summaries'].to_list()
        NER_tags = []
        for summary in tokens_summ:
            filtered_token = [word for word in summary if word not in stop_words]
            lemmatized_tokens = ([lemmatizer.lemmatize(token) for token in filtered_token])
            pos_tags = nltk.pos_tag(lemmatized_tokens)
            # Perform NER chunking to identify named entities
            ner_tags = ne_chunk(pos_tags)
            NER_tags.append(ner_tags)

        full_NER_TAGS[i] = NER_tags



    # Extracting entities 
    full_entities = [[], [], [], [], [], [], []]
    for i, tags in enumerate(full_NER_TAGS): 
        entities = []
        for tree in tags:
            entities.append(extracting_entities(tree))
        full_entities[i] = entities 

    # Creating a new column in each movies dataset containing the entities
    wwii_movies_side1_before['Entities'] = full_entities[0]
    wwii_movies_side1_after['Entities'] = full_entities[1]
    wwii_movies_side2['Entities'] = full_entities[2]
    cold_war_movies_side1_before['Entities'] = full_entities[3]
    cold_war_movies_side1_after['Entities'] = full_entities[4]
    cold_war_movies_side2_before['Entities'] = full_entities[5]
    cold_war_movies_side2_after['Entities'] = full_entities[6]
    # Updating variables necessary for iteration over all of the conflicts
    wars_iterator = [wwii_movies_side1_before, wwii_movies_side1_after, wwii_movies_side2, cold_war_movies_side1_before, cold_war_movies_side1_after, cold_war_movies_side2_before, cold_war_movies_side2_after]
    war_names = ['World War II Allied Side Before','World War II Allied Side After', "World War II Axis Side", 'Cold War Western Side Before', 'Cold War Western Side After', 'Cold War Sovietic Side Before', 'Cold War Sovietic Side After']# "Cold War", "Korean war", "Vietnan war"]
    side_iterator = [wwii_bell_side1, wwii_bell_side1, wwii_bell_side2, cold_bell_side1, cold_bell_side1, cold_bell_side2, cold_bell_side2]

    return wars_iterator, war_names, side_iterator


def entity_level_sent_analysis(wars_iterator, war_names, side_iterator):
    # Perform entity-level sentiment analysis on every conflict's subset of movies
    for i, test_sample in enumerate(wars_iterator):
        # Create a new dataframe for analysis
        summaries = test_sample[['summary', 'Entities', 'Countries']]

        # Perform the entity-level analysis on each row of the dataset and appending it in the results table
        results = []
        for _, row in summaries.iterrows():
            sentiments = entity_sentiment_analysis(row["summary"], row["Entities"], row['Countries'])
            results.extend(sentiments)

        # Convert results to a DataFrame
        entity_sentiments_df = pd.DataFrame(results)

        sentiment = []
        for j in range(len(entity_sentiments_df)):
            if entity_sentiments_df.iloc[j, 2] >= 0.05:
                sentiment.append('Protagonist')
            elif entity_sentiments_df.iloc[j, 2] <= -0.05:
                sentiment.append('Antagonist')
            else:
                sentiment.append('Protagonist')

        # Create a new column for the role of the entity (Protagonist or Antagonist) and drop the duplicates 
        entity_sentiments_df['Role'] = sentiment
        #entity_sentiments_df.drop_duplicates(inplace=True)
        entity_sentiments_df['Country'] = entity_sentiments_df['Country'].apply(literal_eval)

        # Explode the dataframe on the country column in order to analyse country-wise
        entity_sentiments_df_exploded = entity_sentiments_df.explode("Country").reset_index(drop=True)

        # Remove irrelevant countries that might have appeared after exploding
        entity_sentiments_df_exploded = entity_sentiments_df_exploded[entity_sentiments_df_exploded['Country'].isin(side_iterator[i])]

        # Replacing United States of America by USA for better readability
        entity_sentiments_df_exploded.Country = entity_sentiments_df_exploded.Country.str.replace('United States of America', 'USA')

        # Aggregate by contry and sentiment range 
        aggregated_data = entity_sentiments_df_exploded.groupby(["Entity", "Country"]).agg({
        "Compound": "mean"}).reset_index()
        # Pivot aggregated data
        heatmap_data = aggregated_data.pivot(index="Country", columns="Entity", values="Compound")

        # Create a heatmap with Seaborn
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=False, fmt=".2f", cmap="RdBu", xticklabels='auto', yticklabels='auto', cbar_kws={"label": "Compound"})

        # Add title and axis labels
        plt.title(f"Compound Score of Organizations, for {war_names[i]}", fontsize=16)
        plt.xlabel("Entities", fontsize=12)
        plt.ylabel("Countries", fontsize=12)

        # Save or show the heatmap
        plt.tight_layout()
        plt.show()
    

