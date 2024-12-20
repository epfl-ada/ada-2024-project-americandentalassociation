# Cinematography in Times of Crisis: Exploring the Impact of Global Events on Film Genres and Public Preferences

## Abstract
Cinematography serves as both a reflection of reality and a form of entertainment, adapting to historical, political, economic, and social developments. Over the past century, the world has faced various upheavals, including wars, economic crises, pandemics, and natural disasters. This study investigates how such events have influenced film production and public preferences. 

Questions explored include whether global crises lead to shifts toward various genres like war films, spy thrillers, or dystopias and how releases differ in countries involved in a conflict. Additionally, it examines whether propagandistic goals shape films and how genres evolve in response to global events. Ultimately, this research analyzes trends in movie genres and how they reflect societal responses to crises.

## Data Story Link
[Cinematography in Times of Crisis](https://bintadouk.github.io/ada-project-americanDentalAssociation/)

## Main Research Questions
1. **How do wars shape the film production, genres and themes over time?**
   The analysis in the first question examines how various wars influenced the production of different movie genres over time. By defining groups of meaningful genres, analyzing genre proportions, correlations with war-related metrics, co-occurrence networks, and significant changes in genre distributions, this analysis uncovers key insights about the interplay between historical conflicts and cinema production.

2. **How do movie genre preferences differ between countries in conflict?**
   In this question we go deeper into details on one specific side of the analysis of the relation between movies and historical events, by looking at the effect which some global events (in this case we mainly focus on 4 wars) influence different countries involved in them. We do both a quantitative analysis (amounts of movies produced) and qualitative analysis (differences in genres) for subsets of movies filtered starting from the conflicts.

3. **How does the portrayal of historical conflicts in movies vary based on the country of production?**
   How do different nations construct narratives in war films to reflect their unique historical perspectives, cultural values, and national identities? This question seeks to uncover whether variations in sentiment, framing, and entity portrayal across films reveal distinct approaches to processing shared historical events. By examining films about World War II and the Cold War, we aim to understand whether differences emerge in how countries memorialize conflicts, depict allies and adversaries, and convey emotional tones. Exploring these patterns could shed light on the broader role of cinema in shaping collective memory and interpreting the complexities of war.

4. **How do global events, such as natural disasters, pandemics, wars, and economic downturns, influence public preferences for certain movie thematics?**
   Over the course of this analysis, we have explored the evolution of war cinema and the sentiment associated with different entities during global conflicts. All of that was based on the internal production and representation of the movies. In this final part, we switched to viewer sentiment and interest, aligning global events such as wars, pandemics, and natural disasters to inspect both short-term and long-term shifts in audience preferences.


## Data Sources
- **IMDb Movies Dataset**: Used for film rankings, release date, and revenue analysis.
- **Wikipedia Timeline of World Events**: Focuses on 20th and 21st-century crises, including wars, natural disasters, and pandemics, using a zero-shot classifier to categorize events. Positive events are excluded for research focus.
- **Google Search Trends**: Provides insights into regional interest trends from 2004 onward, offering data for genre popularity analysis related to global crises.

# Methods

## Data Preparation & Cleaning
- Wikipedia Timeline was scraped directly from Wikipedia’s website. After cleaning, each entry contained data and a short description of an event. Event types were then added using a pre-trained NLI-based zero-shot text classifier (classes: war, catastrophe, political instability, political resolution, science, technological advancement, natural disaster, pandemics). Positive events were removed to better focus on defined research questions.  
- Google Search Trends is not a unique dataset that can be fetched statically. Therefore, a special API from Oxylabs was used to define utility functions for fetching time series for each chosen keyword and timeline. This utility will then be used in Milestone 3 to fetch structured and clean data for our events and dates of interest.
- We also applied text cleaning techniques such as stopword removal, character folding, tokenization, and lemmatization to the movies synopses directly in the `results.ipynb` file as it was a relatively light data cleaning step. 

## Research Question Analysis

### Q1: Statistical Analysis of Film Trends
Q1 Aims to analyze genres proportion over time and for events chosen
- Genres are grouped into military, military+antiwar, dystopian and political and the proportion is inspected throughout time
- By inspecting the peaks of the plots in the 1st questions, defining wars in the +-2 years window and grouping the events, 5 wars are chosen. Correlation analysis is conducted for the timeline and chosen events - proportion vs. number of ongoing wars; proportion vs. yearly battle deaths. Positive genres list and individual genres are added to analysis. Network for War Film is created for more nuance.
- Significant changes in genres are identified for each chosen war with chi-square testing.
- The changes of proportion for all events in the genre groups are visualized and analyzed.


### Q2: Cross-National Genre Comparisons
In Q2 we delve into a more detailed analysis of the differences in response to conflicts between different countries:
- Performing a brief general analysis of the conflicts, showing the involved countries and how the amount of movies produced referencing them changed over time. 
- Performing a quick quantitative analysis, showing for each conflicts how many movies referencing it are produced by countries on either side of the conflict 
- Doing a deeper analysis on genres, both on movies released in the period of time around the conflict and in movies referencing the conflict, comparing the results obtained for countries on both sides.


### Q3: Cross-National Opinions Comparisons
In Q3 we used Natural Language Processing (NLP) to analyze themes, tones, and conflict portrayals in movies, focusing on variations by country. This was done by using the following steps:
- Applying text preprocessing techniques in order to make them suitable for analysis. Those techniques include tokenization, lemmatization
- Identify relevant conflicts to analyse based on data abundance 
- Classifying countries according to their implication in different conflicts
- Categorizing Movies by Conflict : Movies tagged as “War Movie” are filtered using keywords related to specific conflicts (e.g., WW2).
- Sentiment Analysis: Sentiment analysis on summaries generates negative, neutral, and positive scores, allowing cross-country comparisons of portrayals. 
- Analysis of the distributions of the sentiment with Boxplots visualisations
- T-test to assess the significativity of the differences in the sentiment analysis
- Key Themes and Entities: Named Entity Recognition (NER) extracts key entities (people, organizations, locations) for entity-level sentiment analysis. Focusing on organization tagged entities, this helped us determine how principal war stakeholders are depicted, and whether we can witness shifts through context and time. Results, visualized with heatmaps, reveal patterns in portraying heroes (positive) and villains (negative) across countries.


### Q4: Audience Preferences During Crises
Q4 performs a more general analysis including external information provided by Google Search Trends and inspecting patterns in movies popularity over time, variations, and the impact of global events on the trends in movies interest.
- Analyzed the evolution of interest in the top 20 movie genres over time using Google Search Trends, identifying peaks during significant periods such as the COVID pandemic and 2016-2017. 
- Performed seasonal decomposition analysis on genre popularity to identify patterns, highlighting genres like Horror and Family movies with strong seasonality and others, like Historical movies, with prolonged negative seasonality. 
- Conducted trend analysis using polynomial regression to examine long-term changes in genre popularity, identifying consistent growth for genres like Crime and Thriller, while Historical and Music movies showed a decline.
- Investigated correlations between the number of genre-specific movies released and audience interest, observing negative correlations for most genres but notable exceptions like Sci-Fi and Musicals, though results were not statistically significant.

## Specific Methods Applied
The summary of the main methods that were used in our analysis is the following:
- Standard statistical analysis, correlation analysis, hypothesis testing
- Time-series analysis, impact analysis
- Zero-shot text classification on movie plot summaries and event descriptions
- Tokenize movie plots/Use NLP for sentiment analysis etc, Named Entity Recognition for character analysis
- Seasonality decomposition analysis, regression analysis, causality checks
- Movies characters analysis using semantic analysis

  
## Team Organization
Milestone 3 was the time to deepen our analysis and findings. Globally, each team member is responsible for a specific part of the analysis (see below), while merging and ensuring coherence is done collaboratively or in pairs. We hold weekly meetings to refine task distribution. Here is the exact distribution of the tasks between all team members:   
**Anna**: Question 1.   
**Binta**: Named Entity Recognition, Entity-Level Sentiment Analysis and website setup.   
**Michele**: Initial dataset cleaning and filtering, Question 2 (quantitative and comparative analysis of movies in different countries), README.md setup.   
**Flavia**: War Dataset Research,Question 3: Sentiment Analysis, use of NLP methods for text preparation and NLTK libraries for sentiment analysis.   
**Aidas**: Question 4; Analysis of genre popularity trends, global events impact analysis for both short-term and long-term trends.   

## Plans and Timeline
The timeline followed for the project was the following:   

**Week 1: Refinement**  
- Align on our strategy: decide whether to build on each other's analyses or pursue independent event selections.  
- Refine event selection and adjust variables (e.g., years, countries, topics) to uncover more interesting insights.  
- Clearly state our findings and brainstorm the most effective ways to represent them.  

**Week 2: Visualization**  
- plot visualizations to highlight key findings for all research questions.  
- Strategize how to present the visualizations in a compelling manner, enabling effective comparisons between methods (e.g., sentiment analysis, Google Trends, occurrences).  

**Week 3: Presentation Preparation**  
- Build the website, integrating visualizations and text to create an engaging story.  

**Week 4: Buffer and Feedback**  
- Seek feedback from the TAs to refine and improve the final data story. 