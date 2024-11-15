# Cinematography in Times of Crisis: Exploring the Impact of Global Events on Film Genres and Public Preferences

## Abstract
Cinematography serves as both a reflection of reality and a form of entertainment, adapting to historical, political, economic, and social developments. Over the past century, the world has faced various upheavals, including wars, economic crises, pandemics, and natural disasters. This study investigates how such events have influenced film production and public preferences. 

Questions explored include whether global crises lead to shifts toward various genres like war films, spy thrillers, or dystopias and how releases differ in countries involved in a conflict. Additionally, it examines whether propagandistic goals shape films and how genres evolve in response to global events. Ultimately, this research analyzes trends in movie genres and how they reflect societal responses to crises.

## Main Research Questions
1. **How do global crises and significant world events shape film production, themes, and public preferences?**  
   Examines shifts in film genres and content—such as increased production and popularity of war, spy, or superhero films—during global crises.

2. **How do movie genre preferences differ between countries in conflict or those experiencing similar global events?**  
   Investigates variations in genre trends across nations, focusing on contrasts like those between the U.S. and Russia during political tensions or how pandemics influence preferences globally.

3. **How does the portrayal of historical events in movies vary based on the country of production?**  
   Analyzes narrative and thematic differences in films depicting shared historical events, revealing how cultural and national contexts shape storytelling.

4. **How do global events like natural disasters, pandemics, and wars influence public preferences for specific themes and genres?**  
   Explores shifts toward escapist genres like fantasy or comedy versus more grounded or serious genres, with regional comparisons to highlight cultural differences. Analysis of public interest with Google Search Trends.

## Data Sources
- **IMDb Movies Dataset**: Used for film rankings, release date, and revenue analysis.
- **Wikipedia Timeline of World Events**: Focuses on 20th and 21st-century crises, including wars, natural disasters, and pandemics, using a zero-shot classifier to categorize events. Positive events are excluded for research focus.
- **Google Search Trends**: Provides insights into regional interest trends from 2004 onward, offering data for genre popularity analysis related to global crises.

### Methods

## Data Preparation & Cleaning
- Wikipedia Timeline was scraped directly from Wikipedia’s website. After cleaning, each entry contained data and a short description of an event. Event types were then added using a pre-trained NLI-based zero-shot text classifier (classes: war, catastrophe, political instability, political resolution, science, technological advancement, natural disaster, pandemics). Positive events were removed to better focus on defined research questions.  
  *Note*: The initial exploratory analysis in this milestone was carried out by including events not necessarily in this list. In Milestone 3, only events from this extensive dataset will be used.
- Google Search Trends is not a unique dataset that can be fetched statically. Therefore, a special API from Oxylabs was used to define utility functions for fetching time series for each chosen keyword and timeline. This utility will then be used in Milestone 3 to fetch structured and clean data for our events and dates of interest.
- We also applied text cleaning techniques such as stopword removal, character folding, tokenization, and lemmatization to the movies synopses directly in the `results.ipynb` file as it was a relatively light data cleaning step. 

## Research Question Analysis

# Q1: Statistical Analysis of Film Trends
- Lead statistical analysis and perform visualizations for the number of movies released and the revenue, as well as values like absolute change and relative change, for chosen years. 
- Explore specific genres over the years (“War Film”, “Superhero”, “Spy”) and inspect the spikes in production and revenue. 
- Identify the methodology for year comparison (before the start of a war, start, middle, end, after - ideally, take into account some significant events or changes) and the events that have had significant influence on the movie industry.

# Q2: Cross-National Genre Comparisons
- Filter the movie dataset by countries and observe the differences in produced genres between different countries. 
- Check how the produced genres evolve over time, and try to identify trends related to certain genres with historical events.
- Filter the dataset based on the summary, observing the genres of movies talking about a certain country and the differences in the way they are depicted.

# Q3: Cross-National Opinions Comparisons
- Focus on movies covering specific conflicts (e.g., WW2).
- Conduct sentiment analysis on summaries, generating negative, neutral, and positive scores, allowing for cross-country comparisons.
- Conduct Named Entity Recognition to extract key entities, such as main figures in movies, in order to analyze how sentiment towards figures varies depending on locations.

# Q4: Audience Preferences During Crises
- Conduct the analysis by aligning global events such as wars, pandemics, and natural disasters to inspect both short-term and long-term shifts in audience preferences.
- Perform genre-specific trend analysis to identify immediate spikes in popularity as well as sustained changes over time.
- Compare regional differences and potentially integrate NLP methods (e.g., zero-shot text classification) to classify movie thematics better.
- Visualize the data to highlight both immediate and lasting impacts of global events on movie thematics.

## Plans and Timeline

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
  
## Team Organization
Globally, each team member is responsible for a specific part of the analysis (see below), while merging and ensuring coherence is done collaboratively or in pairs. We hold weekly meetings to refine task distribution. 
Q1: Anna (Film Trends Analysis)
Q2: Michele (Cross-National Genre Comparisons)
Q3: Binta (NER Analysis) and Flavia (Sentiment Analysis)
Q4: Aidas (Audience Preferences and Google Trends Analysis)

## Questions for TAs
- API Access: Google frequently updates backend restrictions, necessitating a third-party API for Question 4. How should authentication keys be handled?
