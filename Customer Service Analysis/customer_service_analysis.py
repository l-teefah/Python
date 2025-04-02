
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')


#%pip install seaborn
import seaborn as sns


df = pd.read_csv('Customer_support_data.csv')
df


missing_values = df.isnull().sum()
missing_values


rows_with_missing_values = df[df.isnull().any(axis=1)]
rows_with_missing_values


df.dtypes


date_columns = ['order_date_time', 'Issue_reported at', 'issue_responded', 'Survey_response_Date']
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Calculate response time in minutes
if 'Issue_reported at' in df.columns and 'issue_responded' in df.columns:
    df['Response time (min)'] = (df['issue_responded'] - df['Issue_reported at']).dt.total_seconds() / 60

# Calculate response time in hours
if 'Issue_reported at' in df.columns and 'issue_responded' in df.columns:
    df['Response time (hrs)'] = (df['issue_responded'] - df['Issue_reported at']).dt.total_seconds() / 3600

df


# check if the problem with rows not importing properly in R also occured here but they didn't


problem_rows = df.iloc[[15437, 45128, 45176]] 
problem_rows


# Setting up the visualization style
plt.style.use('ggplot')
sns.set(style="whitegrid")


# Distribution of CSAT scores
plt.figure(figsize=(10, 6))
sns.histplot(df['CSAT Score'], kde=False, bins=5)
plt.title('Distribution of CSAT Scores')
plt.xlabel('CSAT Score')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()



# CSAT by channel
plt.figure(figsize=(12, 6))
sns.boxplot(x='channel_name', y='CSAT Score', data=df, palette="Set2")
plt.xlabel('Channel Name')
plt.ylabel('CSAT Score')
plt.title('CSAT Scores by Channel')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# CSAT by product category
plt.figure(figsize=(12, 6))
avg_csat = df.groupby('Product_category')['CSAT Score'].mean().sort_values(ascending=False)
sns.barplot(x=avg_csat.index, y=avg_csat.values, palette="Set3")
plt.title('Average CSAT by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Average CSAT')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



# Response time analysis
plt.figure(figsize=(10, 6))
sns.histplot(df['Response time (hrs)'].clip(upper=df['Response time (hrs)'].quantile(0.95)), kde=True)
plt.title('Distribution of Response Time (95th percentile)')
plt.xlabel('Response Time (Hrs)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()



# Response time vs CSAT
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Response time (hrs)', y='CSAT Score', data=df, alpha=0.5)
plt.title('Response Time vs CSAT Score')
plt.xlabel('Response Time (Hours)')
plt.ylabel('CSAT Score')
plt.grid(False)
plt.tight_layout()
plt.show()



# CSAT by agent tenure
plt.figure(figsize=(10, 6))
sns.boxplot(x='Tenure Bucket', y='CSAT Score', data=df, palette="Set3")
plt.title('CSAT Scores by Agent Tenure')
plt.xlabel('Tenure Bucket')
plt.ylabel('CSAT Score')
plt.show()



# CSAT by agent shift
plt.figure(figsize=(10, 6))
sns.boxplot(x='Agent Shift', y='CSAT Score', data=df, palette="Set2")
plt.title('CSAT Scores by Agent Shift')
plt.xlabel('Agent Shift')
plt.ylabel('CSAT Score')
plt.show()



# Average Response time by channel
plt.figure(figsize=(12, 6))
avg_handling = df.groupby('channel_name')['Response time (hrs)'].mean().sort_values(ascending=False)
sns.barplot(x=avg_handling.index, y=avg_handling.values, palette="Set2")
plt.title('Average Response Time by Channel')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Channel Name')
plt.ylabel('Average Response Time (hours)')
plt.tight_layout()
plt.show()



# Correlation heatmap of numeric variables
plt.figure(figsize=(12, 10))
numeric_df = df.select_dtypes(include=['float64', 'int64'])
correlation = numeric_df.corr()
mask = np.triu(correlation)
sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm', mask=mask)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()



# Top 10 cities by number of issues
plt.figure(figsize=(12, 6))
city_counts = df['Customer_City'].value_counts().head(20)
sns.barplot(x=city_counts.index, y=city_counts.values, palette="Set2")
plt.title('Top 20 Cities by Number of Complaints')
plt.xticks(rotation=45, ha='right')
plt.xlabel('City')
plt.ylabel('Number of Complaints')
plt.tight_layout()



# Agent performance analysis
plt.figure(figsize=(14, 8))
agent_perf = df.groupby('Agent_name').agg({
    'CSAT Score': 'mean',
    'Response time (hrs)': 'mean',
    'Unique id': 'count'
}).sort_values('CSAT Score', ascending=False)
agent_perf = agent_perf.rename(columns={'Unique id': 'Ticket count'})
agent_perf = agent_perf.head(20)  # Top 20 agents by CSAT

plt.subplot(1, 2, 1)
sns.barplot(x=agent_perf.index, y=agent_perf['CSAT Score'], palette="Set2")
plt.title('Top 20 Agents by CSAT Score')
plt.xticks(rotation=90)
plt.xlabel('Agent Name')
plt.ylabel('CSAT Score')
plt.ylim(agent_perf['CSAT Score'].min() * 0.9, agent_perf['CSAT Score'].max() * 1.1)

plt.subplot(1, 2, 2)
sns.barplot(x=agent_perf.index, y=agent_perf['Response time (hrs)'], palette="Set3")
plt.title('Handling Time for Top 20 Agents')
plt.xticks(rotation=90)
plt.xlabel('Agent Name')
plt.ylabel('Response Time (Hrs)')

plt.tight_layout()
plt.show()


# Feature Engineering

# Select features
categorical_features = ['channel_name', 'category', 'Sub-category', 'Product_category', 
                       'Customer_City', 'Tenure Bucket', 'Agent Shift']
numeric_features = ['Item_price', 'connected_handling_time', 'Response time (hrs)', 
                   'Remarks_length']

# Add any time-based features created
for col in date_columns:
    if f'{col}_hour' in df.columns:
        numeric_features.append(f'{col}_hour')
    if f'{col}_day' in df.columns:
        numeric_features.append(f'{col}_day')

# Remove any columns that don't exist in the dataset
categorical_features = [col for col in categorical_features if col in df.columns]
numeric_features = [col for col in numeric_features if col in df.columns]


# Time-based features
for col in date_columns:
    if col in df.columns:
        df[f'{col}_hour'] = df[col].dt.hour
        df[f'{col}_day'] = df[col].dt.day_of_week
        df[f'{col}_month'] = df[col].dt.month

# Customer sentiment analysis 
df['Remarks_length'] = df['Customer Remarks'].str.len()

#!pip install nltk
import nltk 
#nltk.download('all')
nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('vader_lexicon')  # For sentiment analysis
#nltk.download('averaged_perceptron_tagger')  # For POS tagging
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Basic text cleaning
df['Remarks_cleaned'] = df['Customer Remarks'].str.lower()  # Lowercase
df['Remarks_cleaned'] = df['Remarks_cleaned'].str.replace(r'[^\w\s]', '', regex=True)  # Remove punctuation
df['Remarks_cleaned'] = df['Remarks_cleaned'].fillna('')  # Replace NaN with empty string
df['Remarks_cleaned'] = df['Remarks_cleaned'].astype(str)  # Ensure all values are strings

# Tokenization
df['Remarks_tokens'] = df['Remarks_cleaned'].astype(str).apply(word_tokenize)

# Count number of words
df['Remarks_word_count'] = df['Remarks_tokens'].apply(len)

# Remove stopwords
stop_words = set(stopwords.words('english'))
df['Remarks_filtered'] = df['Remarks_tokens'].apply(lambda x: [word for word in x if word not in stop_words])

# Lemmatization
lemmatizer = WordNetLemmatizer()
df['Remarks_lemmatized'] = df['Remarks_filtered'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# Sentiment analysis
sia = SentimentIntensityAnalyzer()
# Convert to string but replace NaN with empty string
df['Customer Remarks'] = df['Customer Remarks'].fillna('').astype(str)
df['Sentiment_scores'] = df['Customer Remarks'].apply(lambda x: sia.polarity_scores(x))
df['Sentiment_compound'] = df['Sentiment_scores'].apply(lambda x: x['compound'])
df['Sentiment_category'] = df['Sentiment_compound'].apply(lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral'))

# Create Bag of Words features
count_vectorizer = CountVectorizer(max_features=100)
bow_matrix = count_vectorizer.fit_transform(df['Remarks_cleaned'])
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=count_vectorizer.get_feature_names_out())
df = pd.concat([df, bow_df], axis=1)

# Create TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=100)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Remarks_cleaned'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
df = pd.concat([df, tfidf_df], axis=1)

# Extract key phrases or topics
# (This is a simple approach - more sophisticated topic modeling might use LDA)
df['Key_nouns'] = df['Remarks_tokens'].apply(lambda x: [word for word, pos in nltk.pos_tag(x) if pos.startswith('NN')])



# Create Topics dataframe
# Combine all reviews into a single text
all_reviews = df['Remarks_cleaned'].dropna().tolist()

# Use TfidfVectorizer to convert the text data into TF-IDF features
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf = tfidf_vectorizer.fit_transform(all_reviews)

# Use NMF to extract topics
from sklearn.decomposition import NMF
num_topics = 5
nmf_model = NMF(n_components=num_topics, random_state=42)
nmf_model.fit(tfidf)

# Get the top words for each topic
words = tfidf_vectorizer.get_feature_names_out()
topics = []
for topic_idx, topic in enumerate(nmf_model.components_):
    top_words = [words[i] for i in topic.argsort()[:-11:-1]]
    topics.append(top_words)

# Create a DataFrame to display the topics
topics_df = pd.DataFrame(topics, columns=[f'Word {i+1}' for i in range(10)])
topics_df.index = [f'Topic {i+1}' for i in range(num_topics)]
topics_df



#!pip install gensim
import gensim
from gensim import corpora
from gensim.models import LdaModel
#!pip install pyLDAvis
import pyLDAvis
import pyLDAvis.gensim_models

# Create dictionary (mapping of id to word)
dictionary = corpora.Dictionary(df['Remarks_lemmatized'])

# Filter out extreme values (optional but recommended)
dictionary.filter_extremes(no_below=5, no_above=0.5)  # Adjust parameters based on your data

# Create document-term matrix
corpus = [dictionary.doc2bow(text) for text in df['Remarks_lemmatized']]

# Build LDA model
num_topics = 5  # You can adjust this based on your needs
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=num_topics,
    passes=10,
    alpha='auto',
    random_state=42
)

# Print the topics
for idx, topic in lda_model.print_topics(-1):
    print(f'Topic {idx}: {topic}')

# Get topic distribution for each document
df['Topic_distribution'] = [lda_model[corpus[i]] for i in range(len(df))]

# Assign primary topic to each document
df['Primary_topic'] = df['Topic_distribution'].apply(
    lambda x: max(x, key=lambda item: item[1])[0] if x else None
)



# Visualize topics using pyLDAvis
import pyLDAvis.gensim_models as gensimvis
vis = gensimvis.prepare(lda_model, corpus, dictionary)
#pyLDAvis.save_html(vis, 'lda_visualization.html')


# Extract top keywords for each document based on LDA weights
def get_top_keywords(lda_model, doc_bow, n=3):
    """Get top keywords for a document based on LDA weights"""
    topic_weights = lda_model[doc_bow]
    if not topic_weights:
        return []
    
    # Get the distribution of topics for this document
    topics = sorted(topic_weights, key=lambda x: x[1], reverse=True)
    
    # Get the words for the top topic
    top_topic_idx = topics[0][0]
    topic_terms = lda_model.get_topic_terms(top_topic_idx, n)
    
    # Convert term IDs to actual words
    keywords = [dictionary[term_id] for term_id, _ in topic_terms]
    return keywords

# Apply to each document
df['Top_keywords'] = [get_top_keywords(lda_model, doc) for doc in corpus]



# WordCloud
from wordcloud import WordCloud

# Function to generate a word cloud for a topic
def plot_wordcloud(lda_model, topic_id, dictionary):
    topic_terms = lda_model.get_topic_terms(topic_id, topn=20)
    words = {dictionary[word_id]: weight for word_id, weight in topic_terms}

    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        colormap='viridis'
    ).generate_from_frequencies(words)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud for Topic {topic_id}")
    plt.show()

# Generate word clouds for the first 5 topics
for topic_id in range(5):  # Change based on the number of topics
    plot_wordcloud(lda_model, topic_id, dictionary)


# Machine Learning - CSAT Score Prediction

from sklearn.impute import SimpleImputer
df['High_CSAT'] = (df['CSAT Score'] >= 4).astype(int)

# Create X and y
X = df[categorical_features + numeric_features]
y = df['High_CSAT']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create preprocessor with imputation step to make up for NaN values
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Add imputer for numeric features
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Add imputer for categorical features
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create and train Random Forest model (rest of your code stays the same)
rf_classifier = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy:", accuracy_score(y_test, y_pred))



# Advanced ML - Feature Importance

# Extract feature names
feature_names = []

# Get feature names for numeric columns (they stay the same)
feature_names.extend(numeric_features)

# Get feature names for categorical columns after one-hot encoding
for cat_feature in categorical_features:
    # Get unique values for each categorical feature
    unique_values = df[cat_feature].unique()
    for value in unique_values:
        feature_names.append(f"{cat_feature}_{value}")

# Extract feature importances from Random Forest
try:
    # This assumes the feature names match exactly with the transformed features
    rf_importances = rf_classifier.named_steps['classifier'].feature_importances_
    
    # Use the first n importances (may not match exactly with feature_names)
    n_features = min(len(rf_importances), len(feature_names))
    
    # Sort feature importances
    indices = np.argsort(rf_importances)[::-1][:n_features]
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x=[feature_names[i] for i in indices[:15]], y=rf_importances[indices[:15]], palette ="Set3")
    plt.title('Top 15 Feature Importances for CSAT Prediction')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
except:
    print("Couldn't extract feature importances properly - dimensionality issue with feature names")


# Logistic Regression Model for Interpretability

# Create and train Logistic Regression model
lr_classifier = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

lr_classifier.fit(X_train, y_train)

# Make predictions
lr_y_pred = lr_classifier.predict(X_test)

# Evaluate the model
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, lr_y_pred))

print("\nAccuracy:", accuracy_score(y_test, lr_y_pred))


# Customer Behavior Analysis

# Group by customer city and analyze patterns
city_analysis = df.groupby('Customer_City').agg({
    'Unique id': 'count',
    'CSAT Score': 'mean',
    'Response time (hrs)': 'mean'
}).sort_values('Unique id', ascending=False).head(10)

city_analysis = city_analysis.rename(columns={'Unique id': 'Ticket count'})

print("\nTop 10 Cities Analysis:")
print(city_analysis)

# Product category analysis
product_analysis = df.groupby('Product_category').agg({
    'Unique id': 'count',
    'CSAT Score': 'mean',
    'Response time (hrs)': 'mean',
    'Item_price': 'mean'
}).sort_values('Unique id', ascending=False)

product_analysis = product_analysis.rename(columns={'Unique id': 'Ticket count'})

print("\nProduct Category Analysis:")
print(product_analysis)



# Time-Based Analysis

# Create day of week and hour features 
if 'Issue_reported at' in df.columns:
    df['Day_of_week'] = df['Issue_reported at'].dt.day_name()
    df['Hour_of_day'] = df['Issue_reported at'].dt.hour

    # CSAT by day of week
    plt.figure(figsize=(10, 6))
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_csat = df.groupby('Day_of_week')['CSAT Score'].mean()
    # Reindex to ensure correct order
    day_csat = day_csat.reindex(day_order)
    sns.barplot(x=day_csat.index, y=day_csat.values)
    plt.title('Average CSAT Score by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Average CSAT Score')
    plt.show()

    # CSAT by hour of day
    plt.figure(figsize=(12, 6))
    hour_csat = df.groupby('Hour_of_day')['CSAT Score'].mean()
    sns.lineplot(x=hour_csat.index, y=hour_csat.values, marker='o', palette ="Set2")
    plt.title('Average CSAT Score by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average CSAT Score')
    plt.xticks(range(0, 24))
    plt.grid(False)
    plt.show()



# Agent Performance Metrics

# Create agent performance dashboard
agent_metrics = df.groupby('Agent_name').agg({
    'Unique id': 'count',
    'CSAT Score': 'mean',
    'Response time (hrs)': 'mean',
    'High_CSAT': 'mean'
}).sort_values('Unique id', ascending=False)

agent_metrics = agent_metrics.rename(columns={
    'Unique id': 'Ticket_count',
    'High_CSAT': 'High_CSAT_percentage'
})
agent_metrics['High_CSAT_percentage'] = agent_metrics['High_CSAT_percentage'] * 100

# Only show agents with significant number of tickets (e.g., > 50)
significant_agents = agent_metrics[agent_metrics['Ticket_count'] > 50].head(20)

print("\nTop 20 Agents by Ticket Volume (with >50 tickets):")
print(significant_agents)

# Plot agent performance matrix
plt.figure(figsize=(12, 8))
plt.scatter(
    significant_agents['Response time (hrs)'],
    significant_agents['CSAT Score'],
    s=significant_agents['Ticket_count']/5,  # Size proportional to ticket count
    alpha=0.6
)

# Add agent names as annotations
for idx, row in significant_agents.iterrows():
    plt.annotate(idx, 
                 (row['Response time (hrs)'], row['CSAT Score']),
                 fontsize=8)

plt.title('Agent Performance Matrix')
plt.xlabel('Average Response Time')
plt.ylabel('Average CSAT Score')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.grid(False)
plt.show()


# Summary Findings

print("\n==== SUMMARY OF FINDINGS ====")
print(f"Total records analyzed: {df.shape[0]}")
print(f"Average CSAT Score: {df['CSAT Score'].mean():.2f}")
print(f"Average Response Time: {df['Response time (hrs)'].mean():.2f} hours")
print(f"Percentage of High CSAT (≥4): {df['High_CSAT'].mean()*100:.2f}%")

# Top and bottom channels by CSAT
channel_csat = df.groupby('channel_name').agg({
    'CSAT Score': 'mean',
    'Unique id': 'count'
}).sort_values('CSAT Score', ascending=False)

print("\nTop 3 Channels by CSAT:")
print(channel_csat.head(3))

print("\nBottom 3 Channels by CSAT:")
print(channel_csat.tail(3))

print("\nML Model Performance:")
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_y_pred):.4f}")

print("\nAnalysis Complete!")






