# Movie Recommendation Metadata

## Overview
This folder contains a movie recommendation system project that analyzes movie metadata and user ratings to generate personalized movie recommendations. The project implements recommendation algorithms using collaborative filtering and content-based filtering techniques.

## Contents

### Files
- **movie_recommendation_notebook.ipynb** - Jupyter notebook implementing recommendation algorithms, model training, and analysis
- **movies_metadata.csv** - Movie metadata including titles, genres, release dates, and content information
- **ratings.csv** - User ratings dataset containing user-movie pairs and rating scores (1.6MB+)

## Purpose
This project builds a movie recommendation system to:
- Analyze movie metadata and characteristics
- Process user ratings and preferences
- Implement collaborative filtering algorithms
- Develop content-based filtering approaches
- Generate personalized movie recommendations
- Evaluate recommendation quality and accuracy

## Technology Stack
- **Language**: Python (Jupyter Notebook)
- **Libraries**: pandas, numpy, scikit-learn, scipy (for recommendation algorithms)
- **Algorithms**: Collaborative filtering, content-based filtering, similarity metrics
- **Data Formats**: CSV

## Getting Started

### Using Jupyter Notebook
1. Ensure Jupyter notebook is installed
2. Open `movie_recommendation_notebook.ipynb`
3. The notebook will load both CSV files
4. Run cells sequentially to train models and generate recommendations

## Datasets

### movies_metadata.csv
Contains information about movies:
- Movie ID and title
- Genres and keywords
- Release date
- Runtime and budget
- Ratings and popularity

### ratings.csv
Contains user ratings data:
- User IDs
- Movie IDs
- Rating scores (typically 1-5 scale)
- Timestamps of ratings

## Key Features
- **Collaborative Filtering** - User-based and item-based approaches
- **Content-Based Filtering** - Movie similarity based on genres and features
- **Hybrid Recommendations** - Combines multiple recommendation approaches
- **Performance Metrics** - Evaluates recommendation accuracy
- **Data Visualization** - Charts and graphs of findings
- **Personalization** - Tailored recommendations for different users

## Typical Analyses
- User preference patterns
- Movie similarity analysis
- Rating distributions
- Recommendation accuracy evaluation
- Top-rated movies identification
- Genre popularity trends
- User clustering and segmentation

---
*Generated on 2026-05-08*
