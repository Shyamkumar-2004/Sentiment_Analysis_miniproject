import pandas as pd
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import plotly.express as px
import plotly.io as pio
from wordcloud import WordCloud
import os
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the saved model and vectorizer
model = joblib.load('model/sentiment_model.pkl')
tfidf = joblib.load('model/tfidf_vectorizer.pkl')

# Preprocess Review Function
def preprocess_review(review):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    review = review.lower()
    tokens = word_tokenize(review, language='english')
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Function to predict sentiment of a new review
def predict_sentiment(review):
    cleaned_review = preprocess_review(review)
    review_vector = tfidf.transform([cleaned_review])
    prediction = model.predict(review_vector)
    return prediction[0]



# Function to handle CSV or TXT file, predict sentiment, and generate analysis report
def generate_report(filepath):
    file_ext = os.path.splitext(filepath)[1].lower()

    if file_ext == '.csv':
        df = pd.read_csv(filepath)
    elif file_ext == '.txt':
        with open(filepath, 'r', encoding='utf-8') as file:
            data = file.readlines()
        df = pd.DataFrame(data, columns=['Review Text'])
    else:
        raise ValueError("Unsupported file type")

    if 'Review Text' not in df.columns:
        raise ValueError("File must contain a 'Review Text' column")

    # Predict sentiment
    df['Predicted Sentiment'] = df['Review Text'].apply(predict_sentiment)

    # Generate sentiment distribution pie chart
    sentiment_counts = df['Predicted Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    pie_chart = px.pie(sentiment_counts, values='Count', names='Sentiment', title='Sentiment Distribution')
    pie_chart_html = pio.to_html(pie_chart, full_html=False)

     # Generate sentiment counts bar chart
    bar_chart = px.bar(sentiment_counts, x='Sentiment', y='Count', title='Sentiment Counts', text='Count')
    bar_chart_html = pio.to_html(bar_chart, full_html=False)

    # Average Review Length Comparison (Positive vs. Negative)
    df['Review Length'] = df['Review Text'].apply(lambda x: len(x.split()))
    avg_length_chart = px.bar(df.groupby('Predicted Sentiment')['Review Length'].mean().reset_index(),
                              x='Predicted Sentiment', y='Review Length', title='Average Review Length (Positive vs. Negative)',
                              text='Review Length')
    avg_length_chart_html = pio.to_html(avg_length_chart, full_html=False)

    # Return the full analysis report
    analysis_report = {
        'pie_chart_html': pie_chart_html,
        'bar_chart_html': bar_chart_html,
        'avg_length_chart_html': avg_length_chart_html,
    }

    return analysis_report
