import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocess Review Function
def preprocess_review(review):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    review = review.lower()
    tokens = word_tokenize(review, language='english')
    tokens = [word for word in tokens if word.isalpha()]  # Remove non-alphabetic tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Lemmatize and remove stopwords
    return ' '.join(tokens)

# Label Encoding Function
def label_rating(rating):
    if rating in [1, 2]:
        return 'Negative'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Positive'

# Load Dataset
df = pd.read_csv('reviews.csv', encoding='latin1')

# Handle missing values
df['Review Text'] = df['Review Text'].fillna('')  # Replace NaN with empty strings

# Preprocess Reviews
df['cleaned_review'] = df['Review Text'].apply(preprocess_review)

# Encode Labels
df['label'] = df['Rating'].apply(label_rating)

# Feature Extraction using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cleaned_review'])

# Split the Dataset into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)

# Train Logistic Regression Model (with class_weight for imbalanced classes)
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Predict on the Test Data
y_pred = model.predict(X_test)

# Evaluation Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision (per class):", precision_score(y_test, y_pred, average=None))
print("Recall (per class):", recall_score(y_test, y_pred, average=None))
print("F1-Score (per class):", f1_score(y_test, y_pred, average=None))

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix (shown, not saved)
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# ROC Curve (for multiclass classification)
y_test_mapped = y_test.map({'Negative': 0, 'Neutral': 1, 'Positive': 2})
y_pred_binarized = label_binarize(y_pred, classes=['Negative', 'Neutral', 'Positive'])
y_test_binarized = label_binarize(y_test_mapped, classes=[0, 1, 2])

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test_binarized.ravel(), y_pred_binarized.ravel())
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# Rating Distribution (Pie Chart)
rating_distribution = df['Rating'].value_counts().reset_index()
rating_distribution.columns = ['Rating', 'Count']

pie_chart = px.pie(rating_distribution, values='Count', names='Rating', title='Rating Distribution')
pie_chart.show()

# Sentiment Label Distribution (Bar Chart)
label_distribution = df['label'].value_counts().reset_index()
label_distribution.columns = ['Sentiment', 'Count']

bar_chart = px.bar(label_distribution, x='Sentiment', y='Count', title='Sentiment Label Distribution', text='Count')
bar_chart.show()

# Save the model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
