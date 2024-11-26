from datasets import load_dataset
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download
import kaggle
import pandas as pd
import io
# Download the VADER lexicon for sentiment analysis
download('vader_lexicon')

# Load the COCO dataset from Hugging Face
def load_dataset(owner,dataset_name, file_name):
    api = kaggle.KaggleApi()
    api.authenticate()
    file_stream = api.datasets_download_file(owner,dataset_name,file_name)
    data = pd.read_csv(io.BytesIO(file_stream.content))
    return data

# Compute sentiment scores using VADER
def compute_sentiment_scores(captions):
    sia = SentimentIntensityAnalyzer()
    scores = [sia.polarity_scores(caption)['compound'] for caption in captions]
    return scores

# Plot the histogram of sentiment scores
def plot_histogram(scores):
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=50, alpha=0.75, color='blue')
    plt.title('Histogram of VADER Sentiment Scores for COCO Captions')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# Execute the functions
if __name__ == "__main__":
    dataset_name = 'awsaf49/coco-2017-dataset'
    file_name = 'annotations/captions_train2017.json' 
    owner = 'awsaf' 
    data = load_dataset(owner,dataset_name, file_name)
    print(data.head())
    #sentiment_scores = compute_sentiment_scores(captions)
    #plot_histogram(sentiment_scores)
