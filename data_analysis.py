import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import nltk

# Ensure the VADER lexicon is available
nltk.download('vader_lexicon')

# Load data from CSV
def load_data(csv_path):
    return pd.read_csv(csv_path)

# Compute sentiment scores using VADER
def sentiment_scores(dataframe):
    sia = SentimentIntensityAnalyzer()
    scores = dataframe['caption'].apply(lambda caption: sia.polarity_scores(caption)['compound'])
    return scores

# Plot and save histogram of sentiment scores
def plot_histogram(scores, plot_path):
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=50, color='blue', alpha=0.7)
    plt.title('Histogram of VADER Sentiment Intensity Scores')
    plt.xlabel('Sentiment Intensity Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Save the plot to a file
    plt.savefig(plot_path)
    plt.close()  # Close the plot to free up memory
    print(f"Plot saved to {plot_path}")

# Function to process each dataset
def process_dataset(csv_path, plot_path):
    data = load_data(csv_path)
    scores = sentiment_scores(data)
    plot_histogram(scores, plot_path)

# Main execution function
def main():
    datasets = {
        'MS-COCO': ('Data/coco_captions.csv', 'Results/coco_sentiment_histogram.png'),
        'Flickr30k': ('Data/flickr30k_captions.csv', 'Results/flickr30k_sentiment_histogram.png'),
        'Conceptual Captions': ('Data/conceptual_captions.csv', 'Results/conceptual_captions_histogram.png'),
        
        #'LAION': ('Data/laion_captions.csv', 'Results/laion_histogram.png'),
        #'TextCaps': ('Data/textcaps_captions.csv', 'Results/textcaps_histogram.png')
    }
    
    for dataset_name, (csv_path, plot_path) in datasets.items():
        print(f"Processing {dataset_name}...")
        process_dataset(csv_path, plot_path)

if __name__ == "__main__":
    main()
