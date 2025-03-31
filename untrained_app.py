import streamlit as st
import pandas as pd
import re
import emoji
from apify_client import ApifyClient
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import io
import csv
import chardet
from dotenv import load_dotenv
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Download required NLTK resources if not already present
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
vader = SentimentIntensityAnalyzer()

# Initialize untrained MNB pipeline
mnb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('mnb', MultinomialNB())
])

# Load API keys
load_dotenv()
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")

# Initialize Apify Client
client = ApifyClient(APIFY_API_TOKEN)

# Streamlit App Configuration
st.set_page_config(
    page_title="Pure Untrained Sentiment Analysis", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.title("Pure Untrained Sentiment Analysis")
st.caption("Basic Sentiment Analysis Dashboard")

# Function to clean text
def clean_text(text):
    """Basic text cleaning."""
    if not isinstance(text, str):
        text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# Function to preprocess text
def preprocess_text(text):
    """Cleans and processes text for sentiment analysis."""
    if not isinstance(text, str):
        text = str(text)
    
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    
    # Extract and save emojis before removing them
    emojis_found = ''.join(c for c in text if c in emoji.EMOJI_DATA)
    
    # Convert emojis to text for sentiment analysis
    text_with_emoji_names = emoji.demojize(text, delimiters=(" ", " "))
    
    # Clean text for general analysis
    clean_version = clean_text(text_with_emoji_names)
    
    return {
        'cleaned_text': clean_version,
        'emojis': emojis_found,
        'demojized': text_with_emoji_names
    }

# Function to analyze sentiment using VADER
def analyze_sentiment_vader(text):
    """Analyze sentiment using VADER."""
    if not isinstance(text, str):
        text = str(text)
    
    # Get VADER sentiment scores
    scores = vader.polarity_scores(text)
    
    # Determine sentiment category
    if scores['compound'] >= 0.05:
        sentiment = "Positive"
        score = scores['compound']
    elif scores['compound'] <= -0.05:
        sentiment = "Negative"
        score = scores['compound']
    else:
        sentiment = "Neutral"
        score = scores['compound']
    
    return f"{sentiment} ({score:.2f})"

# Function to analyze sentiment using MNB (untrained)
def analyze_sentiment_mnb(text, processed_texts=None, vader_sentiments=None):
    """Analyze sentiment using untrained MNB that learns from VADER in real-time."""
    if not isinstance(text, str):
        text = str(text)
    
    try:
        # If we have processed texts and VADER sentiments, use them to train the model
        if processed_texts is not None and vader_sentiments is not None:
            # Convert VADER sentiments to labels
            labels = [s.split(' ')[0] for s in vader_sentiments]
            
            # Train the model on the current batch
            mnb_pipeline.fit(processed_texts, labels)
        
        # Get prediction for the current text
        prediction = mnb_pipeline.predict([text])[0]
        
        # Get prediction probabilities
        probs = mnb_pipeline.predict_proba([text])[0]
        max_prob = max(probs)
        
        return f"{prediction} ({max_prob:.2f})"
    except Exception as e:
        # If the pipeline is not fitted or there's an error, return VADER's sentiment
        vader_sentiment = analyze_sentiment_vader(text)
        return f"VADER Fallback: {vader_sentiment}"

# Function to extract hashtags
def extract_hashtags(text):
    """Extract hashtags from text."""
    if not isinstance(text, str):
        text = str(text)
    return re.findall(r'#\w+', text)

# Function to create a wordcloud
def create_wordcloud(text_series):
    """Create a WordCloud from a series of texts."""
    all_text = ' '.join(text_series.fillna(''))
    
    # Generate wordcloud
    wordcloud = WordCloud(
        width=800, 
        height=400,
        background_color='white',
        max_words=100,
        contour_width=1
    ).generate(all_text)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    return fig

# Function to plot sentiment distribution
def plot_sentiment_distribution(df, sentiment_column):
    """Create a bar chart of sentiment distribution."""
    # Extract sentiment categories without scores
    categories = df[sentiment_column].apply(lambda x: x.split(' ')[0])
    
    # Count occurrences
    counts = categories.value_counts()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'}
    
    sns.barplot(x=counts.index, y=counts.values, palette=[colors.get(cat, 'blue') for cat in counts.index], ax=ax)
    ax.set_title('Sentiment Distribution')
    ax.set_ylabel('Count')
    ax.set_xlabel('Sentiment')
    
    # Add percentage labels
    total = counts.sum()
    for i, count in enumerate(counts):
        percentage = 100 * count / total
        ax.text(i, count + 5, f'{percentage:.1f}%', ha='center')
    
    return fig

# Function to fetch comments from TikTok
def fetch_tiktok_comments(video_link, max_comments=1000):
    """Fetches comments from a TikTok video using Apify."""
    run_input = {"postURLs": [video_link], "commentsPerPost": max_comments, "maxRepliesPerComment": 0}
    
    try:
        run = client.actor("BDec00yAmCm1QbMEI").call(run_input=run_input)
    except Exception as e:
        st.error(f"Error calling Apify actor: {e}")
        return None
    
    # Get items from dataset
    items = []
    try:
        items = [item for item in client.dataset(run["defaultDatasetId"]).iterate_items()]
    except Exception as e:
        st.error(f"Error fetching comments: {e}")
    
    # Create DataFrame
    if items:
        df = pd.DataFrame(items)
        # Select relevant columns if they exist
        columns = ['text']
        if 'likes' in df.columns:
            columns.append('likes')
        if 'username' in df.columns:
            columns.append('username')
        if 'created_at' in df.columns:
            columns.append('created_at')
            
        df = df[columns].rename(columns={'text': 'Comment'})
        return df
    return None

# Function to read files in multiple formats
def read_file_with_multiple_formats(uploaded_file):
    """Reads an uploaded file that could be either XLSX or CSV."""
    if uploaded_file is None:
        return None
    
    try:
        # Get file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Process based on file type
        if file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        elif file_extension == 'csv':
            # Read the file content as bytes to detect encoding
            file_content = uploaded_file.read()
            
            # Try different encodings
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'utf-16']
            df = None
            
            for encoding in encodings:
                try:
                    # Reset file pointer
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                # If all encodings fail, try with error handling
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='utf-8', errors='replace')
        else:
            st.error(f"Unsupported file format: {file_extension}. Please upload an XLSX or CSV file.")
            return None
        
        # Check for comment column and rename if necessary
        if "Comment" not in df.columns:
            # Common text column names to look for
            text_column_keywords = ['text', 'comment', 'message', 'content', 'post']
            
            # Try to find a suitable column based on name
            potential_columns = []
            for col in df.columns:
                if any(keyword in col.lower() for keyword in text_column_keywords):
                    potential_columns.append(col)
            
            if potential_columns:
                # Use the first matching column
                df = df.rename(columns={potential_columns[0]: 'Comment'})
                st.info(f"Renamed column '{potential_columns[0]}' to 'Comment'.")
            else:
                # If no suitable column found by name, look for string columns with content
                for col in df.columns:
                    if df[col].dtype == 'object':  # object type usually means strings
                        # Check if column has meaningful content
                        sample = df[col].dropna().astype(str).str.len().mean()
                        if sample > 5:  # If average text length is reasonable
                            df = df.rename(columns={col: 'Comment'})
                            st.info(f"No explicit comment column found. Using '{col}' as the Comment column.")
                            break
                
                # If we still don't have a Comment column
                if "Comment" not in df.columns:
                    st.error("Could not identify a suitable text column to use as 'Comment'.")
                    return None
        
        # Ensure Comment column has string values and handle any encoding issues
        df['Comment'] = df['Comment'].astype(str).apply(lambda x: x.encode('utf-8', errors='replace').decode('utf-8'))
        
        # Drop rows with empty comments
        df = df[df['Comment'].str.strip() != '']
        
        return df
    
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

# Create sidebar menu
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a section", ["Upload Data", "Fetch TikTok Comments", "Sentiment Explorer", "About"])

# About page
if page == "About":
    st.header("About Pure Untrained Sentiment Analysis")
    st.markdown("""
    This is a simplified version of the sentiment analysis tool that uses only rule-based sentiment analysis (VADER) and an untrained MNB classifier.
    
    ### Features:
    - Upload Excel files containing comments
    - Fetch comments directly from TikTok videos using Apify
    - Analyze sentiment using VADER (rule-based sentiment analysis)
    - Analyze sentiment using untrained MNB (learns from VADER in real-time)
    - Visualize sentiment distribution
    - Generate word clouds from comments
    - Extract hashtags and analyze trends
    
    ### How to Use:
    1. Navigate to "Upload Data" to analyze your own data
    2. Or go to "Fetch TikTok Comments" to analyze comments from a TikTok video URL
    3. Use "Sentiment Explorer" to understand how sentiment analysis works
    4. Review the analysis and visualizations
    
    ### Technologies Used:
    - NLTK VADER for rule-based sentiment analysis
    - Untrained MNB classifier that learns from VADER in real-time
    - Apify for data collection
    - Streamlit for the web interface
    - Plotly and Matplotlib for visualizations
    """)

# Upload section
elif page == "Upload Data":
    st.header("Upload Your Data File")
    
    # Update file uploader to accept both xlsx and csv
    uploaded_file = st.file_uploader("Upload a file containing comments", type=["xlsx", "xls", "csv"])
    
    if uploaded_file:
        # Display a spinner while processing
        with st.spinner("Reading and processing file..."):
            # Process the uploaded file
            comments_df = read_file_with_multiple_formats(uploaded_file)
            
            if comments_df is not None:
                st.success(f"File uploaded and processed successfully. Found {len(comments_df)} comments.")
                
                # Process comments
                with st.spinner("Analyzing comments..."):
                    # Process comments
                    processed_data = comments_df['Comment'].apply(preprocess_text)
                    
                    # Add processed text columns
                    comments_df['Processed Comment'] = processed_data.apply(lambda x: x['cleaned_text'])
                    comments_df['Emojis'] = processed_data.apply(lambda x: x['emojis'])
                    comments_df['Demojized'] = processed_data.apply(lambda x: x['demojized'])
                    
                    # Extract hashtags
                    comments_df['Hashtags'] = comments_df['Comment'].apply(extract_hashtags)
                    
                    # Apply VADER sentiment analysis
                    comments_df['VADER Sentiment'] = comments_df['Demojized'].apply(analyze_sentiment_vader)
                    
                    # Apply MNB sentiment analysis (untrained)
                    comments_df['MNB Sentiment'] = comments_df['Demojized'].apply(
                        lambda x: analyze_sentiment_mnb(
                            x,
                            comments_df['Demojized'].tolist(),
                            comments_df['VADER Sentiment'].tolist()
                        )
                    )
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["Data View", "Visualizations", "Statistics"])
                
                with tab1:
                    # Display data
                    st.subheader("Processed Comments")
                    st.dataframe(comments_df[['Comment', 'Processed Comment', 'VADER Sentiment', 'MNB Sentiment']])
                    
                    # Allow download of processed data
                    csv = comments_df.to_csv(index=False)
                    st.download_button(
                        label="Download processed data as CSV",
                        data=csv,
                        file_name="processed_comments.csv",
                        mime="text/csv",
                    )
                
                with tab2:
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Sentiment Distribution")
                        # Plot sentiment distribution
                        fig = plot_sentiment_distribution(comments_df, 'VADER Sentiment')
                        st.pyplot(fig)
                    
                    with col2:
                        st.subheader("Word Cloud")
                        fig = create_wordcloud(comments_df['Processed Comment'])
                        st.pyplot(fig)
                    
                    # Emoji analysis
                    st.subheader("Emoji Analysis")
                    all_emojis = ''.join(comments_df['Emojis'].fillna(''))
                    if all_emojis:
                        emoji_counter = Counter(all_emojis)
                        top_emojis = emoji_counter.most_common(10)
                        
                        emoji_df = pd.DataFrame(top_emojis, columns=['Emoji', 'Count'])
                        
                        # Create horizontal bar chart for emojis
                        fig, ax = plt.subplots(figsize=(10, 5))
                        sns.barplot(y=emoji_df['Emoji'], x=emoji_df['Count'], ax=ax, orient='h')
                        ax.set_title('Top 10 Emojis')
                        st.pyplot(fig)
                    else:
                        st.info("No emojis found in the comments.")
                
                with tab3:
                    # Statistics
                    st.subheader("Comment Statistics")
                    
                    # Basic stats
                    stats = {
                        "Total Comments": len(comments_df),
                        "Average Comment Length": int(comments_df['Comment'].apply(len).mean()),
                        "Comments with Emojis": len(comments_df[comments_df['Emojis'] != '']),
                        "Positive Comments": len(comments_df[comments_df['VADER Sentiment'].str.contains('Positive')]),
                        "Negative Comments": len(comments_df[comments_df['VADER Sentiment'].str.contains('Negative')]),
                        "Neutral Comments": len(comments_df[comments_df['VADER Sentiment'].str.contains('Neutral')])
                    }
                    
                    # Display stats in columns
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Comments", stats["Total Comments"])
                    col1.metric("Average Length", stats["Average Comment Length"])
                    col2.metric("Positive Comments", stats["Positive Comments"])
                    col2.metric("Negative Comments", stats["Negative Comments"])
                    col3.metric("Neutral Comments", stats["Neutral Comments"])
                    col3.metric("Comments with Emojis", stats["Comments with Emojis"])
                    
                    # Hashtag analysis
                    st.subheader("Hashtag Analysis")
                    all_hashtags = [tag for tags in comments_df['Hashtags'] for tag in tags]
                    if all_hashtags:
                        hashtag_counter = Counter(all_hashtags)
                        top_hashtags = hashtag_counter.most_common(15)
                        
                        hashtag_df = pd.DataFrame(top_hashtags, columns=['Hashtag', 'Count'])
                        
                        # Create horizontal bar chart for hashtags
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.barplot(y=hashtag_df['Hashtag'], x=hashtag_df['Count'], ax=ax, orient='h')
                        ax.set_title('Top 15 Hashtags')
                        st.pyplot(fig)
                    else:
                        st.info("No hashtags found in the comments.")

# TikTok Comment Fetching
elif page == "Fetch TikTok Comments":
    st.header("Fetch TikTok Comments")
    
    # Input for TikTok video link
    video_link = st.text_input("Enter TikTok video link:")
    col1, col2 = st.columns(2)
    max_comments = col1.number_input("Maximum comments to fetch:", min_value=10, max_value=2000, value=500)
    analyze_button = col2.button("Fetch and Analyze")
    
    if analyze_button:
        if video_link:
            with st.spinner("Fetching comments, please wait..."):
                comments_df = fetch_tiktok_comments(video_link, max_comments=max_comments)
                
                if comments_df is not None and not comments_df.empty:
                    st.success(f"Fetched {len(comments_df)} comments!")
                    
                    # Process comments
                    with st.spinner("Processing comments..."):
                        processed_data = comments_df['Comment'].apply(preprocess_text)
                        
                        # Add processed text columns
                        comments_df['Processed Comment'] = processed_data.apply(lambda x: x['cleaned_text'])
                        comments_df['Emojis'] = processed_data.apply(lambda x: x['emojis'])
                        comments_df['Demojized'] = processed_data.apply(lambda x: x['demojized'])
                        
                        # Extract hashtags
                        comments_df['Hashtags'] = comments_df['Comment'].apply(extract_hashtags)
                        
                        # Apply sentiment analysis
                        with st.spinner("Performing sentiment analysis..."):
                            comments_df['VADER Sentiment'] = comments_df['Demojized'].apply(analyze_sentiment_vader)
                            comments_df['MNB Sentiment'] = comments_df['Demojized'].apply(
                                lambda x: analyze_sentiment_mnb(
                                    x,
                                    comments_df['Demojized'].tolist(),
                                    comments_df['VADER Sentiment'].tolist()
                                )
                            )
                    
                    # Create tabs for different views
                    tab1, tab2, tab3 = st.tabs(["Data View", "Visualizations", "Statistics"])
                    
                    with tab1:
                        # Display data
                        st.subheader("Processed Comments")
                        st.dataframe(comments_df[['Comment', 'Processed Comment', 'VADER Sentiment', 'MNB Sentiment']])
                        
                        # Allow download of processed data
                        csv = comments_df.to_csv(index=False)
                        st.download_button(
                            label="Download processed data as CSV",
                            data=csv,
                            file_name="processed_comments.csv",
                            mime="text/csv",
                        )
                    
                    with tab2:
                        # Visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Sentiment Distribution")
                            # Plot sentiment distribution
                            fig = plot_sentiment_distribution(comments_df, 'VADER Sentiment')
                            st.pyplot(fig)
                        
                        with col2:
                            st.subheader("Word Cloud")
                            fig = create_wordcloud(comments_df['Processed Comment'])
                            st.pyplot(fig)
                        
                        # Emoji analysis
                        st.subheader("Emoji Analysis")
                        all_emojis = ''.join(comments_df['Emojis'].fillna(''))
                        if all_emojis:
                            emoji_counter = Counter(all_emojis)
                            top_emojis = emoji_counter.most_common(10)
                            
                            emoji_df = pd.DataFrame(top_emojis, columns=['Emoji', 'Count'])
                            
                            # Create horizontal bar chart for emojis
                            fig, ax = plt.subplots(figsize=(10, 5))
                            sns.barplot(y=emoji_df['Emoji'], x=emoji_df['Count'], ax=ax, orient='h')
                            ax.set_title('Top 10 Emojis')
                            st.pyplot(fig)
                        else:
                            st.info("No emojis found in the comments.")
                    
                    with tab3:
                        # Statistics
                        st.subheader("Comment Statistics")
                        
                        # Basic stats
                        stats = {
                            "Total Comments": len(comments_df),
                            "Average Comment Length": int(comments_df['Comment'].apply(len).mean()),
                            "Comments with Emojis": len(comments_df[comments_df['Emojis'] != '']),
                            "Positive Comments": len(comments_df[comments_df['VADER Sentiment'].str.contains('Positive')]),
                            "Negative Comments": len(comments_df[comments_df['VADER Sentiment'].str.contains('Negative')]),
                            "Neutral Comments": len(comments_df[comments_df['VADER Sentiment'].str.contains('Neutral')])
                        }
                        
                        # Display stats in columns
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Comments", stats["Total Comments"])
                        col1.metric("Average Length", stats["Average Comment Length"])
                        col2.metric("Positive Comments", stats["Positive Comments"])
                        col2.metric("Negative Comments", stats["Negative Comments"])
                        col3.metric("Neutral Comments", stats["Neutral Comments"])
                        col3.metric("Comments with Emojis", stats["Comments with Emojis"])
                        
                        # Hashtag analysis
                        st.subheader("Hashtag Analysis")
                        all_hashtags = [tag for tags in comments_df['Hashtags'] for tag in tags]
                        if all_hashtags:
                            hashtag_counter = Counter(all_hashtags)
                            top_hashtags = hashtag_counter.most_common(15)
                            
                            hashtag_df = pd.DataFrame(top_hashtags, columns=['Hashtag', 'Count'])
                            
                            # Create horizontal bar chart for hashtags
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sns.barplot(y=hashtag_df['Hashtag'], x=hashtag_df['Count'], ax=ax, orient='h')
                            ax.set_title('Top 15 Hashtags')
                            st.pyplot(fig)
                        else:
                            st.info("No hashtags found in the comments.")
                    
                else:
                    st.error("Failed to fetch comments. Please check the video link and try again.")
                
        else:
            st.warning("Please enter a TikTok video link.")

# Sentiment Explorer page
elif page == "Sentiment Explorer":
    st.header("Sentiment Analysis Explorer")
    
    st.write("""
    This section allows you to explore how our sentiment analysis works with individual comments.
    Enter a sample comment to see how VADER sentiment analysis evaluates it.
    """)
    
    # Input for testing sentiment
    test_comment = st.text_area("Enter a comment to analyze:", "This video is amazing! The tutorial was so helpful 🔥👍")
    
    if test_comment:
        # Create columns for results
        col1, col2 = st.columns(2)
        
        with col1:
            # Process the comment
            processed = preprocess_text(test_comment)
            
            # Display the processed text
            st.subheader("Processed Text")
            st.write(f"**Original:** {test_comment}")
            st.write(f"**Cleaned:** {processed['cleaned_text']}")
            st.write(f"**Emojis Found:** {processed['emojis'] or 'None'}")
            st.write(f"**Demojized:** {processed['demojized']}")
            
            # Extract and display hashtags
            hashtags = extract_hashtags(test_comment)
            if hashtags:
                st.write(f"**Hashtags:** {', '.join(hashtags)}")
            
            # Perform sentiment analysis
            vader_sentiment = analyze_sentiment_vader(processed['demojized'])
            mnb_sentiment = analyze_sentiment_mnb(processed['demojized'])
            
            # Display sentiment results
            st.subheader("Sentiment Analysis")
            st.write(f"**VADER:** {vader_sentiment}")
            st.write(f"**MNB:** {mnb_sentiment}")
        
        with col2:
            # Add explanation
            st.subheader("How It Works")
            st.write("""
            Our sentiment analysis uses two methods:
            
            1. **VADER (Valence Aware Dictionary and sEntiment Reasoner)**
               - Rule-based sentiment analysis
               - No training required
               - Handles social media language patterns
               - Considers word order and context
            
            2. **Untrained MNB (Multinomial Naive Bayes)**
               - Learns from VADER's output in real-time
               - No pre-trained data
               - Provides an alternative perspective
               - Resets knowledge for each new batch
            
            ### Sentiment Scoring
            - Positive: compound score >= 0.05
            - Negative: compound score <= -0.05
            - Neutral: compound score between -0.05 and 0.05
            """)

# Run the app
if __name__ == "__main__":
    pass 
