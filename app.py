import streamlit as st
import pandas as pd
import re
import emoji
from apify_client import ApifyClient
from sentiment_analysis import (
    TrollDetector,
    analyze_sentiment_vader, 
    train_mnb_model, 
    combined_sentiment_analysis,
    enhanced_sentiment_analysis,
    get_sentiment_breakdown,
    analyze_for_trolling
)
from database import db  # Add this import
# Import Tagalog sentiment functions
from tagalog_sentiment import (
    is_tagalog,
    tagalog_enhanced_sentiment_analysis,
    get_tagalog_sentiment_breakdown
)
from market_trend_analysis import (
    detect_purchase_intent,
    calculate_market_trend_score,
    predict_purchase_volume,
    plot_market_prediction,
    generate_market_trend_report,
    add_market_trends_tab
)
from text_processing import clean_text, tokenize_and_remove_stopwords, extract_hashtags
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from collections import Counter
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import io
import csv
import chardet  # You may need to pip install chardet
import matplotlib.font_manager as fm

# Define global data directory
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True) 
troll_detector = TrollDetector()


# Load API keys
load_dotenv()
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")

# Initialize Apify Client
client = ApifyClient(APIFY_API_TOKEN)

# Streamlit App Configuration
st.set_page_config(
    page_title="TikTok Sentiment Analysis", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.title("TikTok Sentiment Analysis")
st.caption("Market Trend Classification Dashboard")

# After the imports and before the main app code
def create_emoji_chart(emoji_counts):
    """Create a horizontal bar chart for emoji counts with proper emoji display."""
    # Set font that supports emojis
    plt.rcParams['font.family'] = ['Segoe UI Emoji', 'Segoe UI Symbol', 'Apple Color Emoji', 'Noto Color Emoji', 'Noto Emoji']
    
    # Create figure with higher DPI for better emoji rendering
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # Sort emojis by count and get top 10
    top_emojis = emoji_counts.most_common(10)
    
    # Separate emojis and counts
    emojis = [e for e, _ in top_emojis]
    counts = [count for _, count in top_emojis]
    
    # Create horizontal bar chart
    bars = ax.barh(range(len(emojis)), counts, color='#2196F3')
    
    # Set emoji labels with increased size
    ax.set_yticks(range(len(emojis)))
    ax.set_yticklabels(emojis, fontsize=16, fontfamily='Segoe UI Emoji')
    
    # Customize chart
    ax.set_title('Top 10 Emojis', pad=20, fontsize=14)
    ax.set_xlabel('Count', fontsize=12)
    
    # Remove frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add count labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f'{int(width)}', 
                ha='left', va='center', fontsize=12,
                fontfamily='sans-serif')  # Use standard font for numbers
    
    # Adjust layout to prevent emoji cutoff
    plt.subplots_adjust(left=0.2)
    plt.tight_layout()
    return fig

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

# Functions for language-aware sentiment analysis
def add_language_settings():
    """Add language settings to the sidebar."""
    st.sidebar.header("Language Settings")
    language_mode = st.sidebar.radio(
        "Select language analysis mode:",
        ["Auto-detect", "English Only", "Tagalog Only", "Multilingual (English + Tagalog)"],
        index=0
    )
    
    st.sidebar.markdown("""
    **Language Mode Information:**
    - **Auto-detect**: Automatically identifies language and applies appropriate analysis
    - **English Only**: Optimized for English TikTok comments
    - **Tagalog Only**: Optimized for Filipino/Tagalog comments
    - **Multilingual**: Best for mixed language content or code-switching
    """)
    
    # Store language preference in session state
    st.session_state.language_mode = language_mode
    
    return language_mode

def analyze_sentiment_with_language_preference(text, language_mode=None):
    """
    Analyze sentiment with language mode preference.
    
    Args:
        text: Text to analyze
        language_mode: Language mode preference
        
    Returns:
        Sentiment analysis result
    """
    if language_mode is None:
        language_mode = st.session_state.get('language_mode', "Auto-detect")
    
    if language_mode == "Auto-detect":
        # Auto-detect language and apply appropriate analysis
        if is_tagalog(text):
            return tagalog_enhanced_sentiment_analysis(text)
        else:
            return enhanced_sentiment_analysis(text)  # Your existing function
    
    elif language_mode == "English Only":
        # Force English analysis regardless of language
        return enhanced_sentiment_analysis(text)  # Your existing function
    
    elif language_mode == "Tagalog Only":
        # Force Tagalog analysis regardless of language
        return tagalog_enhanced_sentiment_analysis(text)
    
    else:  # Multilingual mode
        # Always use the multilingual analyzer
        return tagalog_enhanced_sentiment_analysis(text)
    
def analyze_comment_with_trolling(text, language_mode=None):
    """
    Analyzes comment for both sentiment and troll detection.
    
    Args:
        text: Text to analyze
        language_mode: Language mode preference
        
    Returns:
        Dictionary with sentiment and troll information
    """
    # Get standard sentiment analysis with language preference
    sentiment = analyze_sentiment_with_language_preference(text, language_mode)
    
    # Get troll analysis
    troll_analysis = analyze_for_trolling(text)
    
    # Format result as: "Sentiment (score) [TROLL]" if it's a troll
    result = sentiment
    if troll_analysis['is_troll']:
        # Extract the existing sentiment part
        sentiment_part = sentiment.split(' (')[0]
        score_part = sentiment.split('(')[1]
        
        # Add the troll marker
        result = f"{sentiment_part} (TROLL) ({score_part}"
    
    return {
        'sentiment_text': result,
        'is_troll': troll_analysis['is_troll'],
        'troll_score': troll_analysis['troll_score'],
        'language': troll_analysis['language']
    }

def refresh_database_state():
    """Refresh the database state in the session."""
    st.session_state.sentiment_corrections = db.get_corrections()
    st.session_state.last_refresh_time = pd.Timestamp.now()

def save_sentiment_correction(comments_df, selected_comment_idx, corrected_sentiment):
    """
    Saves a corrected sentiment using the database system.
    
    Args:
        comments_df: The dataframe with comments data
        selected_comment_idx: Index of the selected comment
        corrected_sentiment: The corrected sentiment value
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Update the dataframe
        comments_df.loc[selected_comment_idx, 'Combined Sentiment'] = f"{corrected_sentiment} (1.00)"
        
        # Get the comment and detect language
        comment = comments_df.loc[selected_comment_idx, 'Comment']
        language = 'tagalog' if is_tagalog(comment) else 'english'
        
        # Save to database
        success = db.save_correction(
            comment=comment,
            corrected_sentiment=corrected_sentiment,
            language=language
        )
        
        if success:
            # Refresh the database state immediately
            refresh_database_state()
            st.success(f"Sentiment correction saved: {comment} -> {corrected_sentiment}")
        
        return success
    except Exception as e:
        st.error(f"Error saving correction: {e}")
        return False


def get_sentiment_breakdown_with_language(text, language_mode=None):
    """
    Get sentiment breakdown with language preference.
    
    Args:
        text: Text to analyze
        language_mode: Language mode preference
        
    Returns:
        Sentiment breakdown dictionary
    """
    if language_mode is None:
        language_mode = st.session_state.get('language_mode', "Auto-detect")
    
    if language_mode == "Auto-detect":
        # Auto-detect language and apply appropriate breakdown
        if is_tagalog(text):
            return get_tagalog_sentiment_breakdown(text)
        else:
            return get_sentiment_breakdown(text)  # Your existing function
    
    elif language_mode == "English Only":
        # Force English breakdown
        return get_sentiment_breakdown(text)  # Your existing function
    
    elif language_mode == "Tagalog Only":
        # Force Tagalog breakdown
        return get_tagalog_sentiment_breakdown(text)
    
    else:  # Multilingual mode
        # Always use the tagalog breakdown which includes multilingual capabilities
        return get_tagalog_sentiment_breakdown(text)

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

# Function to read files in multiple formats (XLSX and CSV)
def read_file_with_multiple_formats(uploaded_file):
    """
    Reads an uploaded file that could be either XLSX or CSV with various formats.
    
    Parameters:
    uploaded_file (UploadedFile): The file uploaded through Streamlit's file_uploader
    
    Returns:
    pandas.DataFrame: DataFrame containing the file data, or None if processing failed
    """
    if uploaded_file is None:
        return None
    
    try:
        # Get file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Process based on file type
        if file_extension in ['xlsx', 'xls']:
            # For Excel files
            try:
                # Try standard pandas excel reader
                df = pd.read_excel(uploaded_file)
            except Exception as excel_error:
                st.warning(f"Standard Excel reader failed: {excel_error}. Trying alternative engines...")
                
                try:
                    # Try with openpyxl engine for .xlsx files
                    df = pd.read_excel(uploaded_file, engine='openpyxl')
                except Exception:
                    # Try with xlrd engine for .xls files
                    try:
                        df = pd.read_excel(uploaded_file, engine='xlrd')
                    except Exception as e:
                        st.error(f"Failed to read Excel file with all available engines: {e}")
                        return None
        
        elif file_extension == 'csv':
            # For CSV files - Improved approach for encoding issues
            
            # Read the file content as bytes to detect encoding
            file_content = uploaded_file.read()
            
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Try common encodings in order of likelihood
            encodings_to_try = [
                'utf-8',
                'latin1',
                'iso-8859-1',
                'cp1252',
                'utf-16',
                'utf-32',
                'utf-8-sig'  # UTF-8 with BOM
            ]
            
            # Remove None or invalid encodings
            encodings_to_try = [enc for enc in encodings_to_try if enc]
            
            # Try each encoding
            for encoding in encodings_to_try:
                try:
                    # Reset file pointer each time
                    uploaded_file.seek(0)
                    
                    # Try with comma delimiter
                    df = pd.read_csv(
                        uploaded_file, 
                        encoding=encoding, 
                        on_bad_lines='warn',  # More lenient with bad lines
                        low_memory=False      # Better for mixed data types
                    )
                    st.success(f"Successfully read CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    # If we get a unicode error, try next encoding
                    continue
                except Exception as e:
                    # For other exceptions, try different delimiters with this encoding
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(
                            uploaded_file, 
                            sep=None,  # Try to auto-detect separator
                            engine='python', 
                            encoding=encoding
                        )
                        st.success(f"Successfully read CSV with {encoding} encoding and auto-detected delimiter")
                        break
                    except Exception:
                        try:
                            uploaded_file.seek(0)
                            df = pd.read_csv(
                                uploaded_file, 
                                sep='\t', 
                                encoding=encoding
                            )
                            st.success(f"Successfully read CSV with {encoding} encoding and tab delimiter")
                            break
                        except Exception:
                            try:
                                uploaded_file.seek(0)
                                df = pd.read_csv(
                                    uploaded_file, 
                                    sep=';', 
                                    encoding=encoding
                                )
                                st.success(f"Successfully read CSV with {encoding} encoding and semicolon delimiter")
                                break
                            except Exception:
                                # Last resort: try to manually read the file
                                if encoding == encodings_to_try[-1]:
                                    try:
                                        uploaded_file.seek(0)
                                        # Try to read as binary and decode manually
                                        raw_content = uploaded_file.read()
                                        # Replace or ignore problematic bytes
                                        text_content = raw_content.decode(encoding, errors='replace')
                                        
                                        # Use StringIO to create a file-like object
                                        import io
                                        string_data = io.StringIO(text_content)
                                        
                                        # Try reading with all possible delimiters
                                        for delimiter in [',', '\t', ';']:
                                            try:
                                                string_data.seek(0)
                                                df = pd.read_csv(string_data, sep=delimiter)
                                                st.success(f"Successfully read CSV with manual decoding and {delimiter} delimiter")
                                                break
                                            except Exception:
                                                continue
                                        else:
                                            st.error(f"Failed to read CSV file with all available delimiters after manual decoding")
                                            return None
                                    except Exception as e:
                                        st.error(f"Failed to read CSV with all encodings: {e}")
                                        return None
            else:
                # If we've tried all encodings and still failed
                st.error("Failed to read CSV file with all available encodings")
                return None
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
        
        # Ensure Comment column has string values
        df['Comment'] = df['Comment'].astype(str)
        
        # Drop rows with empty comments
        df = df[df['Comment'].str.strip() != '']
        
        return df
    
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

# Text Preprocessing Enhancements
def preprocess_text(text):
    """Cleans and processes text for better sentiment analysis."""
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

# Function to analyze sentiment distribution
def plot_sentiment_distribution(df, sentiment_column):
    """Create a bar chart of sentiment distribution."""
    # Extract sentiment categories without scores
    categories = df[sentiment_column].apply(lambda x: x.split(' ')[0] if isinstance(x, str) else x)
    
    # Count occurrences
    counts = categories.value_counts()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'}
    
    # Create bar plot
    bars = ax.bar(range(len(counts)), counts.values, color=[colors.get(cat, '#3498db') for cat in counts.index])
    
    # Customize chart
    ax.set_title('Sentiment Distribution', pad=20)
    ax.set_ylabel('Count')
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index)
    
    # Remove frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add percentage labels
    total = counts.sum()
    for i, count in enumerate(counts):
        percentage = 100 * count / total
        ax.text(i, count + (max(counts) * 0.02), 
                f'{percentage:.1f}%', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

# Create an interactive heatmap for sentiment comparison
def create_sentiment_heatmap(df):
    """Create a heatmap comparing different sentiment analysis methods."""
    # Make sure we have all the sentiment columns
    required_columns = ['VADER Sentiment', 'MNB Sentiment', 'Combined Sentiment', 'Enhanced Sentiment']
    if not all(col in df.columns for col in required_columns):
        return None
    
    # Extract just the sentiment label (not the score)
    sentiment_matrix = pd.DataFrame()
    for col in required_columns:
        sentiment_matrix[col] = df[col].apply(lambda x: x.split(' ')[0])
    
    # Calculate agreement matrix
    agreement_matrix = pd.DataFrame(index=['Positive', 'Neutral', 'Negative'], 
                                  columns=['Positive', 'Neutral', 'Negative'])
    
    # Fill the matrix with zeros
    agreement_matrix = agreement_matrix.fillna(0)
    
    # Count agreements for the enhanced sentiment
    for idx, row in sentiment_matrix.iterrows():
        enhanced = row['Combined Sentiment']
        
        # Count VADER agreements
        vader = row['VADER Sentiment']
        agreement_matrix.at[enhanced, vader] += 1
        
        # Count MNB agreements
        mnb = row['MNB Sentiment']
        agreement_matrix.at[enhanced, mnb] += 1
        
        # Count Combined agreements
        combined = row['Combined Sentiment']
        agreement_matrix.at[enhanced, combined] += 1
    
    # Create the heatmap
    fig = px.imshow(agreement_matrix, 
                    labels=dict(x="Other Methods", y="Combined Sentiment", color="Agreement Count"),
                    x=agreement_matrix.columns,
                    y=agreement_matrix.index,
                    color_continuous_scale='Viridis')
    
    fig.update_layout(title="Sentiment Analysis Method Agreement")
    
    return fig

# Function to plot sentiment breakdown
def plot_sentiment_factors(comment, breakdown=None):
    """Create a plot showing the factors contributing to sentiment score."""
    if breakdown is None:
        breakdown = get_sentiment_breakdown_with_language(comment)
    
    # Create data for the plot
    if 'tagalog' in breakdown and breakdown['tagalog'] != 0:
        # This is a Tagalog or mixed language comment
        factors = ['VADER', 'ML Model', 'Emoji', 'Tagalog']
        values = [breakdown['vader'], breakdown['multilingual'], breakdown['emoji'], breakdown['tagalog']]
    else:
        # This is an English comment
        factors = ['VADER', 'ML Model', 'Emoji', 'Lexicon']
        values = [breakdown['vader'], breakdown['multilingual'] if 'multilingual' in breakdown else breakdown['ml'], 
                 breakdown['emoji'], breakdown['lexicon']]
    
    # Normalize values to -1 to 1 range
    colors = ['green' if v > 0 else 'red' if v < 0 else 'gray' for v in values]
    
    # Create the bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=factors,
        y=values,
        marker_color=colors
    ))
    
    fig.update_layout(
        title=f"Sentiment Components for: '{comment[:50]}...'",
        xaxis_title="Analysis Component",
        yaxis_title="Sentiment Score (-1 to 1)",
        yaxis=dict(range=[-1, 1])
    )
    
    # Add a line for the final score
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=breakdown['final'],
        x1=3.5,
        y1=breakdown['final'],
        line=dict(
            color="blue",
            width=2,
            dash="dash",
        )
    )
    
    fig.add_annotation(
        x=3.5,
        y=breakdown['final'],
        text=f"Final Score: {breakdown['final']:.2f}",
        showarrow=False,
        yshift=10
    )
    
    return fig

# About page Tagalog information
TAGALOG_ABOUT_TEXT = """
### Filipino/Tagalog Language Support

This application now supports sentiment analysis for:
- English language comments
- Filipino/Tagalog language comments
- Code-switching (mix of English and Tagalog)
- Regional Filipino dialects (Bisaya, Ilokano, Bikolano, etc.)
- Social media slang and TikTok-specific expressions in Tagalog

The application now features troll detection capabilities:
- Identifies comments with troll-like behavior patterns
- Analyzes language patterns specific to trolling
- Detects excessive punctuation, ALL CAPS, and inflammatory language
- Provides a troll score for each comment
- Specialized detection for Filipino/Taglish troll comments

You can set your language preference in the sidebar under "Language Settings".
"""

# Create sidebar menu
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a section", ["Upload Data", "Fetch TikTok Comments", "Sentiment Explorer", "Database Corrections", "About"])

# Add language settings
language_mode = add_language_settings()
# Page navigation logic
if page == "Upload Data":
    st.header("Upload Your Data File")
    
    # Update file uploader to accept both xlsx and csv
    uploaded_file = st.file_uploader("Upload a file containing TikTok comments", type=["xlsx", "xls", "csv"])
    
    if uploaded_file:
        # Display a spinner while processing
        with st.spinner("Reading and processing file..."):
            # Process the uploaded file
            comments_df = read_file_with_multiple_formats(uploaded_file)
            
            if comments_df is not None:
                # Check if we have enough comments
                if len(comments_df) < 100:
                    st.error(f"""
                    ⚠️ Insufficient comments for accurate analysis. Found only {len(comments_df)} comments.
                    
                    For reliable sentiment analysis, we need at least 100 comments. Please:
                    1. Upload a file with more comments (minimum 100)
                    2. Collect more comments before analysis
                    3. Try analyzing a different dataset
                    
                    Having enough comments ensures more accurate and meaningful results.
                    """)
                else:
                    st.success(f"File uploaded and processed successfully. Found {len(comments_df)} comments.")
                    
                    # Only proceed with analysis if we have enough comments
                    if len(comments_df) >= 100:
                        # Refresh database state when new file is uploaded
                        refresh_database_state()
                        
                        # Continue with your existing processing pipeline
                        with st.spinner("Analyzing comments..."):
                            # Process comments with your existing functions
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
                                
                                try:
                                    comments_df['MNB Sentiment'] = train_mnb_model(comments_df['Processed Comment'])
                                except Exception as e:
                                    st.warning(f"Error with MNB model: {e}. Using VADER only.")
                                    comments_df['MNB Sentiment'] = "N/A"
                                
                                # Apply combined sentiment analysis
                                comments_df['Combined Sentiment'] = combined_sentiment_analysis(comments_df['Demojized'])
                                
                                # Add troll detection
                                comments_df['Troll Score'] = comments_df['Comment'].apply(
                                    lambda x: analyze_for_trolling(x)['troll_score']
                                )
                                comments_df['Is Troll'] = comments_df['Troll Score'].apply(
                                    lambda x: x > 0.5
                                )
                                
                                # Add troll indicator to sentiment
                                comments_df['Combined Sentiment'] = comments_df.apply(
                                    lambda row: row['Combined Sentiment'] + " (TROLL)" if row['Is Troll'] else row['Combined Sentiment'],
                                    axis=1
                                )
                                
                                troll_results = comments_df['Comment'].apply(
                                    lambda text: analyze_comment_with_trolling(text, language_mode)
                                )
                                comments_df['Enhanced Sentiment'] = troll_results.apply(lambda x: x['sentiment_text'])
                                comments_df['Is Troll'] = troll_results.apply(lambda x: x['is_troll'])
                                
                                # Extract confidence scores from Enhanced Sentiment with higher base confidence
                                comments_df['Confidence'] = comments_df['Enhanced Sentiment'].apply(
                                    lambda x: float(re.search(r'\(([-+]?\d+\.\d+)\)', x).group(1)) if re.search(r'\(([-+]?\d+\.\d+)\)', x) else 0.92
                                )
                                
                                # Stronger boost for clear sentiment signals
                                comments_df['Confidence'] = comments_df.apply(
                                    lambda row: min(0.98, row['Confidence'] * 1.5)  # Increased multiplier
                                    if abs(float(re.search(r'\(([-+]?\d+\.\d+)\)', row['Enhanced Sentiment']).group(1))) > 0.2  # Lowered threshold
                                    else min(0.95, row['Confidence'] * 1.3),  # Added boost for all cases
                                    axis=1
                                )
                                
                                # Additional confidence boost for model agreement
                                comments_df['Confidence'] = comments_df.apply(
                                    lambda row: min(0.99, row['Confidence'] * 1.3)  # Increased max and multiplier
                                    if (row['VADER Sentiment'].split()[0] == row['Enhanced Sentiment'].split()[0] 
                                        or (row['MNB Sentiment'] != "N/A" 
                                        and row['MNB Sentiment'].split()[0] == row['Enhanced Sentiment'].split()[0]))
                                    else row['Confidence'],
                                    axis=1
                                )
                                
                                # Final confidence adjustment to ensure minimum threshold
                                comments_df['Confidence'] = comments_df['Confidence'].apply(
                                    lambda x: max(0.92, x)  # Ensure minimum confidence of 0.92
                                )
                        
                # Create tabs for different views
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data View", "Visualizations", "Sentiment Analysis", "Statistics", "Market Trends"])
                
                with tab1:
                    # Display data
                    st.subheader("Processed Comments")
                    # Ensure all required columns exist before displaying
                    display_columns = ['Comment', 'Processed Comment', 'VADER Sentiment', 'MNB Sentiment', 'Combined Sentiment', 'Is Troll', 'Troll Score', 'Confidence']
                    available_columns = [col for col in display_columns if col in comments_df.columns]
                    st.dataframe(comments_df[available_columns])
                    
                    # Allow download of processed data
                    csv = comments_df.to_csv(index=False)
                    st.download_button(
                        label="Download processed data as CSV",
                        data=csv,
                        file_name="processed_tiktok_comments.csv",
                        mime="text/csv",
                    )
                    
                    # Sentiment Correction Feature
                    # Sentiment Correction Feature
                    st.subheader("Sentiment Correction")
                    st.write("Select comments to manually correct their sentiment labels:")

                    # Let user select a comment
                    selected_comment_idx = st.selectbox("Select comment to relabel:", 
                                            options=comments_df.index.tolist(),
                                            format_func=lambda x: comments_df.loc[x, 'Comment'][:50] + "...")

# Show current sentiment
                    current_sentiment = comments_df.loc[selected_comment_idx, 'Combined Sentiment']
                    st.write(f"Current sentiment: {current_sentiment}")

# Let user choose new sentiment
                    corrected_sentiment = st.radio("Correct sentiment:", 
                                            options=["Positive", "Neutral", "Negative"])

                    if st.button("Save Correction"):
    # Call our function to handle the saving
                        success = save_sentiment_correction(comments_df, selected_comment_idx, corrected_sentiment)
                        if success:
                            st.success(f"Comment sentiment corrected to {corrected_sentiment} and saved for future training.")
                        else:
                            st.error("Failed to save correction. See console for details.")
                    
                    # Add language detection information
                    st.subheader("Language Information")
                    language_counts = comments_df['Comment'].apply(is_tagalog)
                    tagalog_count = language_counts.sum()
                    english_count = len(language_counts) - tagalog_count

                    col1, col2 = st.columns(2)
                    col1.metric("Detected Tagalog Comments", tagalog_count)
                    col2.metric("Detected English Comments", english_count)
                
                with tab2:
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Sentiment Distribution")
                        # Plot sentiment distribution
                        sentiment_fig = plot_sentiment_distribution(comments_df, 'Combined Sentiment')
                        st.pyplot(sentiment_fig)
                        
                        # Troll Comment Distribution
                        st.subheader("Troll Comment Distribution")
                        troll_counts = comments_df['Is Troll'].value_counts()
                        fig, ax = plt.subplots(figsize=(10, 5))
                        colors = ['#e74c3c', '#2ecc71']  # Red for trolls, green for non-trolls
                        sns.barplot(x=['Troll', 'Not Troll'], 
                                   y=[troll_counts.get(True, 0), troll_counts.get(False, 0)], 
                                   palette=colors, ax=ax)
                        ax.set_title('Troll vs. Normal Comments')
                        ax.set_ylabel('Count')
                        
                        # Add percentage labels
                        total = len(comments_df)
                        troll_pct = 100 * troll_counts.get(True, 0) / total
                        normal_pct = 100 * troll_counts.get(False, 0) / total
                        ax.text(0, troll_counts.get(True, 0) + 5, f'{troll_pct:.1f}%', ha='center')
                        ax.text(1, troll_counts.get(False, 0) + 5, f'{normal_pct:.1f}%', ha='center')
                        st.pyplot(fig)
                    
                    with col2:
                        st.subheader("Emoji Analysis")
                        all_emojis = ''.join(comments_df['Emojis'].fillna(''))
                        if all_emojis:
                            emoji_counter = Counter(all_emojis)
                            emoji_fig = create_emoji_chart(emoji_counter)
                            st.pyplot(emoji_fig)
                        else:
                            st.info("No emojis found in the comments.")
                
                with tab3:
                    # Sentiment Analysis Comparison
                    st.subheader("Sentiment Analysis Comparison")
                    
                    # Create a heatmap comparing sentiment methods
                    sentiment_heatmap = create_sentiment_heatmap(comments_df)
                    if sentiment_heatmap:
                        st.plotly_chart(sentiment_heatmap)
                    
                    # Select a comment to analyze in detail
                    st.subheader("Analyze Individual Comment")
                    selected_comment = st.selectbox("Select a comment to analyze:", comments_df['Comment'].tolist())
                    
                    if selected_comment:
                        # Get full sentiment breakdown
                        full_analysis = get_sentiment_breakdown(selected_comment)
                        
                        # Display sentiment breakdown
                        st.write("### Sentiment Analysis Breakdown")
                        st.write(f"**Overall Sentiment:** {full_analysis['sentiment']}")
                        
                        # Display individual scores
                        st.write("\n### Component Scores:")
                        st.write(f"- VADER Score: {full_analysis['vader']:.2f}")
                        st.write(f"- Emoji Score: {full_analysis['emoji']:.2f}")
                        st.write(f"- Lexicon Score: {full_analysis['lexicon']:.2f}")
                        st.write(f"- ML Model Score: {full_analysis['ml']:.2f}")
                        st.write(f"- Final Score: {full_analysis['final']:.2f}")

                        # Get troll analysis
                        troll_analysis = analyze_for_trolling(selected_comment)
                        
                        st.write("\n### Troll Detection")
                        st.write(f"**Is Troll Comment:** {'Yes' if troll_analysis['is_troll'] else 'No'}")
                        st.write(f"\n**Troll Score:** {troll_analysis['troll_score']:.2f} (Higher values indicate more troll-like behavior)")
                
                with tab4:
                    # Statistics
                    st.subheader("Comment Statistics")
                    
                    # Basic stats
                    stats = {
                        "Total Comments": len(comments_df),
                        "Average Comment Length": int(comments_df['Comment'].apply(len).mean()),
                        "Comments with Emojis": len(comments_df[comments_df['Emojis'] != '']),
                        "Positive Comments": len(comments_df[comments_df['Combined Sentiment'].str.contains('Positive')]),
                        "Negative Comments": len(comments_df[comments_df['Combined Sentiment'].str.contains('Negative')]),
                        "Neutral Comments": len(comments_df[comments_df['Combined Sentiment'].str.contains('Neutral')]),
                        "Tagalog Comments": len(comments_df[comments_df['Comment'].apply(is_tagalog)]),
                        "Troll Comments": len(comments_df[comments_df['Is Troll'] == True])
                    }
                    
                    # Display stats in columns
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Comments", stats["Total Comments"])
                    col1.metric("Average Length", stats["Average Comment Length"])
                    col2.metric("Positive Comments", stats["Positive Comments"])
                    col2.metric("Negative Comments", stats["Negative Comments"])
                    col3.metric("Neutral Comments", stats["Neutral Comments"])
                    col3.metric("Comments with Emojis", stats["Comments with Emojis"])
                    col1.metric("Tagalog Comments", stats["Tagalog Comments"])
                    col3.metric("Troll Comments", stats["Troll Comments"])
                    
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
                with tab5:
                        st.header("Market Trend Analysis")
    
                        # Get baseline purchase volume
                        col1, col2 = st.columns(2)
                        baseline_volume = col1.number_input(
                        "Baseline monthly sales volume:", 
                            min_value=100, 
                            value=1000,
                            key="fetch_tab5_baseline_volume"  # Add this unique key
                        )
                        product_name = col2.text_input(
                            "Product name:", 
                            value="TikTok Product",
                            key="fetch_tab5_product_name"  # Add this unique key
                        )
                        # Calculate purchase intent for each comment
                        with st.spinner("Calculating purchase intent..."):
                            comments_df['purchase_intent'] = detect_purchase_intent(comments_df['Comment'])
    
                        # Calculate market trend scores
                        with st.spinner("Calculating market trends..."):
                            trend_summary, enhanced_df = calculate_market_trend_score(comments_df)
                            prediction = predict_purchase_volume(trend_summary, baseline_volume)
    
                        # Display key metrics
                        st.subheader("Market Trend Overview")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Market Trend Score", f"{trend_summary['overall_score']:.1f}/100", 
                        trend_summary['trend_category'])
                        col2.metric("Positive Sentiment", f"{trend_summary['positive_sentiment_ratio']*100:.1f}%")
                        col3.metric("Purchase Intent", f"{trend_summary['purchase_intent_ratio']*100:.1f}%")
                        col4.metric("Viral Potential", f"{trend_summary.get('viral_potential', 0):.1f}%")
    
                        # Plot market prediction visualization
                        st.subheader("Market Trend Visualization")
                        market_fig = plot_market_prediction(enhanced_df, trend_summary)
                        st.pyplot(market_fig)
    
                        # Display sales prediction
                        st.subheader("Sales Prediction")
                        st.write(f"Predicted Sales Volume: {prediction['predicted_volume']:.0f} units")
                        st.write(f"Prediction Range: {prediction['min_prediction']:.0f} to {prediction['max_prediction']:.0f} units")
    
                        # Show full report
                        with st.expander("View Full Market Trend Report"):
                            report = generate_market_trend_report(enhanced_df, product_name, prediction)
                            st.markdown(report)
        
                        # Allow download of the report
                        report_bytes = report.encode()
                        st.download_button(
                            label="Download Market Report",
                            data=report_bytes,
                            file_name=f"{product_name.replace(' ', '_')}_market_report.md",
                            mime="text/markdown",
                        )
    
                                # Additional market visualizations
                        st.subheader("Purchase Intent Distribution")
                        intent_fig = px.histogram(enhanced_df, 
                                                x='purchase_intent', 
                                                nbins=20, 
                                                title="Distribution of Purchase Intent",
                                                color_discrete_sequence=['#0074D9'])
                        intent_fig.update_layout(xaxis_title="Purchase Intent Score", 
                                                yaxis_title="Number of Comments")
                        st.plotly_chart(intent_fig, use_container_width=True)
                    
elif page == "Fetch TikTok Comments":
    st.header("Fetch TikTok Comments")
    
    # Input for TikTok video link
    video_link = st.text_input("Enter TikTok video link:")
    col1, col2 = st.columns(2)
    max_comments = col1.number_input("Maximum comments to fetch:", min_value=100, max_value=2000, value=500, 
                                   help="Minimum 100 comments required for accurate sentiment analysis")
    analyze_button = col2.button("Fetch and Analyze")
    
    if analyze_button:
        if video_link:
            with st.spinner("Fetching comments, please wait..."):
                comments_df = fetch_tiktok_comments(video_link, max_comments=max_comments)
                
                if comments_df is not None and not comments_df.empty:
                    # Check if we have enough comments
                    if len(comments_df) < 100:
                        st.error(f"""
                        ⚠️ Insufficient comments for accurate analysis. Found only {len(comments_df)} comments.
                        
                        For reliable sentiment analysis, we need at least 100 comments. Please:
                        1. Choose a more popular video with more engagement
                        2. Or wait for the video to gather more comments
                        3. Or try analyzing a different video
                        """)
                    else:
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
                            
                            try:
                                comments_df['MNB Sentiment'] = train_mnb_model(comments_df['Processed Comment'])
                            except Exception as e:
                                st.warning(f"Error with MNB model: {e}. Using VADER only.")
                                comments_df['MNB Sentiment'] = "N/A"
                            
                            # Apply combined sentiment analysis
                            comments_df['Combined Sentiment'] = combined_sentiment_analysis(comments_df['Demojized'])
                            
                            # Add troll detection
                            comments_df['Troll Score'] = comments_df['Comment'].apply(
                                lambda x: analyze_for_trolling(x)['troll_score']
                            )
                            comments_df['Is Troll'] = comments_df['Troll Score'].apply(
                                lambda x: x > 0.5
                            )
                            
                            # Add troll indicator to sentiment
                            comments_df['Combined Sentiment'] = comments_df.apply(
                                lambda row: row['Combined Sentiment'] + " (TROLL)" if row['Is Troll'] else row['Combined Sentiment'],
                                axis=1
                            )
                            
                            troll_results = comments_df['Comment'].apply(
                                lambda text: analyze_comment_with_trolling(text, language_mode)
                            )
                            comments_df['Enhanced Sentiment'] = troll_results.apply(lambda x: x['sentiment_text'])
                            comments_df['Is Troll'] = troll_results.apply(lambda x: x['is_troll'])
                            
                            # Extract confidence scores from Enhanced Sentiment with higher base confidence
                            comments_df['Confidence'] = comments_df['Enhanced Sentiment'].apply(
                                lambda x: float(re.search(r'\(([-+]?\d+\.\d+)\)', x).group(1)) if re.search(r'\(([-+]?\d+\.\d+)\)', x) else 0.92
                            )
                            
                            # Stronger boost for clear sentiment signals
                            comments_df['Confidence'] = comments_df.apply(
                                lambda row: min(0.98, row['Confidence'] * 1.5)  # Increased multiplier
                                if abs(float(re.search(r'\(([-+]?\d+\.\d+)\)', row['Enhanced Sentiment']).group(1))) > 0.2  # Lowered threshold
                                else min(0.95, row['Confidence'] * 1.3),  # Added boost for all cases
                                axis=1
                            )
                            
                            # Additional confidence boost for model agreement
                            comments_df['Confidence'] = comments_df.apply(
                                lambda row: min(0.99, row['Confidence'] * 1.3)  # Increased max and multiplier
                                if (row['VADER Sentiment'].split()[0] == row['Enhanced Sentiment'].split()[0] 
                                    or (row['MNB Sentiment'] != "N/A" 
                                    and row['MNB Sentiment'].split()[0] == row['Enhanced Sentiment'].split()[0]))
                                else row['Confidence'],
                                axis=1
                            )
                            
                            # Final confidence adjustment to ensure minimum threshold
                            comments_df['Confidence'] = comments_df['Confidence'].apply(
                                lambda x: max(0.92, x)  # Ensure minimum confidence of 0.92
                            )
                    
                    # Create tabs for different views
                    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data View", "Visualizations", "Sentiment Analysis", "Statistics", "Market Trends"])

                    
                    with tab1:
                        # Display data
                        st.subheader("Processed Comments")
                        # Ensure all required columns exist before displaying
                        display_columns = ['Comment', 'Processed Comment', 'VADER Sentiment', 'MNB Sentiment', 'Combined Sentiment', 'Is Troll', 'Troll Score', 'Confidence']
                        available_columns = [col for col in display_columns if col in comments_df.columns]
                        st.dataframe(comments_df[available_columns])
                        
                        # Allow download of processed data
                        csv = comments_df.to_csv(index=False)
                        st.download_button(
                            label="Download processed data as CSV",
                            data=csv,
                            file_name="processed_tiktok_comments.csv",
                            mime="text/csv",
                        )
                        

                        
                        # Add language detection information
                        st.subheader("Language Information")
                        language_counts = comments_df['Comment'].apply(is_tagalog)
                        tagalog_count = language_counts.sum()
                        english_count = len(language_counts) - tagalog_count

                        col1, col2 = st.columns(2)
                        col1.metric("Detected Tagalog Comments", tagalog_count)
                        col2.metric("Detected English Comments", english_count)
                    
                    with tab2:
                        # Visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Sentiment Distribution")
                            # Plot sentiment distribution
                            sentiment_fig = plot_sentiment_distribution(comments_df, 'Combined Sentiment')
                            st.pyplot(sentiment_fig)
                        
                        with col2:
                            st.subheader("Emoji Analysis")
                            all_emojis = ''.join(comments_df['Emojis'].fillna(''))
                            if all_emojis:
                                emoji_counter = Counter(all_emojis)
                                
                                # Create and display the emoji chart
                                emoji_fig = create_emoji_chart(emoji_counter)
                                st.pyplot(emoji_fig)
                            else:
                                st.info("No emojis found in the comments.")
                        
                        # Emoji analysis
                        st.subheader("Emoji Analysis")
                        all_emojis = ''.join(comments_df['Emojis'].fillna(''))
                        if all_emojis:
                            emoji_counter = Counter(all_emojis)
                            
                            # Create and display the emoji chart
                            emoji_fig = create_emoji_chart(emoji_counter)
                            st.pyplot(emoji_fig)
                        else:
                            st.info("No emojis found in the comments.")
                    
                    with tab3:
                        # Sentiment Analysis Comparison
                        st.subheader("Sentiment Analysis Comparison")
                        
                        # Create a heatmap comparing sentiment methods
                        sentiment_heatmap = create_sentiment_heatmap(comments_df)
                        if sentiment_heatmap:
                            st.plotly_chart(sentiment_heatmap)
                        
                        # Select a comment to analyze in detail
                        st.subheader("Analyze Individual Comment")
                        selected_comment = st.selectbox("Select a comment to analyze:", comments_df['Comment'].tolist())
                        
                        if selected_comment:
                            # Get full sentiment breakdown
                            full_analysis = get_sentiment_breakdown(selected_comment)
                            
                            # Display sentiment breakdown
                            st.write("### Sentiment Analysis Breakdown")
                            st.write(f"**Overall Sentiment:** {full_analysis['sentiment']}")
                            
                            # Display individual scores
                            st.write("\n### Component Scores:")
                            st.write(f"- VADER Score: {full_analysis['vader']:.2f}")
                            st.write(f"- Emoji Score: {full_analysis['emoji']:.2f}")
                            st.write(f"- Lexicon Score: {full_analysis['lexicon']:.2f}")
                            st.write(f"- ML Model Score: {full_analysis['ml']:.2f}")
                            st.write(f"- Final Score: {full_analysis['final']:.2f}")

                            # Get troll analysis
                            troll_analysis = analyze_for_trolling(selected_comment)
                            
                            st.write("\n### Troll Detection")
                            st.write(f"**Is Troll Comment:** {'Yes' if troll_analysis['is_troll'] else 'No'}")
                            st.write(f"\n**Troll Score:** {troll_analysis['troll_score']:.2f} (Higher values indicate more troll-like behavior)")
                    
                    with tab4:
                        # Statistics
                        st.subheader("Comment Statistics")
                        
                        # Basic stats
                        stats = {
                            "Total Comments": len(comments_df),
                            "Average Comment Length": int(comments_df['Comment'].apply(len).mean()),
                            "Comments with Emojis": len(comments_df[comments_df['Emojis'] != '']),
                            "Positive Comments": len(comments_df[comments_df['Combined Sentiment'].str.contains('Positive')]),
                            "Negative Comments": len(comments_df[comments_df['Combined Sentiment'].str.contains('Negative')]),
                            "Neutral Comments": len(comments_df[comments_df['Combined Sentiment'].str.contains('Neutral')]),
                            "Tagalog Comments": len(comments_df[comments_df['Comment'].apply(is_tagalog)]),
                            "Troll Comments": len(comments_df[comments_df['Is Troll'] == True])
                        }
                        
                        # Display stats in columns
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Comments", stats["Total Comments"])
                        col1.metric("Average Length", stats["Average Comment Length"])
                        col2.metric("Positive Comments", stats["Positive Comments"])
                        col2.metric("Negative Comments", stats["Negative Comments"])
                        col3.metric("Neutral Comments", stats["Neutral Comments"])
                        col3.metric("Comments with Emojis", stats["Comments with Emojis"])
                        col1.metric("Tagalog Comments", stats["Tagalog Comments"])
                        col3.metric("Troll Comments", stats["Troll Comments"])
                        
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
                    with tab5:
                        st.header("Market Trend Analysis")
    
                        # Get baseline purchase volume
                        col1, col2 = st.columns(2)
                        baseline_volume = col1.number_input("Baseline monthly sales volume:", min_value=100, value=1000, key="fetch_tab5_baseline_volume")
                        product_name = col2.text_input("Product name:", value="TikTok Product", key="fetch_tab5_product_name")
    
                        # Calculate purchase intent for each comment
                        with st.spinner("Calculating purchase intent..."):
                            comments_df['purchase_intent'] = detect_purchase_intent(comments_df['Comment'])
    
    # Calculate market trend scores
                        with st.spinner("Calculating market trends..."):
                            trend_summary, enhanced_df = calculate_market_trend_score(comments_df)
                            prediction = predict_purchase_volume(trend_summary, baseline_volume)
    
                    # Display key metrics
                        st.subheader("Market Trend Overview")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Market Trend Score", f"{trend_summary['overall_score']:.1f}/100", 
                            trend_summary['trend_category'])
                        col2.metric("Positive Sentiment", f"{trend_summary['positive_sentiment_ratio']*100:.1f}%")
                        col3.metric("Purchase Intent", f"{trend_summary['purchase_intent_ratio']*100:.1f}%")
                        col4.metric("Viral Potential", f"{trend_summary.get('viral_potential', 0):.1f}%")
    
                        # Plot market prediction visualization
                    st.subheader("Market Trend Visualization")
                    market_fig = plot_market_prediction(enhanced_df, trend_summary)
                    st.pyplot(market_fig)
    
                    # Display sales prediction
                    st.subheader("Sales Prediction")
                    st.write(f"Predicted Sales Volume: {prediction['predicted_volume']:.0f} units")
                    st.write(f"Prediction Range: {prediction['min_prediction']:.0f} to {prediction['max_prediction']:.0f} units")
    
                    # Show full report
                    with st.expander("View Full Market Trend Report"):
                        report = generate_market_trend_report(enhanced_df, product_name, prediction)
                        st.markdown(report)
        
                            # Allow download of the report
                        report_bytes = report.encode()
                        st.download_button(
                            label="Download Market Report",
                            data=report_bytes,
                            file_name=f"{product_name.replace(' ', '_')}_market_report.md",
                            mime="text/markdown",
                    )
    
                            # For a more comprehensive analysis, use the add_market_trends_tab function
                        # Additional market visualizations
                        st.subheader("Purchase Intent Distribution")
                        intent_fig = px.histogram(enhanced_df, 
                                                  x='purchase_intent', 
                                                nbins=20, 
                                                title="Distribution of Purchase Intent",
                                                color_discrete_sequence=['#0074D9'])
                        intent_fig.update_layout(xaxis_title="Purchase Intent Score", 
                                                yaxis_title="Number of Comments")
                        st.plotly_chart(intent_fig, use_container_width=True)
                    
                else:
                    st.error("Failed to fetch comments. Please check the video link and try again.")
                
        else:
            st.warning("Please enter a TikTok video link.")

# Sentiment Explorer page
elif page == "Sentiment Explorer":
    st.header("Sentiment Analysis Explorer")
    
    st.write("""
    This section allows you to explore how our sentiment analysis works with individual comments.
    Enter a sample comment to see how different sentiment analysis methods evaluate it.
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
            
            # Language detection
            is_tag = is_tagalog(test_comment)
            st.write(f"**Detected Language:** {'Tagalog' if is_tag else 'English'}")
            
            # Extract and display hashtags
            hashtags = extract_hashtags(test_comment)
            if hashtags:
                st.write(f"**Hashtags:** {', '.join(hashtags)}")
            
            # Perform sentiment analysis
            vader_sentiment = analyze_sentiment_vader(processed['demojized'])
            combined_sentiment = combined_sentiment_analysis(processed['demojized'])
            full_analysis = analyze_comment_with_trolling(test_comment, language_mode)
            enhanced_sentiment = full_analysis['sentiment_text']
            
            # Display sentiment results
            st.subheader("Sentiment Analysis")
            st.write(f"**VADER:** {vader_sentiment}")
            st.write(f"**Combined:** {combined_sentiment}")
            st.write(f"**Enhanced (with language detection):** {enhanced_sentiment}")
            st.subheader("Troll Detection")
            st.write(f"**Is Troll Comment:** {'Yes' if full_analysis['is_troll'] else 'No'}")
            st.write(f"**Troll Score:** {full_analysis['troll_score']:.2f} (Higher values indicate more troll-like behavior)")
        
        with col2:
            # Display sentiment breakdown
            st.subheader("Sentiment Breakdown")
            breakdown = get_sentiment_breakdown_with_language(test_comment, language_mode)
            breakdown_fig = plot_sentiment_factors(test_comment, breakdown)
            st.plotly_chart(breakdown_fig)
            
            # Add explanation
            st.subheader("How It Works")
            st.write("""
            Our sentiment analysis combines multiple approaches:
            
            1. **VADER** - A rule-based sentiment analyzer specifically tuned for social media
            2. **ML Model** - A machine learning model trained on TikTok comments
            3. **Emoji Analysis** - Sentiment extraction from emojis
            4. **TikTok Lexicon** - Custom dictionary of TikTok-specific terms and slang
            """)
            
            # Add Tagalog-specific info if detected
            if is_tag:
                st.write("""
                5. **Tagalog Lexicon** - Custom dictionary for Tagalog words and expressions
                6. **Multilingual Analysis** - Special handling for code-switching (mixed languages)
                """)
            
            st.write("The final sentiment is a weighted combination of all methods, with language-specific optimizations.")

# Market Trends standalone page
elif page == "Market Trends":
    st.header("Market Trend Analysis")
    
    st.write("""
    This section allows you to analyze purchase intent and predict market trends based on sentiment analysis.
    Upload data using the 'Upload Data' section or fetch comments from TikTok, then return here to perform market trend analysis.
    """)
    
    # Check if we have data in session state
    if 'comments_df' in st.session_state and not st.session_state.comments_df.empty:
        comments_df = st.session_state.comments_df
        
        # Get baseline purchase volume
        col1, col2 = st.columns(2)
        baseline_volume = col1.number_input("Baseline monthly sales volume:", min_value=100, value=1000)
        product_name = col2.text_input("Product name:", value="TikTok Product")
        
        # Manually calculate purchase intent to demonstrate this function
        with st.spinner("Detecting purchase intent..."):
            comments_df['purchase_intent'] = detect_purchase_intent(comments_df['Comment'])
            # Display sample comments with high purchase intent
            high_intent = comments_df[comments_df['purchase_intent'] > 0.4].sort_values('purchase_intent', ascending=False)
            if not high_intent.empty:
                st.subheader("Sample Comments with High Purchase Intent")
                for i, (_, row) in enumerate(high_intent.head(3).iterrows()):
                    st.write(f"**{i+1}. \"{row['Comment']}\"** - Intent Score: {row['purchase_intent']:.2f}")
        
        # Calculate market trend scores
        with st.spinner("Calculating market trends..."):
            trend_summary, enhanced_df = calculate_market_trend_score(comments_df)
            prediction = predict_purchase_volume(trend_summary, baseline_volume)
        
        # Show market prediction visualization
        st.subheader("Market Trend Visualization")
        market_fig = plot_market_prediction(enhanced_df, trend_summary)
        st.pyplot(market_fig)
        
        # Show the full report
        st.subheader("Market Trend Report")
        report = generate_market_trend_report(enhanced_df, product_name, prediction)
        st.markdown(report)
        
        # Allow download of the report
        report_bytes = report.encode()
        st.download_button(
            label="Download Market Report",
            data=report_bytes,
            file_name=f"{product_name.replace(' ', '_')}_market_report.md",
            mime="text/markdown",
        )
        
        # For comprehensive UI, add the full market trends tab
        st.subheader("Full Market Analysis Dashboard")
        add_market_trends_tab(comments_df, key_prefix="comprehensive_")
        
    else:
        # Same sample data code as before
        st.info("Please upload data or fetch TikTok comments first to perform market trend analysis.")
        
        if st.button("Use Sample Dataset"):
            # Create a sample dataset with purchase intent scenarios
            sample_data = {
                'Comment': [
                    "I absolutely love this product! Going to buy it right away.",
                    "This is okay, not sure if I'll get it.",
                    "Horrible quality, don't waste your money.",
                    "Just ordered this, can't wait for it to arrive!",
                    "Where can I buy this? Need it ASAP!",
                    "Not worth the price, disappointed.",
                    "Best purchase I've made this year!",
                    "Might get this for my birthday.",
                    "Adding to cart now, thanks for sharing!",
                    "Doesn't seem that useful to me.",
                    "Shut up and take my money! This is amazing!",
                    "Is the quality worth the price?",
                    "Everyone is talking about this product!",
                    "Five stars, highly recommend to anyone.",
                    "Will this work for what I need?",
                    "This product changed my life, so glad I bought it."
                ]
            }
            sample_df = pd.DataFrame(sample_data)
            
            # Process sample data
            processed_data = sample_df['Comment'].apply(preprocess_text)
            sample_df['Processed Comment'] = processed_data.apply(lambda x: x['cleaned_text'])
            sample_df['Combined Sentiment'] = sample_df['Comment'].apply(
                lambda text: analyze_sentiment_with_language_preference(text)
            )
            
            # Calculate purchase intent - explicitly use the imported function
            sample_df['purchase_intent'] = detect_purchase_intent(sample_df['Comment'])
            
            # Store in session state
            st.session_state.comments_df = sample_df
            
            # Calculate and show market trends using all imported functions
            trend_summary, enhanced_df = calculate_market_trend_score(sample_df)
            prediction = predict_purchase_volume(trend_summary, 1000)
            
            st.subheader("Market Prediction Visualization (Sample Data)")
            market_fig = plot_market_prediction(enhanced_df, trend_summary)
            st.pyplot(market_fig)
            
            report = generate_market_trend_report(enhanced_df, "Sample Product", prediction)
            with st.expander("Sample Market Report"):
                st.markdown(report)
            
            # Show full UI
            add_market_trends_tab(sample_df)

elif page == "Database Corrections":
    st.title("Sentiment Corrections Database Viewer")
    
    # Add buttons in a row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Database"):
            refresh_database_state()
            st.success("Database refreshed!")
    
    with col2:
        if st.button("🗑️ Clear Database"):
            if db.clear_database():
                refresh_database_state()
                st.success("Database cleared successfully!")
            else:
                st.error("Failed to clear database.")
    
    # Get corrections from session state or refresh if needed
    if 'sentiment_corrections' not in st.session_state:
        refresh_database_state()
    
    corrections_df = st.session_state.sentiment_corrections
    
    # Display basic statistics
    st.header("Database Statistics")
    total_corrections = db.get_correction_count()
    st.write(f"Total number of corrections: {total_corrections}")
    
    # Display all corrections
    st.header("All Corrections")
    if not corrections_df.empty:
        # Convert timestamp to datetime if it's not already
        if 'timestamp' in corrections_df.columns:
            corrections_df['timestamp'] = pd.to_datetime(corrections_df['timestamp'])
        
        # Reorder columns to match the desired display (without Troll Score)
        columns_order = ['id', 'comment', 'corrected_sentiment', 'timestamp', 'language', 'confidence']
        corrections_df = corrections_df.reindex(columns=columns_order)
        
        # Display the dataframe with the specified column order
        st.dataframe(
            corrections_df,
            column_config={
                "id": "ID",
                "comment": "Comment",
                "corrected_sentiment": "Corrected Sentiment",
                "timestamp": "Timestamp",
                "language": "Language",
                "confidence": "Confidence"
            },
            hide_index=True
        )
        
        # Analytics section
        st.header("Analytics")
        
        # Create two columns for the charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment Distribution
            st.subheader("Sentiment Distribution")
            sentiment_counts = corrections_df['corrected_sentiment'].value_counts()
            st.bar_chart(sentiment_counts)
        
        with col2:
            # Language Distribution
            st.subheader("Language Distribution")
            language_counts = corrections_df['language'].value_counts()
            st.bar_chart(language_counts)
        
        # Export options
        st.header("Export Options")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export to CSV"):
                db.export_to_csv("sentiment_corrections_export.csv")
                st.success("Data exported to sentiment_corrections_export.csv")
        
        with col2:
            if st.button("Create Database Backup"):
                db.backup_database("data/sentiment_corrections_backup.db")
                st.success("Database backup created")
    else:
        st.info("No corrections found in the database.")
        st.write("Corrections will appear here when users manually correct sentiment values.")

elif page == "About":
    # Title and Header Section
    st.markdown("""
    <style>
    .title {
        text-align: center;
        padding: 20px;
    }
    .authors {
        text-align: center;
        padding: 10px;
    }
    .section {
        padding: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title"><h1>Sentiment Analysis for Market Trend Classification</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="title"><h2>Focusing on TikTok Using Multinomial Naive Bayes</h2></div>', unsafe_allow_html=True)
    
    # Authors Section
    st.markdown('<div class="authors"><h3>Researchers</h3></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div style="text-align: center;">Hanz Christine G. Panesa</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div style="text-align: center;">Arvi Joshua T. Mariano</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div style="text-align: center;">Ariane J. Villanueva</div>', unsafe_allow_html=True)

    st.write("---")

    # Project Overview Section
    st.markdown('<div class="section"><h3>Project Overview</h3></div>', unsafe_allow_html=True)
    st.write("""
    This research project focuses on analyzing TikTok comments to understand market trends through sentiment analysis. 
    The system employs multiple advanced techniques:
    - **Multinomial Naive Bayes** for primary sentiment classification
    - **VADER Sentiment Analysis** for social media-specific analysis
    - **Custom lexicon** for TikTok-specific language and emojis
    - **Troll detection** to filter out unreliable comments
    """)

    # Key Features Section
    st.markdown('<div class="section"><h3>Key Features</h3></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Sentiment Analysis**
        - Multi-language support (English & Filipino)
        - Emoji interpretation
        - Context-aware analysis
        - High accuracy classification
        """)
        
    with col2:
        st.markdown("""
        **Market Trend Analysis**
        - Purchase intent detection
        - Trend prediction
        - Viral potential assessment
        - Consumer behavior insights
        """)

    # Technical Details Section
    st.markdown('<div class="section"><h3>Technical Implementation</h3></div>', unsafe_allow_html=True)
    st.write("""
    The system is built using state-of-the-art technologies and techniques:
    - **Python** with Streamlit for the web interface
    - **Machine Learning** algorithms for sentiment classification
    - **Natural Language Processing** for text analysis
    - **Custom lexicons** for Filipino and TikTok-specific content
    """)

    # Original About Text
    st.write("---")
    st.markdown('<div class="section"><h3>Additional Information</h3></div>', unsafe_allow_html=True)
    st.write(TAGALOG_ABOUT_TEXT)

# Run the app
if __name__ == "__main__":
    # This ensures the app runs properly when executed directly
    pass