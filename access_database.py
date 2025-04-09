import pandas as pd
from database import db
import streamlit as st

def view_database_contents():
    """Display the contents of the sentiment corrections database."""
    st.title("Sentiment Corrections Database Viewer")
    
    # Get all corrections
    corrections_df = db.get_corrections()
    
    # Display basic statistics
    st.header("Database Statistics")
    st.write(f"Total number of corrections: {db.get_correction_count()}")
    
    # Display the data
    st.header("All Corrections")
    if not corrections_df.empty:
        st.dataframe(corrections_df)
        
        # Show some basic analytics
        st.header("Analytics")
        
        # Sentiment distribution
        st.subheader("Sentiment Distribution")
        sentiment_counts = corrections_df['corrected_sentiment'].value_counts()
        st.bar_chart(sentiment_counts)
        
        # Language distribution
        st.subheader("Language Distribution")
        language_counts = corrections_df['language'].value_counts()
        st.bar_chart(language_counts)
        
        # Export options
        st.header("Export Options")
        if st.button("Export to CSV"):
            db.export_to_csv("sentiment_corrections_export.csv")
            st.success("Data exported to sentiment_corrections_export.csv")
            
        if st.button("Create Database Backup"):
            db.backup_database("data/sentiment_corrections_backup.db")
            st.success("Database backup created")
    else:
        st.info("No corrections found in the database.")

if __name__ == "__main__":
    view_database_contents() 