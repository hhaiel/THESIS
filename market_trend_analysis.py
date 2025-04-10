# market_trend_analysis.py
"""
Market trend analysis functions for TikTok Sentiment Analysis.
This module provides functions to analyze purchase intent and predict market trends
based on sentiment analysis of social media comments.
"""

import re
from turtle import st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from scipy.stats import linregress
from datetime import datetime

def get_troll_risk_level(data):
    """
    Assess the troll risk level based on comment data.
    
    Args:
        data (pandas.DataFrame): DataFrame with comment data that has troll scores
        
    Returns:
        dict: Risk assessment information
    """
    # Check if troll score column exists
    if 'Troll Score' not in data.columns:
        return {
            'risk_level': 'Unknown',
            'score': 0.0,
            'explanation': 'No troll score data available'
        }
    
    # Calculate average troll score
    avg_troll_score = data['Troll Score'].mean() if 'Troll Score' in data.columns else 0.0
    
    # Count high troll risk comments (score > 0.7)
    high_risk_count = (data['Troll Score'] > 0.7).sum() if 'Troll Score' in data.columns else 0
    high_risk_percentage = (high_risk_count / len(data)) * 100 if len(data) > 0 else 0
    
    # Determine risk level
    if avg_troll_score > 0.6 or high_risk_percentage > 15:
        risk_level = 'High'
        explanation = 'High concentration of potential trolling activity detected.'
    elif avg_troll_score > 0.3 or high_risk_percentage > 5:
        risk_level = 'Medium'
        explanation = 'Some potential trolling activity present.'
    else:
        risk_level = 'Low'
        explanation = 'Minimal trolling activity detected.'
    
    return {
        'risk_level': risk_level,
        'score': avg_troll_score,
        'high_risk_percentage': high_risk_percentage,
        'high_risk_count': high_risk_count,
        'explanation': explanation
    }

def detect_purchase_intent(text_series):
    """
    Enhanced analysis of text to detect purchase intent signals with improved sensitivity.
    EXTREME FIX: Much higher baseline intent for positive comments.
    
    Args:
        text_series: Series or list of text strings to analyze
        
    Returns:
        List of scores from 0.0-1.0 where higher values indicate stronger purchase intent
    """
    import re
    
    # Enhanced keywords indicating purchase intent - now with more variations and subtle signals
    intent_phrases = [
        # Direct intent phrases (expanded)
        r'(want|need|going|plan|intend|think|consider|might|could|would|should|wish) (to )?(buy|get|purchase|order|acquire|own|try|have)',
        r'(will|gonna|about to|planning on|looking forward to) (buy|get|purchase|order|pick up|grab|shop for)',
        r'i\'m (buying|getting|purchasing|ordering|considering|interested in)',
        r'adding to (cart|basket|wishlist|favorites|list)',
        r'take my money',
        r'shut up and take my money',
        r'where (can|do|could|should|would|might) (i|you|we|someone) (buy|get|purchase|find|order)',
        r'looking to (buy|purchase|get|acquire|invest in|try|have)',
        r'(buying|purchasing|getting|ordering|trying) this',
        r'can\'t wait to (buy|get|have|own|try|use)',
        r'need this in my life',
        r'(ordering|buying|purchasing) (this|now|soon|today|tomorrow|online)',
        r'purchased',
        r'bought',
        r'(in|added to) my cart',
        r'on my wishlist',
        r'link to (buy|purchase|shop|order|website|store)',
        r'link in bio',
        r'where to (buy|shop|find|order|get)',
        
        # Additional social indicators (expanded)
        r'must (have|buy|get|own|try)',
        r'(everyone|everybody) (needs|should get|should buy|would love|wants)',
        r'(obsessed|in love|amazed|impressed) with this',
        r'best purchase',
        r'worth (every penny|the money|the price|buying|checking out|looking into)',
        r'(perfect|ideal|amazing|great|excellent|good) for (me|my|anyone|everyone|people)',
        r'i (need|want|could use|would use|like|love) (this|one|it|these)',
        r'(ordering|buying) (another|a second|more)',
        r'just (bought|ordered|purchased|got|received)',
        r'already (bought|ordered|purchased|own|have)',
        r'selling out (fast|quickly)',
        r'add to cart',
        r'buy now',
        r'take my money',
        r'drop the link',
        r'where did you (get|buy|find)',
        r'where can i (get|buy|find|order|purchase)',
        r'how much (is it|does it cost|would this be)',
        r'(great|good|amazing|excellent|awesome|perfect) (deal|price|value|product|item)',
        r'invest in this',
        
        # Implied intent phrases (NEW - more subtle signals)
        r'(need|could use|would be useful|would help|would be perfect)',
        r'(wish|hope) (i|my)',
        r'(seems|looks) (useful|helpful|perfect|great|amazing)',
        r'(better than|superior to|beats) (what i have|my current|existing)',
        r'(would complement|would go well with|matches|fits with) my',
        r'(impressive|incredible|outstanding|remarkable|exceptional)',
        r'(exactly what|just what) (i|we) (need|want|looking for)',
        r'(saving up|saving money) for',
        r'(birthday|christmas|holiday) (list|gift|present|idea)',
        r'(checks all|ticks all) (the boxes|my boxes)',
        r'(quality|features|design) (is|looks|seems) (amazing|excellent|great|perfect)',
        r'(want|need) this for (my|the)',
        r'(tempting|tempted|considering|contemplating)',
        r'(definitely|certainly|absolutely) (considering|an option|a possibility)'
    ]
    
    # Build regex pattern for all phrases
    intent_pattern = '|'.join(intent_phrases)
    
    # Expanded question phrases indicating consideration (now with more variations)
    question_phrases = [
        r'(is|are) (it|they|this|these) worth',
        r'should i (buy|get|invest in|spend money on|consider|look into|try)',
        r'(is|are) (it|they|this|these) (good|great|worth|quality|durable|reliable|effective|useful)',
        r'(how|what) (is|are) the (quality|price|cost|value|durability|performance|shipping|delivery)',
        r'does (it|this) (work|perform|last|hold up|do the job)',
        r'how (much|long|well|quickly|easily|effectively)',
        r'(good|worth|wise) (investment|purchase|buy|choice|decision)',
        r'anyone (tried|bought|used|have|recommend|like|love)',
        r'thoughts on',
        r'reviews on',
        r'recommend',
        r'comparable to',
        r'better than',
        r'thinking (about|of) (getting|buying|trying|ordering)',
        r'would you recommend',
        r'(pros|cons) of',
        r'(any|are there any) (issues|problems|concerns)',
        r'(how long|how much time) (does it take|to)',
        r'(is|are) (there|it|they) (compatible|available)',
        r'(would|could|might|should) (work for|be good for|fit)'
    ]
    question_pattern = '|'.join(question_phrases)
    
    # Social proof phrases (expanded with more subtle signals)
    social_proof_phrases = [
        r'(everyone|everybody) (has|loves|wants|needs|raves about|talks about) (this|it|one)',
        r'(trending|viral|popular|hot|must-have|essential|top|favorite)',
        r'(sold out|selling fast|going quickly|limited stock|limited availability)',
        r'(best seller|top rated|highly rated|five star|5 star|well reviewed)',
        r'influencer (favorite|choice|pick|recommended)',
        r'celebrity (endorsed|approved|used|seen with)',
        r'(tiktok|instagram|social media) made me (buy|want|need) it',
        r'(worth the hype|lives up to the hype|believe the hype|deserves the hype)',
        r'(my friend|everyone i know|people) (has|got|loves|recommends|uses) (this|it)',
        r'(saw|seen) (it|this) (everywhere|all over|online)',
        r'(everyone|everybody) (is talking about|raves about)',
        r'(went|going) viral',
        r'(blowing up|taking off|gaining popularity)',
        r'(online|internet) (sensation|favorite|obsession)'
    ]
    social_pattern = '|'.join(social_proof_phrases)
    
    # Positive review phrases (expanded)
    review_phrases = [
        r'(highly|definitely|absolutely|strongly|totally) recommend',
        r'(love|loving|loved|adore|enjoy|enjoying|enjoyed) (this|it|mine|using it)',
        r'(best|greatest|amazing|awesome|fantastic|excellent|outstanding) (purchase|investment|decision|buy|product)',
        r'(changed|improved|transformed|enhanced|upgraded) my life',
        r'(excellent|fantastic|incredible|outstanding|superb|exceptional|superior) (quality|value|product|item|design)',
        r'(five|5|4\.5|4) stars?',
        r'would (buy|purchase|get|choose|select|pick) again',
        r'(satisfied|happy|impressed|pleased|delighted|thrilled) with (my purchase|this|it)',
        r'(no regrets|worth every penny|money well spent|great value)',
        r'(exceeded|surpassed|beat) (my|all) expectations',
        r'(can\'t|couldn\'t) be (happier|more satisfied)',
        r'(perfect|ideal|exactly what i wanted|just what i needed)',
        r'(so glad|very happy|really pleased) (i bought|i got|i purchased)',
        r'(best thing|best product) (i\'ve|i have) (bought|purchased|owned|used)'
    ]
    review_pattern = '|'.join(review_phrases)
    
    # Product interest phrases (NEW - detecting interest without explicit intent)
    interest_phrases = [
        r'(interesting|intriguing|cool|neat|nice|good|great) (product|design|idea|concept|thing)',
        r'(love|like|appreciate|enjoy) the (design|look|style|concept|features)',
        r'(beautiful|gorgeous|attractive|elegant|sleek|stylish|pretty)',
        r'(useful|practical|functional|handy|convenient|versatile|helpful)',
        r'(innovative|creative|clever|smart|brilliant|genius)',
        r'(high quality|well made|well built|well designed|well crafted|premium)',
        r'(would be|could be|might be) (perfect|great|useful|helpful) for',
        r'(solves|fixes|addresses|tackles) (a|the|my) (problem|issue|challenge)',
        r'(wish|if only) (i|we) (had|could)',
        r'(never seen|first time seeing) (anything|something) (like this|so)',
        r'(didn\'t know|didn\'t realize) (i|you|we|they) (needed|wanted|could)',
        r'(brilliant|genius|clever) (idea|solution|design|concept)',
        r'(i\'m|i am) (curious|interested|intrigued) (about|by|in)'
    ]
    interest_pattern = '|'.join(interest_phrases)
    
    results = []
    
    for text in text_series:
        if not isinstance(text, str):
            results.append(0.0)
            continue
            
        text = text.lower()
        
        # Check for direct purchase intent
        intent_matches = len(re.findall(intent_pattern, text))
        
        # Check for consideration questions
        question_matches = len(re.findall(question_pattern, text))
        
        # Check for social proof indicators
        social_matches = len(re.findall(social_pattern, text))
        
        # Check for positive review indicators
        review_matches = len(re.findall(review_pattern, text))
        
        # Check for interest indicators (NEW)
        interest_matches = len(re.findall(interest_pattern, text))
        
        # Check for specific actions
        actions = {
            'purchase_completed': any(word in text for word in ['bought', 'purchased', 'ordered', 'received', 'arrived', 'delivered', 'own', 'mine', 'got it', 'have it']),
            'future_intent': any(phrase in text for phrase in ['will buy', 'gonna buy', 'planning to', 'going to buy', 'about to order', 'adding to cart', 'might buy', 'considering', 'thinking about']),
            'desire': any(word in text for word in ['want', 'need', 'wish', 'hoping', 'must have', 'obsessed', 'love this', 'can\'t wait', 'eager', 'excited']),
            'urgency': any(phrase in text for phrase in ['selling out', 'limited stock', 'while supplies last', 'almost gone', 'back in stock', 'exclusive', 'limited time', 'special offer']),
            'research': any(phrase in text for phrase in ['researching', 'looking into', 'comparing', 'checking out', 'reading reviews', 'watching videos'])
        }
        
        # Calculate enhanced intent score with greater weight on subtle signals and interest
        intent_score = min(1.0, (
            0.9 * intent_matches +                # Increased from 0.8
            0.5 * question_matches +              # Increased from 0.3
            1.0 * actions['purchase_completed'] + # Increased from 0.9
            0.8 * actions['future_intent'] +      # Increased from 0.7
            0.7 * actions['desire'] +             # Increased from 0.5
            0.6 * social_matches +                # Increased from 0.4
            0.5 * review_matches +                # Increased from 0.3
            0.7 * actions['urgency'] +            # Increased from 0.6
            0.4 * interest_matches +              # NEW
            0.3 * actions['research']             # NEW
        ))
        
        # EXTREME FIX: Much higher baseline for positive comments
        # This ensures positive comments ALWAYS have high purchase intent
        if 'positive' in text or 'love' in text or 'great' in text or 'good' in text or 'amazing' in text:
            intent_score = max(0.7, intent_score)  # GREATLY INCREASED from 0.5 to 0.7
        
        results.append(intent_score)
    
    return results
def calculate_market_trend_score(comments_df):
    # Create a copy of the dataframe to avoid modifying the original
    enhanced_df = comments_df.copy()
    
    # If purchase_intent isn't already calculated, calculate it
    if 'purchase_intent' not in enhanced_df.columns:
        enhanced_df['purchase_intent'] = detect_purchase_intent(enhanced_df['Comment'])
    
    # Extract sentiment scores from Combined Sentiment instead of Enhanced Sentiment
    # You'll need to adjust this based on how your Combined Sentiment is formatted
    enhanced_df['sentiment_score'] = enhanced_df['Combined Sentiment'].apply(
        lambda x: 1 if 'Positive' in x else (-1 if 'Negative' in x else 0)
    )
    
    # Calculate positive sentiment ratio
    positive_count = len(enhanced_df[enhanced_df['sentiment_score'] > 0])
    positive_sentiment_ratio = positive_count / len(enhanced_df) if len(enhanced_df) > 0 else 0
    
    # Calculate purchase intent ratio
    high_intent_count = len(enhanced_df[enhanced_df['purchase_intent'] > 0.4])
    purchase_intent_ratio = high_intent_count / len(enhanced_df) if len(enhanced_df) > 0 else 0
    
    # Calculate troll-free positive sentiment (if 'Is Troll' column exists)
    if 'Is Troll' in enhanced_df.columns:
        troll_free_df = enhanced_df[~enhanced_df['Is Troll']]
        troll_free_positive = len(troll_free_df[troll_free_df['sentiment_score'] > 0])
        troll_free_positive_ratio = troll_free_positive / len(troll_free_df) if len(troll_free_df) > 0 else 0
    else:
        troll_free_positive_ratio = positive_sentiment_ratio
    
    # Calculate overall market trend score (0-100 scale)
    # Weighted formula: 40% positive sentiment + 40% purchase intent + 20% troll-free positive sentiment
    overall_score = (
        positive_sentiment_ratio * 40 + 
        purchase_intent_ratio * 40 + 
        troll_free_positive_ratio * 20
    )
    
    # Scale to 0-100
    overall_score = min(100, overall_score * 100)
    
    # Determine trend category
    if overall_score >= 70:
        trend_category = "Strong Positive"
    elif overall_score >= 50:
        trend_category = "Positive"
    elif overall_score >= 30:
        trend_category = "Neutral"
    elif overall_score >= 10:
        trend_category = "Negative"
    else:
        trend_category = "Strong Negative"
    
    # Optional: Calculate viral potential based on engagement metrics
    # If you have metrics like comment count, like count, etc.
    # For now, we'll use a simple formula based on sentiment and purchase intent
    viral_potential = (positive_sentiment_ratio * 0.6 + purchase_intent_ratio * 0.4) * 100
    
    # Create a summary dictionary
    trend_summary = {
        'overall_score': overall_score,
        'positive_sentiment_ratio': positive_sentiment_ratio,
        'purchase_intent_ratio': purchase_intent_ratio,
        'trend_category': trend_category,
        'viral_potential': viral_potential
    }
    
    return trend_summary, enhanced_df


# Then, update the plot_market_prediction function to also use Combined Sentiment
def plot_market_prediction(enhanced_df, trend_summary):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Sentiment distribution with Combined Sentiment
    sentiment_counts = enhanced_df['Combined Sentiment'].apply(
        lambda x: 'Positive' if 'Positive' in x else ('Negative' if 'Negative' in x else 'Neutral')
    ).value_counts()
    
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']  # Green, Yellow, Red
    sentiment_colors = [colors[i] for i in range(len(sentiment_counts))]
    
    wedges, texts, autotexts = ax1.pie(
        sentiment_counts, 
        labels=sentiment_counts.index, 
        autopct='%1.1f%%',
        startangle=90,
        colors=sentiment_colors
    )
    
    # Make text more readable
    for text in texts + autotexts:
        text.set_fontsize(10)
        text.set_color('black')
    
    ax1.set_title('Sentiment Distribution')
    
    # Plot 2: Market Trend Score gauge
    gauge_colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60']
    gauge_positions = [0, 20, 40, 60, 80, 100]
    ax2.axis('equal')
    
    # Create color segments
    for i in range(len(gauge_positions)-1):
        ax2.barh(
            0, 
            gauge_positions[i+1] - gauge_positions[i], 
            left=gauge_positions[i], 
            height=0.5, 
            color=gauge_colors[i],
            alpha=0.8
        )
    
    # Add needle to indicate score
    score = trend_summary['overall_score']
    needle_length = 0.4
    ax2.arrow(
        score, 0, 
        0, needle_length, 
        head_width=3, 
        head_length=0.1, 
        fc='black', 
        ec='black'
    )
    
    # Add score text
    ax2.text(
        score, 
        -0.1, 
        f"{score:.1f}", 
        ha='center', 
        va='center', 
        fontsize=12, 
        fontweight='bold'
    )
    
    # Add category
    ax2.text(
        50, 
        0.6, 
        trend_summary['trend_category'], 
        ha='center', 
        va='center', 
        fontsize=14, 
        fontweight='bold'
    )
    
    # Set ticks and labels
    ax2.set_xlim(0, 100)
    ax2.set_ylim(-0.5, 1)
    ax2.set_xticks([0, 20, 40, 60, 80, 100])
    ax2.set_xticklabels(['0', '20', '40', '60', '80', '100'])
    ax2.set_yticks([])
    
    ax2.set_title('Market Trend Score')
    
    # Adjust layout
    plt.tight_layout()
    return fig
def predict_purchase_volume(trend_summary, baseline_volume=1000):
    """
    Enhanced model to estimate potential purchase volume based on sentiment, intent and viral potential.
    
    Parameters:
    trend_summary (dict): Summary from calculate_market_trend_score
    baseline_volume (int): Baseline number of purchases for neutral sentiment
    
    Returns:
    dict: Dictionary with purchase predictions and growth projections
    """
    # Calculate score-based multiplier with enhanced sensitivity (0.4 to 2.5)
    # This turns the 0-100 score into a multiplier where:
    # - Score 50 (neutral) = 1.0x baseline
    # - Score 100 (max positive) = 2.5x baseline (increased from 2.0x)
    # - Score 0 (max negative) = 0.4x baseline (decreased from 0.5x)
    trend_multiplier = 1.0 + (trend_summary['overall_score'] - 50) / 40
    
    # Ensure multiplier stays within reasonable bounds
    trend_multiplier = max(0.4, min(2.5, trend_multiplier))
    
    # Consider sentiment density as a factor (if available)
    sentiment_boost = 1.0
    if 'sentiment_density_factor' in trend_summary:
        sentiment_boost = trend_summary['sentiment_density_factor']
    
    # Calculate viral potential boost
    viral_boost = 1.0
    if 'viral_potential' in trend_summary:
        # Viral potential transforms to a multiplier (1.0 to 1.5)
        viral_potential = trend_summary['viral_potential']
        viral_boost = 1.0 + (viral_potential - 50) / 100 if viral_potential > 50 else 1.0
    
    # Final enhanced multiplier with viral component
    enhanced_multiplier = trend_multiplier * sentiment_boost * viral_boost
    
    # Calculate predicted volume
    predicted_volume = baseline_volume * enhanced_multiplier
    
    # Calculate more sophisticated confidence interval
    # Higher confidence with more positive sentiment
    confidence_factor = 0.3 - (0.1 * trend_summary['positive_sentiment_ratio'])  # 0.2-0.3 range
    confidence_margin = confidence_factor * predicted_volume
    
    # Project growth over time (if trending positively)
    growth_projection = {}
    if trend_summary['overall_score'] > 50:
        # Calculate monthly growth rate based on viral potential
        monthly_growth_rate = 0.05  # Base growth
        if 'viral_potential' in trend_summary:
            # Viral products grow faster
            viral_growth = (trend_summary['viral_potential'] - 50) / 50 * 0.15
            monthly_growth_rate += max(0, viral_growth)
        
        # Project for 3 months
        for month in range(1, 4):
            growth_projection[f'month_{month}'] = predicted_volume * ((1 + monthly_growth_rate) ** month)
    else:
        # For negative trending products, project decline
        monthly_decline_rate = 0.05 * (50 - trend_summary['overall_score']) / 50
        for month in range(1, 4):
            growth_projection[f'month_{month}'] = predicted_volume * ((1 - monthly_decline_rate) ** month)
    
    # Create an enhanced prediction dictionary
    prediction = {
        'baseline_volume': baseline_volume,
        'trend_score': trend_summary['overall_score'],
        'trend_multiplier': trend_multiplier,
        'sentiment_boost': sentiment_boost,
        'viral_boost': viral_boost,
        'enhanced_multiplier': enhanced_multiplier,
        'predicted_volume': predicted_volume,
        'min_prediction': predicted_volume - confidence_margin,
        'max_prediction': predicted_volume + confidence_margin,
        'confidence_margin': confidence_margin,
        'confidence_factor': confidence_factor,
        'positive_sentiment_ratio': trend_summary['positive_sentiment_ratio'],
        'purchase_intent_ratio': trend_summary['purchase_intent_ratio'],
        'trend_category': trend_summary['trend_category'],
        'growth_projection': growth_projection
    }
    
    # Add viral potential if available
    if 'viral_potential' in trend_summary:
        prediction['viral_potential'] = trend_summary['viral_potential']
    
    return prediction

def generate_market_trend_report(enhanced_df, product_name, prediction):
    # Extract sentiment distribution
    sentiment_counts = enhanced_df['Combined Sentiment'].apply(
        lambda x: 'Positive' if 'Positive' in x else ('Negative' if 'Negative' in x else 'Neutral')
    ).value_counts()
    
    # Calculate percentages
    total = sentiment_counts.sum()
    positive_pct = sentiment_counts.get('Positive', 0) / total * 100 if total > 0 else 0
    neutral_pct = sentiment_counts.get('Neutral', 0) / total * 100 if total > 0 else 0
    negative_pct = sentiment_counts.get('Negative', 0) / total * 100 if total > 0 else 0
    
    # Generate the report
    report = f"""
# Market Analysis Report for {product_name}

## Executive Summary
Based on analysis of {total} comments, the market sentiment for {product_name} is **{sentiment_counts.idxmax()}** ({sentiment_counts.max() / total * 100:.1f}%).

## Sentiment Analysis
- **Positive**: {positive_pct:.1f}%
- **Neutral**: {neutral_pct:.1f}%
- **Negative**: {negative_pct:.1f}%

## Purchase Intent
- **High Intent Comments**: {len(enhanced_df[enhanced_df['purchase_intent'] > 0.7])} ({len(enhanced_df[enhanced_df['purchase_intent'] > 0.7]) / total * 100:.1f}%)
- **Medium Intent Comments**: {len(enhanced_df[(enhanced_df['purchase_intent'] > 0.4) & (enhanced_df['purchase_intent'] <= 0.7)])} ({len(enhanced_df[(enhanced_df['purchase_intent'] > 0.4) & (enhanced_df['purchase_intent'] <= 0.7)]) / total * 100:.1f}%)
- **Low Intent Comments**: {len(enhanced_df[enhanced_df['purchase_intent'] <= 0.4])} ({len(enhanced_df[enhanced_df['purchase_intent'] <= 0.4]) / total * 100:.1f}%)

## Sales Prediction
- **Baseline Monthly Volume**: {prediction['baseline_volume']} units
- **Predicted Volume**: {prediction['predicted_volume']:.0f} units
- **Prediction Range**: {prediction['min_prediction']:.0f} to {prediction['max_prediction']:.0f} units

## Key Insights
"""
    
    # Add insights based on analysis
    # Sentiment insights
    if positive_pct > 70:
        report += "- Very strong positive sentiment indicates high market enthusiasm\n"
    elif positive_pct > 50:
        report += "- Overall positive sentiment suggests favorable market reception\n"
    elif negative_pct > 50:
        report += "- High negative sentiment indicates potential market concerns\n"
    
    # Purchase intent insights
    high_intent_ratio = len(enhanced_df[enhanced_df['purchase_intent'] > 0.7]) / total if total > 0 else 0
    if high_intent_ratio > 0.3:
        report += "- Strong purchase intent signals potential high conversion rates\n"
    elif high_intent_ratio < 0.1:
        report += "- Low purchase intent may require stronger call-to-action marketing\n"
    
    # Compare prediction to baseline
    prediction_change = (prediction['predicted_volume'] - prediction['baseline_volume']) / prediction['baseline_volume'] * 100
    if prediction_change > 20:
        report += f"- Projected sales increase of {prediction_change:.1f}% indicates strong growth potential\n"
    elif prediction_change > 0:
        report += f"- Modest projected sales increase of {prediction_change:.1f}% suggests stable market performance\n"
    else:
        report += f"- Projected sales decrease of {abs(prediction_change):.1f}% indicates potential challenges ahead\n"
    
    # Add sample high intent comments
    high_intent_comments = enhanced_df[enhanced_df['purchase_intent'] > 0.7].sort_values('purchase_intent', ascending=False)
    if not high_intent_comments.empty:
        report += "\n## Sample High Intent Comments\n"
        for i, (_, row) in enumerate(high_intent_comments.head(3).iterrows()):
            report += f"{i+1}. \"{row['Comment']}\" - Intent Score: {row['purchase_intent']:.2f}\n"
    
    # Add recommendations section
    report += "\n## Recommendations\n"
    
    if positive_pct > 60 and high_intent_ratio > 0.2:
        report += "- Capitalize on positive sentiment with targeted conversion campaigns\n"
        report += "- Consider slight price premium to maximize revenue\n"
    elif positive_pct > 40:
        report += "- Highlight positive aspects in marketing to strengthen perception\n"
        report += "- Address neutral sentiment with more compelling product benefits\n"
    else:
        report += "- Address negative sentiment points in product development\n"
        report += "- Consider promotional pricing to overcome market hesitation\n"
    
    # Add general recommendations
    report += "- Monitor sentiment trends over time for long-term strategy adjustment\n"
    report += "- Target high-intent segments with special offers to maximize conversion\n"
    
    return report
    
    
# Add the Streamlit integration function
def add_market_trends_tab(comments_df, key_prefix=""):
    # Get baseline purchase volume
    col1, col2 = st.columns(2)
    baseline_volume = col1.number_input(
        "Baseline monthly sales volume:", 
        min_value=100, 
        value=1000, 
        key=f"{key_prefix}baseline_volume"
    )
    product_name = col2.text_input(
        "Product name:", 
        value="TikTok Product", 
        key=f"{key_prefix}product_name"
    )
    
    # Calculate purchase intent for each comment
    with st.spinner("Calculating purchase intent..."):
        if 'purchase_intent' not in comments_df.columns:
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
    intent_fig = px.histogram(enhanced_df,  # type: ignore
                              x='purchase_intent', 
                            nbins=20, 
                            title="Distribution of Purchase Intent",
                            color_discrete_sequence=['#0074D9'])
    intent_fig.update_layout(xaxis_title="Purchase Intent Score", 
                            yaxis_title="Number of Comments")
    st.plotly_chart(intent_fig, use_container_width=True)