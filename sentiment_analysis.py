import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import nltk
import re
import string
import emoji
import joblib
import os
from pathlib import Path
import langdetect
from langdetect import detect

# Download required NLTK resources if not already present
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Enhanced emoji sentiment dictionary (expanded)
EMOJI_SENTIMENT = {
    # Highly positive
    "😊": 1.0,  # beaming face with smiling eyes
    "🥰": 1.0,  # smiling face with hearts
    "😍": 1.0,  # heart eyes
    "❤️": 1.0,  # heart
    "😁": 1.0,  # grinning face with smiling eyes
    "😄": 0.9,  # grinning face with smiling eyes
    "😃": 0.9,  # grinning face
    "😀": 0.9,  # grinning face
    "🤗": 1.0,  # hugging face
    "🥳": 1.0,  # partying face
    "😇": 1.0,  # smiling face with halo
    "🙂": 0.8,  # slightly smiling face
    "☺️": 0.8,  # smiling face
    "😌": 0.8,  # relieved face
    "😉": 0.7,  # winking face
    "🤣": 0.7,  # rolling on the floor laughing
    "😂": 0.7,  # face with tears of joy
    "😆": 0.7,  # grinning squinting face
    "😅": 0.6,  # grinning face with sweat
    
    # Moderately positive
    "👍": 0.7,  # thumbs up
    "👏": 0.6,  # clapping hands
    "🤩": 0.9,  # star-struck
    "😎": 0.6,  # smiling face with sunglasses
    "💪": 0.7,  # flexed biceps
    
    # Slightly positive
    "👌": 0.4,  # OK hand
    "🤭": 0.3,  # face with hand over mouth
    "💕": 0.6,  # two hearts
    "💓": 0.6,  # beating heart
    "💖": 0.6,  # sparkling heart
    
    # Neutral
    "🤔": 0.0,  # thinking face
    "😐": 0.0,  # neutral face
    "🙄": -0.1,  # face with rolling eyes
    "😶": 0.0,  # face without mouth
    "🤷": 0.0,  # person shrugging
    "⭐": 0.2,  # star
    "📱": 0.0,  # mobile phone
    "📸": 0.1,  # camera with flash
    "🎵": 0.1,  # musical note
    "🤨": -0.1,  # face with raised eyebrow
    
    # Slightly negative
    "😒": -0.4,  # unamused face
    "😕": -0.3,  # confused face
    "😟": -0.4,  # worried face
    "😬": -0.3,  # grimacing face
    "😧": -0.4,  # anguished face
    "😓": -0.4,  # downcast face with sweat
    "😨": -0.5,  # fearful face
    "😥": -0.4,  # sad but relieved face
    "😔": -0.4,  # pensive face
    "🙁": -0.3,  # slightly frowning face
    
    # Moderately negative
    "😢": -0.6,  # crying face
    "😭": -0.6,  # loudly crying face
    "😡": -0.7,  # pouting face
    "👎": -0.7,  # thumbs down
    "😠": -0.6,  # angry face
    "😤": -0.6,  # face with steam from nose
    "😩": -0.6,  # weary face
    "💔": -0.7,  # broken heart
    "😫": -0.6,  # tired face
    "😖": -0.5,  # confounded face
    
    # Highly negative
    "😱": -0.8,  # face screaming in fear
    "🤬": -0.9,  # face with symbols on mouth
    "😈": -0.7,  # smiling face with horns
    "💀": -0.7,  # skull
    "🤢": -0.8,  # nauseated face
    "🤮": -0.9,  # face vomiting
    "😷": -0.6,  # face with medical mask
    "🙅": -0.7,  # person gesturing no
    "👿": -0.8,  # angry face with horns
    "🤡": -0.5,  # clown face
}

# TikTok-specific sentiment lexicon
TIKTOK_LEXICON = {
    # Positive TikTok slang
    'slay': 0.8, 'fire': 0.7, 'lit': 0.7, 'goated': 0.9, 'based': 0.7,
    'bussin': 0.8, 'valid': 0.6, 'vibes': 0.5, 'iconic': 0.8, 'ate': 0.7,
    'fax': 0.5, 'facts': 0.5, 'bet': 0.4, 'fyp': 0.3, 'foryou': 0.3,
    'relatable': 0.5, 'talented': 0.7, 'queen': 0.7, 'king': 0.7, 'legend': 0.8,
    'periodt': 0.6, 'period': 0.6, 'win': 0.7, 'viral': 0.4, 'trending': 0.4,
    'clean': 0.5, 'chef kiss': 0.8, 'no cap': 0.3, 'sheesh': 0.7, 'glowing': 0.6,
    'stan': 0.5, 'vibe check': 0.4, 'rent free': 0.3, 'main character': 0.6,
    'elite': 0.7, 'chef\'s kiss': 0.8, 'baddie': 0.6, 'gem': 0.7, 'masterpiece': 0.9,
    
    # Negative TikTok slang
    'cringe': -0.7, 'flop': -0.8, 'mid': -0.5, 'ratio': -0.6, 'dead': -0.4,
    'basic': -0.5, 'ick': -0.7, 'yikes': -0.6, 'cap': -0.5, 'sus': -0.4,
    'cancel': -0.7, 'clickbait': -0.6, 'toxic': -0.7, 'cheugy': -0.5, 'copium': -0.4,
    'cursed': -0.6, 'clout chasing': -0.7, 'fake': -0.7, 'fraud': -0.8, 'scam': -0.8,
    'disappointing': -0.7, 'trash': -0.8, 'nightmare': -0.7, 'unfollow': -0.6, 'worst': -0.8,
    'shadowban': -0.7, 'shadow ban': -0.7, 'triggered': -0.5, 'cancelled': -0.7,
    'flopped': -0.8, 'overrated': -0.6, 'boring': -0.6, 'annoying': -0.7, 'wtf': -0.6,
    
    
    # Negative product-specific terms
    'defective': -0.8, 'sira': -0.8, 'broken': -0.8, 'not working': -0.8, 
    'budol': -0.9, 'peke': -0.8, 'counterfeit': -0.8, 'knockoff': -0.8,
    'overpriced': -0.7, 'mahal': -0.6, 'sobrang mahal': -0.8, 'not worth it': -0.7, 
    'hindi sulit': -0.7, 'sayang': -0.7, 'waste': -0.8, 'regret': -0.8,
    'misleading': -0.7, 'false advertising': -0.8, 'refund': -0.5, 'return': -0.5,
    'terrible': -0.8, 'avoid': -0.7, 'stay away': -0.8, 'pangit': -0.7,
    'marupok': -0.7, 'madaling masira': -0.8, 'walang kwenta': -0.8, 'useless': -0.8,
    'expired': -0.8, 'expiry': -0.7, 'arrived damaged': -0.8, 'doa': -0.8,
    'not as described': -0.7, 'hindi kapareho': -0.7, 'catfishing': -0.8,
    
    # Shopping platform specific terms
    'shopee scam': -0.9, 'lazada scam': -0.9, 'tiktok shop scam': -0.9,
    'shopee budol': -0.8, 'lazada budol': -0.8, 'tiktok shop budol': -0.8,
    'cod scam': -0.9, 'cash on delivery scam': -0.9,
    'seller ghosted': -0.8, 'hindi nagrereply': -0.7, 'no response': -0.7,
    'cancellation': -0.6, 'cancelled order': -0.6, 'delayed shipping': -0.6,
    
    # Positive product-specific terms
    'sulit': 0.7, 'worth it': 0.7, 'quality': 0.7, 'excellent': 0.8, 'legit': 0.8,
    'authentic': 0.7, 'original': 0.7, 'recommended': 0.7, 'mura': 0.6, 'affordable': 0.6,
    'fast delivery': 0.7, 'mabilis dumating': 0.7, 'responsive seller': 0.7,
    'generous seller': 0.7, 'exceeded expectations': 0.8, 'free gift': 0.6,
    'good packaging': 0.6, 'secure packaging': 0.6, 'well packed': 0.6
}
FILIPINO_LEXICON = {
    # Positive Filipino/Taglish words
    'ganda': 0.7, 'astig': 0.8, 'galing': 0.8, 'maganda': 0.7,
    'husay': 0.7, 'ang galing': 0.8, 'ang cute': 0.7, 'ang ganda': 0.8,
    'idol': 0.7, 'petmalu': 0.8, 'lodi': 0.7, 'solid': 0.7, 'lupet': 0.8,
    'panalo': 0.8, 'sana all': 0.6, 'nakakatuwa': 0.7, 'bongga': 0.7,
    
    # Negative Filipino/Taglish words
    'panget': -0.7, 'pangit': -0.7, 'chaka': -0.7, 'nakakabwisit': -0.8,
    'basura': -0.8, 'bastos': -0.7, 'walang kwenta': -0.8, 'epal': -0.7,
    'tanga': -0.8, 'bobo': -0.8, 'tae': -0.7, 'gago': -0.8, 'gaga': -0.8,
    'ulol': -0.8, 'pakyu': -0.9, 'bwisit': -0.7, 'nakakaasar': -0.7,
    'kadiri': -0.7, 'bulok': -0.7, 'tarantado': -0.8, 'pokpok': -0.8,
    
    # Filipino troll/bait words (common on TikTok)
    'lutang': -0.7, 'lenlen': -0.7, 'dilawan': -0.6, 'pinklawan': -0.6, 
    'yellowtard': -0.7, 'kulto': -0.8, 'bayaran': -0.7, 'paid': -0.6,
    'troll': -0.7, 'fake news': -0.7, 'apologist': -0.6, 'fanatic': -0.6,

        # Add more negative/troll Filipino words
    'engot': -0.7, 'inutil': -0.7, 'gunggong': -0.8, 'hangal': -0.7, 
    'ungas': -0.8, 'ugok': -0.7, 'ulol': -0.8, 'abnormal': -0.7,
    'baliw': -0.7, 'sira ulo': -0.8, 'sinungaling': -0.7, 'budol': -0.7,
    'balimbing': -0.6, 'traydor': -0.7, 'duwag': -0.7, 'takot': -0.5,
    'manahimik ka': -0.7, 'tumahimik ka': -0.7, 'tumigil ka': -0.7,
    'magsara': -0.6, 'umalis': -0.5, 'pabebe': -0.6, 'feeling': -0.6,
    'epal': -0.7, 'attention': -0.5, 'papansin': -0.6, 'praning': -0.6,
    'plastik': -0.7, 'fake': -0.7, 'peke': -0.7, 'poser': -0.6,
    
    # Additional political troll terms
    'bobotante': -0.8, 'abnoy': -0.8, 'lugaw': -0.7, 'laylayan': -0.6,
    'gurang': -0.7, 'matanda': -0.6, 'luka-luka': -0.7, 'oligarch': -0.7,
    'elitista': -0.7, 'tuta': -0.8, 'sundalong kanin': -0.7,
    'puppet': -0.7, 'diktador': -0.7, 'magnanakaw': -0.8, 'adik': -0.7,
    'drugas': -0.7, 'addict': -0.7, 'korap': -0.8, 'corrupt': -0.8,
    
    # Sarcastic terms often used in trolling
    'maka diyos': -0.5, 'disente': -0.5, 'itong': -0.4, 'heto': -0.4,
    'talaga naman': -0.5, 'asa pa': -0.6, 'good luck': -0.5, 'lmao': -0.6,
    'tatawa': -0.4, 'natawa': -0.5, 'patawa': -0.5, 'aliw': -0.4,
    'naaliw': -0.5, 'pinapatawa': -0.5, 'katatawa': -0.5

    
}

TROLL_PATTERNS = [
    # Basic patterns
    r'(ha){3,}',              # hahaha patterns
    r'(he){3,}',              # hehehe patterns
    r'!!+',                   # multiple exclamation marks
    r'\?\?+',                 # multiple question marks - fixed to escape ?
    r'[A-Z]{3,}',             # ALL CAPS WORDS
    r'\.{3,}',                # multiple periods (ellipsis)
    r'😂{2,}',                # multiple laughing emojis
    r'🤡|💀|🫠|💩',           # emojis commonly used in trolling
    r'(lutang|bobo|tanga)\s*(ka|talaga|naman)', # common insults 
    r'(\w+)(?:(?:\s+\1){2,})',  # repeated words (e.g., "galing galing galing")
    r'respect\s*my\s*opinion',  # common troll defense
    
    # Additional patterns - with fixes
    r'#(bbm|sara|leni|kakampink|pinklawan|dilawan|duterte|dutz|marcos)',  # Political hashtags
    r'(bbm|marcos|duterte|dutz|leni|lugaw|lutang)\s*pa\s*more',  # Political trolling phrase
    r'(bobo|tanga|gago)\s*(ka\s*ba|naman|talaga|amp)',  # Enhanced insult patterns  
    r'(haha|hihi|huhu|hehe){2,}',  # Repeated laugh patterns (hahahaha, etc)
    r'\?{2,}',                  # Fixed question mark patterns in Filipino comments
    r'(\s*ha){3,}',            # Enhanced "ha ha ha" pattern
    r'(\s*eh){3,}',            # "eh eh eh" pattern
    r'(naman){2,}',            # Repeated "naman"
    r'(edi\s*wow|edi\s*ikaw\s*na)',  # Sarcastic expressions
    r'(dami\s*alam|dami\s*satsat)',  # Dismissive phrases
    r'(feeling|filang)\s*(expert|galing|maganda|pogi|matalino)',  # Sarcastic compliment
    r'(ayaw|gusto|trip)\s*ko\s*yan[!]+',  # Exaggerated statements
    r'(fake|imbento|gawa-gawa|kasinungalingan)\s*(news|yan)',  # Fake news accusations
    r'(delawan|dilawan|yellowtard|pinklawan)',  # Political faction insults
    r'(paid|bayad|bayaran)',  # Paid troll accusations
    r'(communist|komunista|npa|terorista)',  # Political labeling
    r'(wala\s*kang\s*alam|wala\s*kang\s*karapatan)',  # Dismissive statements
    r'(kabobohan|katangahan|kaululan)',  # Name-calling
    r'\b(ok|sige|sure)\s*na\s*yan\s*for\s*you',  # Dismissive agreement
    r'#(tiktokfamous|viralvideo|foryoupage|fyp)',  # Hashtag baiting
        # Product-specific troll patterns
    r'(fake|peke|scam|budol|lokohan|manloloko)\s*(product|item|seller)',  # Fake product accusations
    r'(over|sobrang)\s*(priced|mahal)',  # Overpriced complaints
    r'(don\'t|wag|huwag)\s*(buy|order|bilhin)',  # Discouraging purchases
    r'(waste|sayang)\s*(of|ng)\s*(money|pera)',  # Money waste claims
    r'(worst|pinakamasamang)\s*(purchase|bili|product)',  # Extreme negative claims
    r'(returns?|refunds?)\s*(denied|rejected|hindi)',  # Return/refund complaints
    r'(buyer|customer)\s*(beware|ingat)',  # Warning other customers
    r'(marketing|ad|advertisement)\s*(scam|lie|kasinungalingan)',  # Marketing dishonesty claims
    r'(not|hindi)\s*(worth|sulit)',  # Value complaints
    r'(broken|sira|defective)\s*(on|upon|pagka)\s*(arrival|dating|deliver)',  # DOA claims
    
    # Exaggerated reviews
    r'(never|hindi\s*na\s*ulit)\s*(buying|bibili)',  # Never buying again
    r'(regret|nagsisisi)\s*(buying|purchase)',  # Purchase regret
    r'(0|zero)\s*(stars|rating)',  # Zero rating claims
    r'(this|ito)\s*(ain\'t|hindi)\s*(it|maganda|okay)',  # Dismissive language
    
    # Suspicious behavior patterns
    r'(all|lahat\s*ng)\s*(reviews|ratings|comments)\s*(fake|peke|paid|bayad)',  # Fake review accusations
    r'(shop|store)\s*(paying|nagbabayad)\s*for\s*(good|positive|5\s*star)',  # Paid review accusations
    r'(daming|andaming|ang\s*dami\s*ng)\s*(tanga|bobo|uto-uto)',  # Insulting other customers
    r'(obvious|halatang)\s*(paid|bayad)',  # Calling out paid endorsements
    
    # Specific product claim patterns
    r'(expired|expiry|lumang)\s*(product|item)',  # Expired products
    r'(fake|peke|counterfeit|pirated|class\s*[a-z])',  # Counterfeit accusations
    r'(factory|manufacturer)\s*(defect|reject)',  # Factory defects
    r'(not|hindi)\s*(authentic|original|tunay)',  # Authenticity questions
    r'(expectations)\s*vs\s*(reality)',  # Expectation vs reality
    r'(order|expectation)\s*vs\s*(received|reality)',  # Order vs received
    
    # Mocking patterns
    r'(laughing|tumawa|natawa)\s*so\s*(hard|much)',  # Mocking reactions
    r'(clown|circus|joke)\s*(emoji|face)?',  # Calling product/seller a joke
    r'(imagine|isipin)\s*(paying|nagbayad)',  # Mocking buyers
    
    # Extreme comparison claims
    r'(better|mas\s*maganda)\s*(off|pa)\s*(buying|bumili)',  # Better off elsewhere claims
    r'(local|china|chinese)\s*(products|alternatives)\s*(better|mas\s*maganda)',  # Comparison to alternatives
    r'(could|pwede)\s*(have|sana)\s*(bought|bumili)',  # Regret comparisons
    
    # General catastrophizing
    r'(worst|pinaka\s*malala|pinakamasamang)\s*(product|experience)',  # Extreme negative statements
    r'(completely|totally|absolutely)\s*(useless|worthless|walang\s*kwenta)',  # Extreme uselessness claims
    r'(biggest|pinakamalaking)\s*(mistake|regret)',  # Extreme regret claims
    
    # Filipino-specific patterns
    r'(budol|na-budol|nabudol)',  # Scammed
    r'(sayang|nasayang)\s*(pera|money|funds)',  # Wasted money
    r'(lubog|nilubog|nilubog)\s*(pera|money)',  # Sunken cost
    r'(pinagkakitaan|kita|kumikita)\s*(lang)',  # Just for profit
    r'(puro|pure)\s*(kalokohan|pangako|promises)',  # Empty promises
    r'(panloloko|manloloko|pangloloko)',  # Deception terms
    # Add these patterns to your TROLL_PATTERNS list

    # Commercial solicitation patterns
    r'(dm|message|contact)\s+(me|us|for|to)',  # DM solicitations
    r'(collab|collaboration|ambassador|ambassadorship)',  # Collaboration requests
    r'(rent|rental|for rent|available)',  # Rental offers
    r'(shipping|ship|deliver|worldwide)',  # Shipping mentions
    r'(follow|back|mutual|engage)',  # Follow-for-follow patterns
    r'(jewelry|marketing|team|brand)',  # Marketing terms
    r'(quality|products|better|best)',  # Product quality claims
    r'(sourcing|purchasing|products from china)',  # Dropshipping references
    r'(promo|code|discount|voucher)',  # Promotion offers
    r'(send|receive|items|free|goodies)',  # Free item offers
    r'(instagram|ig|tiktok)\s+(@|account)',  # Social media account solicitations
    r'(clothes|wear|outfit|look)',  # Fashion promotion indicators
    r'(check us out|visit|profile)',  # Profile visit requests
]
def validate_regex_patterns(patterns):
    """
    Validates a list of regex patterns and returns only those that compile successfully.
    
    Args:
        patterns: List of regex pattern strings
        
    Returns:
        List of valid regex patterns
    """
    valid_patterns = []
    for i, pattern in enumerate(patterns):
        try:
            re.compile(pattern)
            valid_patterns.append(pattern)
        except re.error as e:
            print(f"Invalid regex pattern #{i}: '{pattern}'")
            print(f"Error: {e}")
            # Either skip the pattern or try to fix it
            # Here we'll skip it for safety
    
    return valid_patterns
# Function to detect language
def detect_language(text):
    """
    Detect if text is Tagalog, English, or Taglish (mixed).
    Returns 'tl' for Tagalog, 'en' for English, 'mixed' for Taglish, or 'unknown'.
    """
    if not isinstance(text, str) or not text:
        return 'unknown'
    
    # Common Filipino markers that help identify Tagalog/Taglish
    filipino_markers = ['ang', 'ng', 'mga', 'sa', 'ko', 'mo', 'ka', 'naman', 'po',
                        'na', 'ay', 'yung', 'ito', 'yan', 'siya', 'ikaw', 'ako']
    
    # Count Filipino marker words
    words = text.lower().split()
    filipino_word_count = sum(1 for word in words if word in filipino_markers)
    filipino_lexicon_words = sum(1 for word in words if word in FILIPINO_LEXICON)
    
    # If significant Filipino markers are found
    if filipino_word_count >= 2 or filipino_lexicon_words >= 2:
        # Simple check for English words
        english_markers = ['the', 'of', 'and', 'to', 'is', 'in', 'it', 'you', 'that']
        english_count = sum(1 for word in words if word in english_markers)
        
        if english_count >= 2:
            return 'mixed'  # Likely Taglish
        else:
            return 'tl'     # Likely Filipino/Tagalog
    
    # Try standard language detection (this can be error-prone for short texts)
    try:
        return detect(text)
    except:
        # If language detection fails
        return 'unknown'
    
    # Function to detect troll patterns
def detect_troll_patterns(text):
    """
    Detect patterns commonly found in troll comments.
    Returns a score from 0 (not troll-like) to 1 (highly troll-like).
    """
    if not isinstance(text, str) or not text:
        return 0.0
    
    # Initialize score
    troll_score = 0.0
    
    # Check for pattern matches
    pattern_matches = 0
    for pattern in TROLL_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            pattern_matches += 1
    
    # Increase impact of pattern matches (changed from 0.1 to 0.15 per match)
    # Normalize score based on matches (max impact 0.75 instead of 0.6)
    if pattern_matches > 0:
        troll_score += min(0.75, pattern_matches * 0.15)
    
    # Check for extreme sentiment words (common in trolling)
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Count strong negative Filipino words
    strong_negative_count = sum(1 for word in words if word in FILIPINO_LEXICON 
                               and FILIPINO_LEXICON[word] <= -0.7)
    
    # Increase impact of negative words (changed from 0.1 to 0.15 per word)
    # Add to score based on strong negative words (max impact 0.5 instead of 0.4)
    if strong_negative_count > 0:
        troll_score += min(0.5, strong_negative_count * 0.15)
    
    return min(1.0, troll_score)  # Cap at 1.0

def has_excessive_formatting(text):
    """
    Improved detection of excessive formatting typical of troll comments.
    Now better handles different types of content.
    """
    # Check for ALL CAPS (if the comment is mostly uppercase)
    words = text.split()
    
    # If comment is very short or emoji-only, formatting is less relevant
    if len(words) < 2 or all(c in emoji.EMOJI_DATA for c in text.strip()):
        return 0.0
        
    # Check for ALL CAPS    
    uppercase_words = sum(1 for word in words if word.isupper() and len(word) > 1)
    uppercase_ratio = uppercase_words / len(words)
    if uppercase_ratio > 0.5:  # If more than 50% of words are ALL CAPS
        return 0.5
    
    # Check for excessive punctuation
    punctuation_count = sum(1 for char in text if char in '!?.')
    if len(text) > 0:
        punctuation_ratio = punctuation_count / len(text)
        if punctuation_ratio > 0.1:  # If more than 10% of the text is punctuation
            return 0.4
    
    # Check for repeated characters (like "hahahaha" or "!!!!!")
    if re.search(r'(.)\1{3,}', text):  # Same character repeated 4+ times
        return 0.3
        
    return 0.0
def analyze_emoji_sentiment_for_trolls(emoji_text):
    """
    Enhanced analysis of emojis for troll detection.
    Now properly handles positive vs negative emojis.
    
    Returns a troll score from 0 to 1 based on emojis only
    """
    if not emoji_text:
        return 0.0
    
    # Emojis commonly used in trolling
    troll_emojis = {
        "🤡": 0.8,  # Clown face - very common in trolling
        "💀": 0.6,  # Skull - often used mockingly
        "😂": 0.4,  # Laughing crying - when used excessively
        "🤣": 0.4,  # Rolling on floor laughing - when used excessively
        "🫠": 0.5,  # Melting face - often used sarcastically
        "💩": 0.7,  # Poop emoji - directly insulting
        "🙄": 0.5,  # Eye roll - dismissive
        "🤦": 0.5,  # Facepalm - dismissive
        "🤪": 0.4,  # Zany face - mocking
        "🥴": 0.4,  # Woozy face - often used mockingly
        "👋": 0.3,  # Waving hand - dismissive in context
        "😴": 0.4,  # Sleeping - dismissive
        "🤓": 0.5,  # Nerd face - often used mockingly
    }
    
    # Get sentiment of emojis from the main emoji sentiment dictionary
    general_sentiment = analyze_emoji_sentiment(emoji_text)
    
    # If emojis are clearly positive, they're less likely to be trolling
    if general_sentiment > 0.7:
        return 0.0
    
    total_score = 0
    count = 0
    
    # Count occurrences of each emoji
    emoji_counts = {}
    for char in emoji_text:
        if char in emoji_counts:
            emoji_counts[char] += 1
        else:
            emoji_counts[char] = 1
    
    # Calculate total score based on emojis and their counts
    for emoji, count in emoji_counts.items():
        if emoji in troll_emojis:
            # Higher score for repeated troll emojis
            repetition_factor = min(2.0, 1.0 + (0.2 * (count - 1)))
            total_score += troll_emojis[emoji] * repetition_factor
            count += 1
    
    # If no troll emojis were found
    if count == 0:
        return 0.0
        
    # Normalize to 0-1 range
    return min(1.0, total_score / count)

class TrollDetector:
    """
    A class to detect troll comments with more contextual awareness
    """
    def __init__(self):
        # Keep track of common troll phrases/patterns
        self.troll_phrases = set()
        self.learn_common_troll_phrases()
        
    def learn_common_troll_phrases(self):
        """Load common troll phrases into memory"""
        # Add common Filipino troll phrases
        self.troll_phrases.update([
            "respect my opinion",
            "edi wow",
            "lutang",
            "leni lugaw",
            "dilawan",
            "pinklawan",
            "bias ka",
            "fakeNews",
            "paid troll",
            "bayaran",
            "snowflake",
            # Add more phrases as needed
        ])
    
    def contains_troll_phrase(self, text):
        """Check if text contains any known troll phrases"""
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in self.troll_phrases)
    
    def detect_troll(self, text):
        """
        Enhanced troll detection with multiple factors
        """
        # Get basic troll analysis
        basic_analysis = analyze_for_trolling(text)
        
        # Add phrase detection
        if self.contains_troll_phrase(text):
            # Increase troll score by 0.3 if it contains known troll phrases
            basic_analysis['troll_score'] = min(1.0, basic_analysis['troll_score'] + 0.3)
            basic_analysis['is_troll'] = basic_analysis['troll_score'] >= 0.3
            
        return basic_analysis



# Function to preprocess text for sentiment analysis
def preprocess_for_sentiment(text):
    """
    Preprocess text specifically for sentiment analysis, preserving emoticons and key phrases.
    """
    if not isinstance(text, str):
        return {"processed_text": "", "emojis": "", "demojized_text": "", "language": "unknown"}
    
    # Detect language - ADD THIS LINE
    language = detect_language(text)
    
    # Convert to lowercase
    text = text.lower()

    # Replace URLs with token
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
    
    # Replace user mentions with token
    text = re.sub(r'@\w+', ' USER ', text)
    
    # Extract and save emojis before modifying text
    emojis_found = ''.join(c for c in text if c in emoji.EMOJI_DATA)
    
    # Convert emojis to text representations
    text_with_emoji_names = emoji.demojize(text, delimiters=(" ", " "))
    
    # Remove hashtag symbol but keep the text
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove punctuation except ! and ? which can indicate sentiment
    text = text.translate(str.maketrans('', '', string.punctuation.replace('!', '').replace('?', '')))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()


    
    
    return {
        'processed_text': text,
        'emojis': emojis_found,
        'demojized_text': text_with_emoji_names,
        'language': language  
    }

# VADER Sentiment Analysis
def analyze_sentiment_vader(text):
    """
    Analyze sentiment using VADER with optimizations for social media content.
    """
    if not text or not isinstance(text, str):
        return "Neutral (0.00)"
    
    sid = SentimentIntensityAnalyzer()
    
    # VADER works better with original capitalization and punctuation
    scores = sid.polarity_scores(text)
    compound = scores['compound']
    
    # Adjust thresholds for social media content which tends to be more polarized
    if compound >= 0.05:
        return f"Positive ({compound:.2f})"
    elif compound <= -0.05:
        return f"Negative ({compound:.2f})"
    else:
        return f"Neutral ({compound:.2f})"

# Advanced Emoji Sentiment Analysis
def analyze_emoji_sentiment(emoji_text):
    """
    Analyze sentiment from emojis using a predefined dictionary.
    """
    if not emoji_text:
        return 0.0
    
    total_score = 0
    count = 0
    
    # Analyze each emoji in the text
    for char in emoji_text:
        if char in EMOJI_SENTIMENT:
            score = EMOJI_SENTIMENT[char]
            total_score += score
            count += 1
            print(f"Emoji: {char}, Score: {score}")  # Debug line
    
    # If no known emojis were found
    if count == 0:
        return 0.0
        
    final_score = total_score / count
    print(f"Final emoji score: {final_score}")  # Debug line
    return final_score

def train_with_labeled_data():
    """
    Train sentiment model using manually labeled data.
    Returns the trained model or None if no labeled data exists.
    """
    try:
        # Try to load labeled data
        labeled_data = pd.read_csv('sentiment_corrections.csv')
        
        if len(labeled_data) < 10:
            # Not enough data yet
            return None
            
        # Get base training data
        base_texts, base_labels = generate_training_data()
        
        # Extract labeled comments and sentiments
        labeled_texts = labeled_data['Comment'].tolist()
        labeled_sentiments = labeled_data['Corrected_Sentiment'].tolist()
        
        # Combine base and labeled data
        all_texts = base_texts + labeled_texts
        all_labels = base_labels + labeled_sentiments
        
        # Create and train the pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', MultinomialNB())
        ])
        
        pipeline.fit(all_texts, all_labels)
        
        # Save the custom model
        try:
            joblib.dump(pipeline, "custom_sentiment_model.joblib")
        except:
            pass
            
        return pipeline
        
    except Exception as e:
        # If any error occurs (like file not found), return None
        print(f"Error training with labeled data: {e}")
        return None

# Advanced lexicon-based sentiment with TikTok-specific terms
def analyze_lexicon_sentiment(text, language=None):
    """
    Analyze sentiment using TikTok and Filipino-specific lexicon.
    
    Args:
        text: Text to analyze
        language: Optional language code. If None, will detect internally.
    """
    if not text or not isinstance(text, str):
        return 0.0
    
    # Detect language if not provided
    if language is None:
        # Use the detect_language function
        language = detect_language(text)
    
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    
    # Also check for multi-word phrases
    bigrams = [words[i] + ' ' + words[i+1] for i in range(len(words)-1)]
    
    total_score = 0
    count = 0
    
    # Initialize lexicons with a default
    lexicons = [TIKTOK_LEXICON, FILIPINO_LEXICON]
    
    # Then modify the order based on detected language
    if language == 'tl':
        # Filipino content - check Filipino lexicon first, then TikTok
        lexicons = [FILIPINO_LEXICON, TIKTOK_LEXICON]
    elif language == 'mixed':
        # Taglish content - check both lexicons with equal priority
        lexicons = [FILIPINO_LEXICON, TIKTOK_LEXICON]
    # Default for 'en' or 'unknown' is already set above
    
    # Check for words in the appropriate lexicons
    for word in words:
        for lexicon in lexicons:
            if word in lexicon:
                total_score += lexicon[word]
                count += 1
                break  # Found in one lexicon, no need to check the other
    
    # Check for phrases
    for phrase in bigrams:
        for lexicon in lexicons:
            if phrase in lexicon:
                total_score += lexicon[phrase]
                count += 1
                break
    
    # If no sentiment words were found
    if count == 0:
        return 0.0
        
    return total_score / count

# Generate enhanced training data for ML models
def generate_training_data():
    """
    Generate enhanced training data for sentiment analysis models.
    """
    # Positive examples with TikTok-specific vocabulary
    positive_examples = [
        "i love this", "this is amazing", "great video", "awesome content", "love it",
        "so good", "incredible", "fantastic", "perfect", "the best",
        "outstanding", "excellent", "wonderful", "brilliant", "fabulous",
        "impressive", "superb", "exceptional", "terrific", "top notch",
        "this is fire", "absolutely slaying", "you ate this up", "living for this content",
        "this is bussin fr", "goated", "no cap this is lit", "iconic", "this is so based",
        "vibes are immaculate", "period queen", "talented af", "true masterpiece",
        "obsessed with this", "chef's kiss", "rent free in my mind", "sheesh",
        "main character energy", "elite content", "pop off", "stan forever"
    ]
    
    # Negative examples with TikTok-specific vocabulary
    negative_examples = [
        "i hate this", "this is terrible", "awful video", "bad content", "dislike it",
        "so bad", "horrible", "disappointing", "worst ever", "waste of time",
        "useless", "pathetic", "terrible", "awful", "dreadful",
        "poor quality", "unbearable", "rubbish", "lame", "disgusting",
        "this is cringe", "major flop", "mid at best", "giving me the ick",
        "big yikes", "that's cap", "kinda sus", "toxic behavior", "clickbait",
        "basic content", "cursed content", "nightmare fuel", "so fake",
        "actually delusional", "flopped hard", "pure trash", "annoying af",
        "get this off my fyp", "make it stop", "instant unfollow"
    ]
    
    # Neutral examples with TikTok-specific vocabulary
    neutral_examples = [
        "okay", "not sure", "maybe", "average", "alright",
        "not bad", "so so", "ordinary", "standard", "mediocre",
        "fair", "tolerable", "passable", "reasonable", "moderate",
        "neither good nor bad", "acceptable", "adequate", "middle of the road", "nothing special",
        "just scrolling", "pov: me watching", "idk about this", "no thoughts",
        "for legal reasons that's a joke", "here before it blows up", "algorithm bring me back",
        "wondering if", "anyone else notice", "first time seeing this", "interesting concept",
        "what's the song", "trying to understand", "need more context", "still processing",
        "commenting for the algorithm", "the algorithm blessed me"
    ]
    
    # Create training data
    texts = positive_examples + negative_examples + neutral_examples
    labels = (["Positive"] * len(positive_examples) + 
             ["Negative"] * len(negative_examples) + 
             ["Neutral"] * len(neutral_examples))
    
    return texts, labels

# Train multiple models for ensemble sentiment analysis
def train_ensemble_model():
    """
    Train an ensemble of ML models for sentiment analysis.
    Save the model for reuse.
    """
    # Check if model already exists
    model_path = Path("tiktok_sentiment_model.joblib")
    if model_path.exists():
        try:
            return joblib.load(model_path)
        except:
            pass  # If loading fails, train a new model
    
    # Generate training data
    texts, labels = generate_training_data()
    
    # Create feature extraction and model pipeline
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), min_df=2)
    
    # Create ensemble of models
    estimators = [
        ('nb', MultinomialNB()),
        ('svc', LinearSVC(C=1.0, class_weight='balanced', dual=False))
    ]
    
    # Build voting classifier
    ensemble = VotingClassifier(estimators=estimators, voting='hard')
    
    # Create and train the pipeline
    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('classifier', ensemble)
    ])
    
    # Train the model
    pipeline.fit(texts, labels)
    
    # Save the model
    try:
        joblib.dump(pipeline, model_path)
    except:
        pass  # If saving fails, just continue
    
    return pipeline

# Function that uses ML model to predict sentiment
def predict_sentiment_ml(text_series):
    """
    Use trained ensemble model to predict sentiment.
    
    Args:
        text_series: Pandas Series containing processed text
    
    Returns:
        Series of sentiment predictions with confidence
    """
    # Load or train the model
    model = train_ensemble_model()
    
    # Convert to list if it's a single string
    if isinstance(text_series, str):
        text_series = [text_series]
    
    # Ensure all inputs are strings
    text_series = [str(text) if text is not None else "" for text in text_series]
    
    # Predict class
    predictions = model.predict(text_series)
    
    # Try to get probabilities if possible (not all models support predict_proba)
    try:
        probabilities = model.predict_proba(text_series)
        confidence_scores = np.max(probabilities, axis=1)
        # Format results with confidence scores
        result = [f"{pred} ({conf:.2f})" for pred, conf in zip(predictions, confidence_scores)]
    except:
        # If predict_proba not available, use fixed confidence
        result = [f"{pred} (0.80)" for pred in predictions]
    
    if len(result) == 1 and isinstance(text_series, list) and len(text_series) == 1:
        return result[0]
        
    return pd.Series(result)

# Replace your existing train_mnb_model function with this one

def train_mnb_model(text_series):
    """
    Trains a MultinomialNB model on given text and returns predictions.
    First tries to use custom labeled data if available.
    
    Args:
        text_series: Pandas Series containing processed text
    
    Returns:
        Series of sentiment predictions
    """
    # Try to use custom labeled model first
    custom_model = train_with_labeled_data()
    
    if custom_model is not None:
        # We have enough labeled data, use the custom model
        pipeline = custom_model
    else:
        # Fall back to the regular training data
        train_texts, train_labels = generate_training_data()
        
        # Create and train the pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', MultinomialNB())
        ])
        
        pipeline.fit(train_texts, train_labels)
    
    # Convert to list if it's a single string
    if isinstance(text_series, str):
        text_series = [text_series]
    
    # Ensure all inputs are strings
    text_series = [str(text) if text is not None else "" for text in text_series]
    
    # Predict on input texts
    predictions = pipeline.predict(text_series)
    
    # Add confidence scores
    try:
        probabilities = pipeline.predict_proba(text_series)
        confidence_scores = np.max(probabilities, axis=1)
    except:
        # Fixed confidence if predict_proba fails
        confidence_scores = [0.8] * len(predictions)
    
    # Format results with confidence scores
    result = [f"{pred} ({conf:.2f})" for pred, conf in zip(predictions, confidence_scores)]
    
    if len(result) == 1 and isinstance(text_series, list) and len(text_series) == 1:
        return result[0]
        
    return pd.Series(result)

# Function to get sentiment scores breakdown
def get_sentiment_breakdown(text):
    """
    Get detailed breakdown of sentiment scores from different methods.
    
    Args:
        text: String text to analyze
    
    Returns:
        Dictionary with sentiment scores from each method
    """
    if not isinstance(text, str) or not text:
        return {
            "vader": 0.0,
            "emoji": 0.0,
            "lexicon": 0.0,
            "ml": 0.0,
            "final": 0.0,
            "sentiment": "Neutral"
        }
    
    # Process text
    processed = preprocess_for_sentiment(text)
    
    # VADER sentiment
    sid = SentimentIntensityAnalyzer()
    vader_scores = sid.polarity_scores(text)
    vader_score = vader_scores['compound']
    
    # Emoji sentiment
    emoji_score = analyze_emoji_sentiment(processed['emojis'])
    
    # Lexicon sentiment
    lexicon_score = analyze_lexicon_sentiment(text)
    
    # ML model prediction
    try:
        ml_sentiment = predict_sentiment_ml(processed['processed_text'])
        ml_score_match = re.search(r'\(([-+]?\d+\.\d+)\)', ml_sentiment)
        ml_score = float(ml_score_match.group(1)) if ml_score_match else 0.0
        
        # Convert categorical to numerical (-1 to 1 scale)
        if "Positive" in ml_sentiment:
            ml_score = abs(ml_score)
        elif "Negative" in ml_sentiment:
            ml_score = -abs(ml_score)
        else:
            ml_score = 0.0
    except:
        ml_score = 0.0
    
    # Adjusted weights - MNB now has higher weight than VADER
    weights = {
        'vader': 0.35,  # Reduced from 0.4
        'ml': 0.40,     # Increased from 0.3
        'emoji': 0.15,  # Kept the same
        'lexicon': 0.10 # Slightly reduced
    }
    
    # Calculate final score
    final_score = (
        vader_score * weights['vader'] +
        ml_score * weights['ml'] +
        emoji_score * weights['emoji'] +
        lexicon_score * weights['lexicon']
    )
    
    # Determine sentiment
    if final_score >= 0.05:
        sentiment = "Positive"
    elif final_score <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    return {
        "vader": vader_score,
        "emoji": emoji_score,
        "lexicon": lexicon_score,
        "ml": ml_score,
        "final": final_score,
        "sentiment": sentiment
    }

# Function to analyze comments for trolling behavior
def analyze_for_trolling(text):
    """
    Perform comprehensive analysis to detect troll comments with improved accuracy.
    Returns a dictionary with troll score and language information.
    
    Args:
        text: Comment text to analyze
        
    Returns:
        Dictionary with troll_score, language, and is_troll flag
    """
    # Process text and detect language
    processed = preprocess_for_sentiment(text)
    language = processed['language']
    
    # Check if text is only emojis
    emojis_found = processed['emojis']
    is_emoji_only = len(text.strip()) == len(emojis_found)
    
    # For emoji-only comments, we need special handling
    if is_emoji_only:
        # Check if emojis are positive
        emoji_score = analyze_emoji_sentiment(emojis_found)
        
        # If predominantly positive emojis, this is likely not a troll
        if emoji_score > 0.3:
            return {
                'troll_score': 0.0,
                'language': language,
                'is_troll': False,
                'sentiment': "Positive",
                'sentiment_score': emoji_score
            }
    
    # Get troll pattern score
    troll_pattern_score = detect_troll_patterns(text)
    
    # Get sentiment scores
    sentiment_breakdown = get_sentiment_breakdown(text)
    sentiment_score = sentiment_breakdown['final']
    
    # Get emoji troll score if emojis exist
    emoji_troll_score = 0
    if emojis_found:
        emoji_troll_score = analyze_emoji_sentiment_for_trolls(emojis_found)
    
    # Higher troll likelihood if extremely negative sentiment
    sentiment_factor = 0.0
    if sentiment_score <= -0.7:
        sentiment_factor = 0.4
    elif sentiment_score <= -0.4:
        sentiment_factor = 0.2
    elif sentiment_score <= -0.2:
        sentiment_factor = 0.1
    
    # Check for very short comments with strong negative words
    words = re.findall(r'\b\w+\b', text.lower())
    strong_negative_count = sum(1 for word in words if word in FILIPINO_LEXICON 
                              and FILIPINO_LEXICON[word] <= -0.7)
    
    # Short negative comments are often trolls
    is_short_comment = len(words) < 5
    if is_short_comment and strong_negative_count > 0:
        additional_factor = 0.2
    else:
        additional_factor = 0.0
    
    # Check for excessive formatting
    formatting_score = has_excessive_formatting(text)

    # Reduce emoji factor for predominantly positive emojis
    if emojis_found and analyze_emoji_sentiment(emojis_found) > 0.5:
        emoji_factor = 0.0  # Positive emojis shouldn't contribute to troll score
    else:
        emoji_factor = emoji_troll_score * 0.3
    
    # Context-aware analysis: comment length impacts significance of factors
    if len(words) > 10:  # Longer comments
        # Content matters more than formatting for longer comments
        troll_pattern_weight = 0.6
        formatting_weight = 0.2
    else:  # Shorter comments
        # Formatting and patterns are more significant in short comments
        troll_pattern_weight = 0.4
        formatting_weight = 0.4
    
    # Final troll score with weighted factors
    final_troll_score = min(1.0, 
                         (troll_pattern_score * troll_pattern_weight) + 
                         sentiment_factor + 
                         additional_factor + 
                         emoji_factor + 
                         (formatting_score * formatting_weight))
    
    # Flag as troll if score exceeds threshold
    is_troll = final_troll_score >= 0.3
    
    # Add commercial spam detection
    commercial_spam_score = detect_commercial_spam(text)
    
    # If very high commercial spam score, this is likely spam rather than a troll
    if commercial_spam_score > 0.6:
        return {
            'troll_score': commercial_spam_score,
            'language': language,
            'is_troll': True,
            'is_commercial_spam': True,  # Add this flag to indicate commercial spam
            'sentiment': sentiment_breakdown['sentiment'],
            'sentiment_score': sentiment_score
        }
    
    return {
        'troll_score': final_troll_score,
        'language': language,
        'is_troll': is_troll,
        'sentiment': sentiment_breakdown['sentiment'],
        'sentiment_score': sentiment_score
    }

# Example usage code for testing troll detection
def test_troll_detection():
    test_comments = [
        "I really love this content, so helpful!",
        "HAHAHAHA ANG BOBO MO NAMAN!!! 🤡🤡🤡",
        "Ang ganda ng video na to, very informative",
        "dilawan ka siguro bayaran ng mga aquino!!!!!",
        "This is mid at best, kinda disappointing",
        "RESPECT MY OPINION NALANG PO MGA SNOWFLAKE 🤡🤡🤡",
        "Neutral comment just saying hello",
        "walang kwentang content tapos ang panget pa ng presentation",
        "Sana all ganito kagaling mag explain",
    ]
    
    for comment in test_comments:
        result = analyze_for_trolling(comment)
        print(f"\nComment: {comment}")
        print(f"Language: {result['language']}")
        print(f"Troll Score: {result['troll_score']:.2f} (Is Troll: {result['is_troll']})")
        print(f"Sentiment: {result['sentiment']} (Score: {result['sentiment_score']:.2f})")

# To run the test when the script is executed directly
if __name__ == "__main__":
    # You can call your test function here
    test_troll_detection()
    
    # Or add your own test cases
    print("\nTesting a single comment:")
    result = analyze_for_trolling("Ang pangit ng content mo, walang kwenta!!!")
    print(f"Troll Score: {result['troll_score']:.2f} (Is Troll: {result['is_troll']})")
    
def detect_commercial_spam(text):
    """
    Detect commercial spam patterns in comments.
    Returns a score from 0 (not spam-like) to 1 (likely spam).
    """
    if not isinstance(text, str) or not text:
        return 0.0
    
    # Initialize score
    spam_score = 0.0
    
    # Common commercial spam indicators
    commercial_patterns = [
        r'(dm|message)\s+(me|us)\s+for',
        r'(collab|collaboration|ambassador)',
        r'(follow|back|mutual)\s+(me|back)',
        r'(shipping|worldwide|delivery)',
        r'(brand|marketing|team)',
        r'(instagram|dm)\s+(@\w+)',
        r'(check|visit)\s+(out|profile)',
        r'(we|i)\s+(would|will)\s+(love|like)',
        r'(better|quality|best)\s+(products|quality)',
        r'(china|sourcing|supplier)',
        r'(we|i)\s+(offer|provide|sell|have)',
        r'(code|discount|promo|voucher)',
        r'(invest|opportunity|business)'
    ]
    
    # Check for pattern matches
    pattern_matches = 0
    for pattern in commercial_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            pattern_matches += 1
    
    # Calculate score based on matches
    if pattern_matches > 0:
        spam_score += min(0.9, pattern_matches * 0.15)
    
    # Check for social media handles or URLs
    if re.search(r'@\w+|https?://\S+|www\.\S+', text):
        spam_score += 0.2
    
    # Cross-promotion indicators
    if re.search(r'follow|dm|message|contact', text, re.IGNORECASE) and re.search(r'instagram|ig|email|dm', text, re.IGNORECASE):
        spam_score += 0.3
    
    return min(1.0, spam_score)  # Cap at 1.0
    
