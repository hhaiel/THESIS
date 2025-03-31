import pandas as pd
import numpy as np
import re
import string
import emoji
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import joblib
import random
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

    # Tagalog stopwords to add to the existing stopwords set
TAGALOG_STOPWORDS = {
        'ang', 'ng', 'sa', 'na', 'at', 'ay', 'mga', 'ko', 'siya', 'mo', 'ito', 'po',
        'ako', 'ka', 'niya', 'si', 'ni', 'kayo', 'kami', 'tayo', 'sila', 'nila',
        'namin', 'natin', 'kanila', 'kaniya', 'ating', 'amin', 'akin', 'iyo', 'nito',
        'dito', 'diyan', 'doon', 'kung', 'pag', 'para', 'dahil', 'upang', 'nga',
        'lang', 'lamang', 'din', 'rin', 'pa', 'pala', 'ba', 'naman', 'kasi', 'hindi',
        'huwag', 'wag', 'oo', 'hindi', 'yung', 'yan', 'yun'
    }

    # Tagalog sentiment lexicon - words with sentiment scores (-1.0 to 1.0)
TAGALOG_LEXICON = {
        # Positive words (high positive score)
        'maganda': 0.8, 'masaya': 0.9, 'mabuti': 0.7, 'masarap': 0.7, 'makabuluhan': 0.6,
        'magaling': 0.8, 'husay': 0.7, 'galing': 0.8, 'mahusay': 0.8, 'perpekto': 0.9,
        'kahanga-hanga': 0.9, 'nakakabilib': 0.8, 'nakakamangha': 0.9, 'tagumpay': 0.7,
        'mahal': 0.6, 'mahalaga': 0.6, 'sulit': 0.6, 'gusto': 0.5, 'sobrang': 0.5,
        'bongga': 0.8, 'astig': 0.7, 'asik': 0.6, 'solid': 0.7, 'bet': 0.5,
        'ang galing': 0.9, 'ang ganda': 0.9, 'napakahusay': 0.9, 'lubos': 0.7,
        'salamat': 0.6, 'maraming salamat': 0.8, 'panalo': 0.8, 'ayos': 0.6,
        'nakaka-proud': 0.8, 'kahanga-hanga': 0.9, 'napakasaya': 0.9, 'nakakatuwa': 0.7,
        'nakakagaan ng loob': 0.8, 'nakakagana': 0.7, 'nakaka-inspire': 0.8, 'walang katulad': 0.8,
        'hinangaan': 0.7, 'pinapansin': 0.6, 'kinagigiliwan': 0.8, 'pinupuri': 0.8,
        'hinahangaan': 0.8, 'kinikilig': 0.8, 'kinagalakan': 0.7, 'pinapahalagahan': 0.7,
        
        # Moderately positive words
        'mabait': 0.5, 'maayos': 0.5, 'masayang': 0.6, 'masaganang': 0.6, 
        'tama': 0.4, 'tumpak': 0.4, 'swak': 0.5, 'pasado': 0.5, 'ok': 0.3,
        'okay': 0.3, 'makatotohanan': 0.4, 'matulungin': 0.5, 'marangal': 0.6,
        'paborito': 0.6, 'gusto ko': 0.6, 'magalang': 0.5, 'masigla': 0.5,
        'nakakagaan': 0.5, 'keri': 0.4, 'pwedeng pwede': 0.5, 'ayos lang': 0.4,
        'magandang araw': 0.5, 'maligayang bati': 0.6, 'mabuhay': 0.6, 'makakaraos': 0.5,
        'hindi masama': 0.4, 'kaaya-aya': 0.6, 'nakakagalak': 0.6, 'masaganang buhay': 0.6,
        'masuwerte': 0.6, 'pagpalain': 0.6, 'maginhawa': 0.5, 'makakapasa': 0.5,
        
        # Slightly positive words
        'sige': 0.2, 'pwede': 0.2, 'puwede': 0.2, 'totoo': 0.3, 'payag': 0.3,
        'pwede na': 0.2, 'sakto': 0.3, 'goods': 0.3, 'good': 0.3, 'kasya': 0.2,
        'nakakaintindi': 0.3, 'maiintindihan': 0.3, 'naiintindihan': 0.3, 'pagsang-ayon': 0.3,
        'tanggap': 0.3, 'tinatanggap': 0.3, 'sasang-ayon': 0.2, 'papayag': 0.2,
        'pagkakasundo': 0.3, 'nasa ayos': 0.2, 'makakapagpatuloy': 0.3, 'mabagal man': 0.1,
        'matutunan': 0.3, 'matututunan': 0.3, 'masusubukan': 0.3, 'susubukin': 0.2,
        
        # Neutral words
        'siguro': 0.0, 'baka': 0.0, 'marahil': 0.0, 'medyo': 0.0, 'pwede na rin': 0.0,
        'ganun': 0.0, 'ganyan': 0.0, 'bahala': 0.0, 'ewan': 0.0, 'ewan ko': 0.0,
        'di ko alam': 0.0, 'hindi ko alam': 0.0, 'baka naman': 0.0, 'kung sakali': 0.0,
        'depende': 0.0, 'nakadepende': 0.0, 'malay mo': 0.0, 'malay natin': 0.0,
        'hindi ko pa alam': 0.0, 'di pa sure': 0.0, 'tignan natin': 0.0, 'tingnan muna': 0.0,
        'wala pang kasiguraduhan': 0.0, 'wala pa akong masabi': 0.0, 'walang komento': 0.0,
        'walang maisip': 0.0, 'walang masabi': 0.0, 'walang ideya': 0.0, 'wala akong alam': 0.0,
        'mamaya na': 0.0, 'sa susunod': 0.0, 'kapag may oras': 0.0, 'sa ibang pagkakataon': 0.0,
        
        # Slightly negative words
        'hindi maganda': -0.3, 'hindi mabuti': -0.3, 'hindi masarap': -0.3,
        'ayaw': -0.3, 'ayaw ko': -0.4, 'mali': -0.3, 'di tama': -0.3,
        'kulang': -0.2, 'medyo pangit': -0.2, 'sana': -0.1, 'sayang': -0.3,
        'medyo masakit': -0.3, 'nakakalungkot': -0.3, 'nakakasama ng loob': -0.3,
        'hindi kasiya-siya': -0.3, 'nakakabagot': -0.2, 'nakakainip': -0.2,
        'hindi ko gusto': -0.3, 'hindi ko trip': -0.3, 'di ko bet': -0.3, 'di ko type': -0.3,
        'parang hindi tama': -0.2, 'parang may mali': -0.2, 'may pagkukulang': -0.3, 
        'kinukulang': -0.3, 'may kakulangan': -0.3, 'hindi pa sapat': -0.2,
        
        # Moderately negative words
        'pangit': -0.5, 'masama': -0.5, 'bulok': -0.6, 'panget': -0.5, 'malungkot': -0.5,
        'nakakabadtrip': -0.6, 'badtrip': -0.5, 'nakakainis': -0.5, 'kalungkutan': -0.5,
        'palpak': -0.6, 'walang kwenta': -0.6, 'walang kuwenta': -0.6, 'sablay': -0.5,
        'di gusto': -0.5, 'hindi gusto': -0.5, 'hindi ko gusto': -0.6, 'di ko gusto': -0.6,
        'kainis': -0.5, 'nakakabuwis-buhay': -0.6, 'nakakatamad': -0.5, 'nakakasawa': -0.5,
        'nakakaumay': -0.5, 'nakakayamot': -0.5, 'nakakasira ng araw': -0.6, 'nakakasama ng loob': -0.6,
        'nakakagalit': -0.6, 'nakakabagabag': -0.5, 'nakakakaba': -0.5, 'kabaliwan': -0.6,
        'kabobohan': -0.6, 'kacheapan': -0.5, 'kawalang hiya': -0.6, 'kasamaan': -0.6,
        'kasalanan': -0.5, 'katamaran': -0.5, 'kapabayaan': -0.5, 'kawalan ng respeto': -0.6,
        
        # Highly negative words
        'napakasama': -0.9, 'kasuklam-suklam': -0.9, 'nakakagalit': -0.8, 'nakakainis': -0.7,
        'napaka-sama': -0.9, 'sobrang pangit': -0.9, 'sobrang panget': -0.9, 'basura': -0.8,
        'walang kwentang': -0.8, 'walang kuwentang': -0.8, 'nakakayamot': -0.7,
        'kakahiya': -0.7, 'nakakahiya': -0.7, 'napakasamang': -0.9, 'pekeng': -0.7,
        'scam': -0.8, 'nakakabuwisit': -0.8, 'napakababa': -0.7, 'sobrang sama': -0.9,
        'katanga': -0.8, 'tanga': -0.8, 'hindi ko na alam': -0.6,
        'nakakabaliw': -0.8, 'nakakapanlumo': -0.8, 'nakakabagabag': -0.7, 'nakakabalisa': -0.7,
        'nakakasuka': -0.9, 'nakakaduwal': -0.9, 'nakakadiri': -0.8, 'nakakasama ng ugali': -0.8,
        'napakababa ng uri': -0.8, 'napaka-walang hiya': -0.9, 'napaka-walang galang': -0.8,
        'napaka-walang respeto': -0.8, 'walang pakundangan': -0.8, 'walang pakundangan sa tao': -0.9,
        'walang konsiderasyon': -0.7, 'walang awa': -0.8, 'walang puso': -0.8,
        'walang kaluluwa': -0.8, 'walang pagmamahal': -0.7, 'walang pagsisisi': -0.7,

        # Additional highly negative words
        'bobo': -0.8, 'engot': -0.8, 'inutil': -0.8, 'gago': -0.8, 'gaga': -0.8,
        'ulol': -0.8, 'tae': -0.7, 'bulok': -0.7, 'walang silbi': -0.8, 'ewan ko sayo': -0.6,
        'bwiset': -0.7, 'bwisit': -0.7, 'hayop': -0.7, 'hayup': -0.7, 'putangina': -0.9,
        'punyeta': -0.8, 'lintik': -0.7, 'leche': -0.7, 'peste': -0.7, 'kupal': -0.8,
        'gunggong': -0.8, 'hangal': -0.7, 'ungas': -0.8, 'ugok': -0.7, 'abnormal': -0.7,
        'sira ulo': -0.8, 'baliw': -0.7, 'sinungaling': -0.7, 'balimbing': -0.6,
        'traydor': -0.7, 'duwag': -0.7, 'makasarili': -0.7, 'ganid': -0.7,
        'malandi': -0.7, 'pokpok': -0.8, 'manyakis': -0.8, 'bastos': -0.7,
        'walang hiya': -0.8, 'walang modo': -0.7, 'tamad': -0.6, 'pabaya': -0.6,
        'mabaho': -0.6, 'maasim': -0.6, 'maitim': -0.6, 'pangit na ugali': -0.7,
        'madamot': -0.7, 'walang utang na loob': -0.8, 'makapal ang mukha': -0.7,
        'masakit sa ulo': -0.6, 'sakit sa ulo': -0.7, 'sakit sa puso': -0.7,
        'nakakabaliw': -0.7, 'nakakasuka': -0.8, 'nakakaumay': -0.7, 'nakakawalang gana': -0.7,
        'kadiri': -0.7, 'kasuklam': -0.8, 'kaawa-awa': -0.6, 'kahiya-hiya': -0.7,
        'kainis': -0.7, 'kabwiset': -0.7, 'kaloka': -0.6,
        'hindot': -0.9, 'kingina': -0.9, 'tang ina': -0.9, 'tarantado': -0.8, 'taragis': -0.8,
        'hinayupak': -0.8, 'hudas': -0.7, 'demonyo': -0.8, 'satanas': -0.8, 'impakto': -0.7,
        'impaktita': -0.7, 'walanghiya': -0.8, 'salot': -0.8, 'salbahe': -0.7, 'masamang damo': -0.7,
        'halimaw': -0.7, 'manloloko': -0.7, 'mandaraya': -0.7, 'bayaran': -0.7, 'balahura': -0.7,
        'taeng aso': -0.8, 'dedma': -0.6, 'buraot': -0.7, 'bruha': -0.7, 'etchos': -0.6,
        'eklavung': -0.7, 'eklabu': -0.7, 'ek-ek': -0.6, 'bolok': -0.7, 'bulag': -0.6,
        'bingi': -0.6, 'pipi': -0.6, 'walang utak': -0.8, 'walang bait': -0.8, 'walang isip': -0.8,

        # Common sarcastic phrases
        'wow ang galing': -0.7, 'talino talaga': -0.7, 'galing mo dyan': -0.6,
        'napakatalino mo': -0.7, 'ang husay mo talaga': -0.7, 'ang talino naman': -0.6,
        'sige nga': -0.5, 'matalino ka': -0.6, 'hanga ako sayo': -0.6,
        'bongga ka day': -0.6, 'naku po': -0.6, 'jusko po': -0.6,
        'ay wow': -0.6, 'edi wow': -0.7, 'ikaw na': -0.6, 'idol na kita': -0.6,
        'ang galing mo talaga': -0.7, 'genius ka talaga': -0.7, 'napakahusay': -0.7,
        'ang bright mo naman': -0.7, 'eh di ikaw na matalino': -0.8, 'ikaw na ang magaling': -0.7,
        'edi ikaw na': -0.7, 'apir naman dyan': -0.6, 'bow ako sayo': -0.6, 'nakakaproud ka': -0.6,
        'nakakatuwa ka naman': -0.6, 'proud na proud ako sayo': -0.7, 'saludo ako sayo': -0.6,
        'grabe ang talino': -0.7, 'ang laki ng utak': -0.6, 'grabe ang galing': -0.7,
        'nosebleed sa galing': -0.7, 'napakagaling mo talaga': -0.7, 'genius yarn': -0.7,
        'matalino yarn': -0.6, 'ang talino naman nito': -0.6, 'wow may nagsalita': -0.7,
        'edi ikaw na ang the best': -0.7, 'edi wow congrats': -0.7, 'bravo naman sayo': -0.6,
        'wow clap clap': -0.6, 'grabe naman ang galing': -0.7, 'nakakabilib talaga': -0.6,
        'medal ka dyan': -0.6, 'edi pabuhat kami sayo': -0.7, 'grabe ang husay ha': -0.7,
        'walang sinabi ang einstein sayo': -0.8, 'mas magaling ka pa kay einstein': -0.8,
        'i-nominate kita sa nobel prize': -0.7, 'prof ka ba': -0.6, 'master mo talaga to': -0.6,
        'expert ka dito ah': -0.6, 'sobrang dami mong alam': -0.7,
        
        # Additional positive expressions
        'mabuhay': 0.8, 'mabuhay ka': 0.8, 'pinagpala': 0.7, 'pagpalain ka': 0.8,
        'pagpalain ka ng Diyos': 0.9, 'Diyos na mahabagin': 0.8, 'mahal kita': 0.9,
        'love kita': 0.9, 'mahal na mahal kita': 0.9, 'sobrang mahal kita': 0.9,
        'mahal na mahal': 0.9, 'pinakamamahal': 0.9, 'iginagalang': 0.8, 'nirerespeto': 0.8,
        'ginagalang': 0.8, 'minamahal': 0.9, 'pinahahalagahan': 0.8, 'pinahahalagahan kita': 0.8,
        'ikaw ang pinakamagaling': 0.9, 'ikaw ang pinakamahusay': 0.9, 'ikaw ang the best': 0.9,
        'ikaw ang nangunguna': 0.8, 'ikaw ang number one': 0.8, 'may pag-asa': 0.7,
        'may pag-asa pa': 0.7, 'walang imposible': 0.8, 'kaya mo yan': 0.8, 'kaya natin to': 0.8,
        
        # Additional negative expressions
        'walang pag-asa': -0.8, 'wala nang pag-asa': -0.9, 'hopeless': -0.8, 'imposible': -0.7,
        'hindi mo kaya': -0.7, 'hindi natin kaya': -0.7, 'sumuko na': -0.8, 'give up na': -0.8,
        'i-surrender na': -0.8, 'wala nang magagawa': -0.8, 'hindi na mababago': -0.7,
        'wala nang remedy': -0.8, 'wala nang lunas': -0.8, 'hopeless case': -0.8,
        'malala na': -0.7, 'walang solusyon': -0.7, 'hindi na masosolusyunan': -0.8,
        'wala nang paraan': -0.8, 'walang paraang maisip': -0.7, 'walang exit strategy': -0.7,
        'wala nang kawala': -0.8, 'nakasukol': -0.7, 'nakakulong': -0.7, 'nakapiit': -0.7,
        'nakahuli': -0.7, 'nahuhuli': -0.7, 'nakakahon': -0.7, 'wala nang magagawa': -0.8,
        'hindi na mababago': -0.8, 'wala nang pupuntahan': -0.8, 'walang patutunguhan': -0.8,
        
        # Social media specific positive expressions
        'apir': 0.7, 'apir tayo': 0.7, 'petmalu': 0.8, 'lodi': 0.8, 'idol': 0.8,
        'labs': 0.7, 'labs kita': 0.8, 'labyu': 0.8, 'vareh': 0.6, 'pars': 0.6,
        'beshie': 0.6, 'bes': 0.6, 'beshy': 0.6, 'teh': 0.5, 'mars': 0.6,
        'mumsh': 0.6, 'mamsh': 0.6, 'sisteret': 0.6, 'bruder': 0.6, 'sis': 0.6,
        'siz': 0.6, 'bro': 0.6, 'chika': 0.5, 'chiks': 0.5, 'chika tayo': 0.5,
        'solid': 0.7, 'solido': 0.7, 'shelemet': 0.7, 'tenkyuuu': 0.7,
        'tenks': 0.6, 'tanks': 0.6, 'labmuch': 0.8, 'naol': 0.7, 'sanaol': 0.7,
        'omg': 0.7, 'OMG': 0.7, 'angas': 0.7, 'ansaveh': 0.7, 'werpa': 0.7,

        # Mixed sentiment patterns (positive then negative)
        'maganda pero': 0.2,  # "beautiful but" - starts positive but likely ends negative
        'maganda kaso': 0.1,  # "beautiful however" - stronger negative connotation after positive
        'maganda sana kaso': 0.0,  # "would have been beautiful but" - neutralizes positive
        'maganda sana kung hindi': -0.1,  # "would have been beautiful if not for" - slightly negative
        'masarap pero': 0.2,  # "delicious but" - starts positive but has caveat
        'masarap sana kung': 0.1,  # "would be delicious if" - conditional positive
        'ok naman pero': 0.0,  # "it's okay but" - neutral with negative tendency
        'mabait pero': 0.2,  # "nice but" - positive with reservation
        'magaling kaso': 0.2,  # "good at it but" - qualified positive
        'mahusay sana pero': 0.1,  # "would have been excellent but" - diminished positive

        # Mixed sentiment patterns (negative then positive)
        'pangit pero': -0.2,  # "ugly but" - starts negative but may end positive
        'masama pero': -0.2,  # "bad but" - negative with possible redemption
        'nakakabadtrip pero': -0.3,  # "annoying but" - negative with some positive
        'nakakainis pero': -0.3,  # "irritating but" - negative first impression but potential positive
        'bulok pero': -0.3,  # "rotten but" - strong negative with possible exception
        'hindi maganda pero': -0.2,  # "not beautiful but" - negative with positive possibility
        'ayaw ko pero': -0.3,  # "I don't like it but" - personal negative with potential positive
        'mali pero': -0.2,  # "wrong but" - negative with potential learning
        'hindi ko gusto pero': -0.3,  # "I don't like it but" - strong dislike with exception
        'disappointed pero': -0.4,  # "disappointed but" - stronger negative with minor positive
}

def is_tagalog(text):
        """
        Simple heuristic to detect if text is primarily in Tagalog.
        Uses common Tagalog function words as indicators.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            bool: True if text appears to be in Tagalog, False otherwise
        """
        if not isinstance(text, str) or not text:
            return False
        
        # Common Tagalog function words as detection markers
        tagalog_markers = {
            'ang', 'ng', 'sa', 'ay', 'mga', 'na', 'naman', 'po', 'ko', 'mo', 
            'siya', 'niya', 'ito', 'iyon', 'yung', 'yan', 'din', 'rin', 'pa',
            'kung', 'para', 'kasi', 'dahil', 'nga', 'wag', 'huwag'
        }
        
        # Clean and tokenize text
        cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
        words = cleaned.split()
        
        # Count Tagalog marker words
        tagalog_word_count = sum(1 for word in words if word in tagalog_markers)
        
        # If at least 15% of words are Tagalog markers or at least 2 Tagalog markers are present
        # in a short message, consider it Tagalog
        if len(words) > 0:
            if (tagalog_word_count / len(words) >= 0.15) or (tagalog_word_count >= 2 and len(words) <= 10):
                return True
        
        return False

def analyze_tagalog_lexicon_sentiment(text):
        """
        Analyze sentiment using Tagalog lexicon with enhanced sarcasm detection.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            float: Sentiment score from -1.0 to 1.0
        """
        if not isinstance(text, str) or not text:
            return 0.0
        
        text = text.lower()
        
        # Get words for single-word matching
        words = re.findall(r'\b\w+\b', text)
        
        # Look for multi-word expressions
        # Create n-grams from the text for matching multi-word expressions
        tokens = text.split()
        bigrams = [' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)]
        trigrams = [' '.join(tokens[i:i+3]) for i in range(len(tokens)-2)]
        
        # Combine all possible matches
        all_possible_matches = words + bigrams + trigrams
        
        # Check for sarcasm indicators
        sarcasm_detected = False
        positive_terms = set(['galing', 'husay', 'talino', 'wow', 'magaling', 'idol', 'sana all'])
        negative_terms = set(['tanga', 'bobo', 'pangit', 'hindi', 'bulok', 'ulol', 'gago'])
        
        # If text has both positive and negative terms, might be sarcastic
        has_positive = any(term in text for term in positive_terms)
        has_negative = any(term in text for term in negative_terms)
        
        # Sarcasm often starts with a positive statement followed by negative
        positive_at_start = any(text.startswith(term) for term in ['ang galing', 'wow', 'grabe', 'husay'])
        
        # Detect sarcasm patterns
        if (has_positive and has_negative) or positive_at_start:
            sarcasm_detected = True
        
        total_score = 0
        count = 0
        
        # Check in Tagalog lexicon (both regular and TikTok terms are included here)
        for item in all_possible_matches:
            if item in TAGALOG_LEXICON:
                score = TAGALOG_LEXICON[item]
                # If sarcasm is detected, reverse the score for positive terms
                if sarcasm_detected and score > 0:
                    score = -score
                total_score += score
                count += 1
        
        # If no sentiment words were found
        if count == 0:
            return 0.0
            
        return total_score / count

def generate_enhanced_tagalog_training_data():
        """
        Generate a more comprehensive and diverse Tagalog training dataset
        that includes various dialects, social media expressions, and slang.
        """
        # STANDARD TAGALOG EXPRESSIONS - POSITIVE
        standard_positive = [
            "ang ganda nito", "napakahusay", "gustong-gusto ko ito", "ang galing!", 
            "sobrang nagandahan ako", "napakaganda", "napakahusay mo", "magaling ka talaga",
            "the best to", "nagagandahan ako dito", "ang galing mo talaga", "nakakatuwa naman",
            "hanep to", "ansaya ko", "ang saya-saya", "ang galing galing", "sobrang perfect",
            "sobrang gandang content", "sobrang nakakatuwa", "nakakaaliw to", "galeng nito",
            "bet ko to", "sobrang nakakatuwa po", "nakakarelate ako dito", "saya dito",
            "ang tawang tawa ako", "nakakagana", "panalo ka dito", "lodi ka talaga",
            "ang husay ng content mo", "ang ganda ng pagkakagawa", "salamat sa tutorial",
            "nakakatulong talaga", "gandang tip ito", "naenjoy ko to", "ang tuwa ko naman",
            "napakagandang ideya ito", "napakagaling mong gumawa", "lubos akong nasiyahan",
            "tunay na nakakabilib", "napakaganda ng pagkakasulat", "mahusay na gawain", 
            "kahanga-hanga ang iyong talento", "talaga namang nakakatuwa", "maganda ang punto mo",
            "maayos na paliwanag", "lubhang nakakatuwa", "labis akong nagalit sa kagandahan nito"
        ]
        
        # STANDARD TAGALOG EXPRESSIONS - NEGATIVE
        standard_negative = [
            "ang pangit nito", "hindi maganda", "ayaw ko to", "ang sama nito", 
            "nakakairita", "badtrip ako dito", "ang pangit", "sobrang pangit",
            "walang kwenta", "walang kwenta to", "sayang oras ko", "hindi nakakatuwa",
            "nakakainis ka", "sobrang nakakainis", "nakakabwisit", "kadiri", 
            "walang sense", "sobrang panget", "ang sakit sa mata", "nakakawalang gana",
            "nakakasuka", "sobrang baduy", "baduy nito", "chaka", "basura content",
            "palpak to", "sablay to", "hindi nakakatawa", "walang kakwenta-kwenta",
            "iyak na lang ako dito", "napakasama nito", "pinakamatindi kong kinasusuklaman",
            "hindi ko lubos maisip ang kapangitan", "sobrang nakakagalit", "napakapangit talaga",
            "hindi ko ito nirerekomenda", "lubhang nakakadismaya", "walang silbi",
            "lubhang nakakagalit", "hindi kapani-paniwala", "basura lang ito",
            "pangit na pangit ako dito", "sobrang sama ng pagkakaluto nito"
        ]
        
        # STANDARD TAGALOG EXPRESSIONS - NEUTRAL
        standard_neutral = [
            "ok lang", "pwede na", "sakto lang", "hindi naman masama", 
            "naintindihan ko", "ganun pala yun", "medyo ok", "pwede na rin",
            "di ko pa sure", "siguro", "nagdadalawang isip ako", "baka ganun nga",
            "di ko alam", "ewan ko", "hindi ko pa napapanood buong video", 
            "first time ko makita to", "sino ba yan", "may tanong ako", 
            "sino yan?", "kailan to ginawa", "di ko pa alam", "gusto ko sana itanong",
            "pahingi ng link", "san mo nabili yan", "pa share naman ng recipe",
            "marahil ay kapani-paniwala", "hindi ko tiyak", "hindi ko masabi kung ano",
            "nasa gitna lang", "wala akong opinyon", "kapwa maganda at pangit",
            "depende sa pagtingin", "bahala na kayo", "ikaw bahala", "nasa tamang kalahati"
        ]
        
        # TIKTOK SPECIFIC SLANG - POSITIVE
        tiktok_positive = [
            "slayyy lodi", "pak ganern", "petmalu lodi", "solid talaga to", "grabeng husay",
            "awra bhie", "deserve mo to", "peri may pag serve ka", "apaka ganda sis",
            "dasurv mo to", "grabeng galing", "ang astig nito", "grabeng content",
            "yass queen slayyyy", "shookt sa galing", "mukhang sanaol", "kinikilig ako sa content mo",
            "ang pogi/ganda ng serve", "apaka iconic bhie", "sheeeesh ang galing mo bhie",
            "damnnn girl", "ang power mo mhie", "werpa sa content mo", "galingan mo teh",
            "naur you ate that", "mother ate that", "bhie ate n left no crumbs", "ure so slay",
            "hard slay bhie", "serve mother serve", "very mother behavior", "icon era",
            "yun oh! laptrip to", "charaught", "iba ka talaga bhie", "nasampal ako sa husay",
            "TAMA BEHAVIOR"
        ]
        
        # TIKTOK SPECIFIC SLANG - NEGATIVE
        tiktok_negative = [
            "flop to bhie", "cringe talaga", "edi wow", "irita ako dito", "yuck talaga",
            "flop era", "ampanget", "chaka neto", "nochuchuchu", "ekis to",
            "kadiring content", "anuba teh", "nakakabobo", "umay nako sa content mo",
            "stop na bhie", "ew ka bhie", "flop behavior", "very unmother", "nakapagod panuorin",
            "HAHAHAH AALIS NA KO", "hirap mong intindihin mhie", "weird neto", "not giving",
            "cringey yarn", "icky behavior talaga", "delete mo na bhie", "kalma bhie",
            "di ko bet", "pabida era", "cheugy yarn", "not the serve", "nochuchuchu sa content",
            "pick me behavior", "di ikaw ang main character", "cheap tingnan"
        ]
        
        # TIKTOK SPECIFIC SLANG - NEUTRAL
        tiktok_neutral = [
            "mema lang", "waiting pa", "abangan", "skl", "charot", "emz",
            "borde lang", "forda content", "mhie naman", "bhie anong meron", 
            "para saan to?", "ferson lang po", "mumsh ko lang skl",
            "nacucurious lang ako", "skeri bhie", "noted with thanks", "shookt ako",
            "eme lang", "hala seryoso?", "legit ba to?", "true po ba?", "luh ano to",
            "charr! baka naman", "ohmy ghorl", "akala ko pa naman", "pabakla-bakla",
            "seryoso o charot?", "pls context bhie", "ano kaya to", "ferson for today",
            "for today's video ha", "talaga bhie?", "nakakaloka, bat ganon?", "mumsh ano to",
            "accla thoughts?", "mag-isip bago mag-click dzai"
        ]
        
        # REGIONAL VARIANTS (BISAYA, ILOKANO, ETC) - POSITIVE
        regional_positive = [
            # Bisaya
            "nindot kaayo ni", "kaluoy nako ganahan ko ani", "lingaw kaayo", "nindot jud ni uy",
            "maayo jud kaayo", "ganahan ko ani", "grabe ka nice ani oi", "the best ni sya",
            "nindot jud kaayo ni da", "kalipay nako ani", "grabe jud ka nice", "paka nindot ani",
            # Ilokano
            "napintas unay", "naimbag daytoy", "nagpintas", "nagsayaat", "naglaing", 
            "naimas", "napia unay", "naglaing nga aramid",
            # Bikolano
            "maribok na ini", "masarap na ini", "magayon na ini", "makusog na ini", 
            "maray ini", "maraying gibo", "maray na gibo"
        ]
        
        # REGIONAL VARIANTS (BISAYA, ILOKANO, ETC) - NEGATIVE
        regional_negative = [
            # Bisaya
            "pangit kaayo ni", "ayaw ko ani", "wa juy pulos", "luod kaayo ni", 
            "wa juy ayo ni", "sayang ra kaayo", "di jud ko ganahan", "samok kaayo",
            "kapoy tan-awon", "di nindot", "laay kaayo", "kurat man diri",
            # Ilokano
            "madi unay", "dakep unay", "nakababain", "nakakauma", "nalaing da ngamin",
            "dakes daytoy", "naalas", "nakababain daytoy",
            # Bikolano
            "maraot na ini", "hababa na ini", "maati na ini", "dai ko ini gusto", 
            "dai maghali", "maraot na aramid", "maati na trabaho"
        ]
        
        # REGIONAL VARIANTS (BISAYA, ILOKANO, ETC) - NEUTRAL
        regional_neutral = [
            # Bisaya
            "okay ra man", "pwede na sad", "ambot lang", "wa ko kasabot", 
            "unsay nahitabo ani?", "di ko sigurado ani", "ambot asa ni padulong",
            "wala koy comment ani", "okay ra man sad", "pwede ra pod ni",
            # Ilokano
            "mabalin met laeng", "diak ammo", "baka kasta", "saan ko ammo", 
            "no kasta", "mabalin met a saan", "kasano ngay dayta",
            # Bikolano
            "pwede na ini", "baka pwede", "dai ko aram", "baka tama", 
            "pwede na man", "dai ko man aram", "haloy na"
        ]
        
        # TAGLISH (MIX OF TAGALOG & ENGLISH) - POSITIVE
        taglish_positive = [
            "super nice nito", "ang cute talaga", "very helpful video", "ang ganda ng effect",
            "super informative nito", "thanks sa tips", "nice tutorial po", "very useful information po",
            "wow ang galing ng idea na to", "sobrang helpful talaga", "good job sa pag explain",
            "very clear yung instructions", "ang talino ng solution mo", "perfect yung timing",
            "ang dami kong natutunan", "effective talaga yung tips mo", "ang enjoyable panoorin",
            "super entertaining and informative", "ang galing ng concept", "love the way inideliver mo",
            "sobrang well explained", "ang perfect timing neto", "keep it up bhie",
            "very insightful yung perspective mo", "sobrang inspiring ng content mo",
            "incredibly helpful video", "amazing yung way mo mag-explain", "so creative ng ideas mo",
            "sobrang interesting ng topic", "well-researched talaga", "passionate ka sa topic",
            "refreshing yung content mo", "elegant solution sa problem", "outstanding performance",
            "exceptional quality talaga", "sobrang professional ng dating"
        ]
        
        # TAGLISH (MIX OF TAGALOG & ENGLISH) - NEGATIVE
        taglish_negative = [
            "ang boring nito", "so cringe yung video", "waste of time to", "ang annoying talaga",
            "super disappointing", "yung quality ng video parang basura", "fake news to",
            "wag niyo to i-try, scam to", "super ineffective nito", "worst tutorial ever",
            "so unreliable yung information", "ang outdated na ng tips", "don't waste your time dito",
            "ang misleading ng video", "so inaccurate yung sinabi mo", "bulok yung quality",
            "sobrang toxic ng message", "so frustrating, di naman gumana", "clickbait lang to",
            "extremely irritating video", "absolutely nonsense yung content", "so deceptive talaga",
            "totally useless information", "very poorly executed", "ang unnecessarily complicated",
            "I regret watching this video", "completely wrong yung instructions",
            "super redundant content", "utterly pointless", "entirely misleading"
        ]
        
        # TAGLISH (MIX OF TAGALOG & ENGLISH) - NEUTRAL
        taglish_neutral = [
            "may point ka naman", "okay lang yung quality", "average lang", "pa-explain ng video",
            "need pa ba mag register?", "paano mag avail nito?", "san pwede bumili?",
            "can you make a tutorial in English?", "waiting for part 2", "kelan next video?",
            "check niyo yung comment section", "pahingi nga link", "sino may alam kung saan to?",
            "trying to understand pa", "let me think about it muna", "still processing the info",
            "I'm somewhat convinced", "medyo gets ko na", "moderately interesting naman",
            "I'll consider your points", "neither agree nor disagree ako", "di ko pa masabi",
            "could go either way", "medyo complex yung topic", "I'll reserve judgment muna",
            "okay lang naman, but...", "so-so yung execution", "50/50 ang impression ko"
        ]
        
        # SOCIAL MEDIA SPECIFIC SHORTENED FORMS - POSITIVE
        short_positive = [
            "10/10", "⭐⭐⭐⭐⭐", "💯", "A+++", "👍👍👍", "g2g", "gege", "yas", 
            "LT", "hahaha", "HAHAHAHA", "lmao", "rofl", "ROTFL", "LUB IT", "lubet",
            "♥️♥️♥️", "😍", "❤️", "lodi", "idol", "🥰", "👏👏👏", "🙌", "OP",
            "lit", "fire", "🔥", "💕", "poggers", "W", "W content", "big W",
            "benta", "bentang benta", "natawa ako", "natawa me", "hagalpak", "mega love"
        ]
        
        # SOCIAL MEDIA SPECIFIC SHORTENED FORMS - NEGATIVE
        short_negative = [
            "0/10", "⭐", "👎", "cringe", "meh", "ugh", "eww", "yikes", 
            "🙄", "😒", "🤮", "🤢", "L", "big L", "L content", "ngek",
            "nyek", "meh", "🤦‍♀️", "🤦‍♂️", "sus", "ew", "🚮", "nge",
            "heh", "duh", "wtf", "scam", "fake", "cap", "mid", "mid af",
            "kadire", "kadiri", "hays", "hayst", "hay nako", "jusko", "juskopo"
        ]
        
        # SOCIAL MEDIA SPECIFIC SHORTENED FORMS - NEUTRAL
        short_neutral = [
            "k", "kk", "ok", "hmm", "hmmm", "idk", "idc", "mema", 
            "lol", "lmao", "🤷‍♀️", "🤷‍♂️", "brb", "ttyl", "teka", "wait",
            "hehe", "haha", ".", "...", "?", "??", "interesting", "noted",
            "sge", "oki", "okie", "sigi", "oryt", "ayt", "aight", "alr",
            "alright", "ge", "g", "next", "keri", "keri naman", "pwede na"
        ]
        
        # CODE-SWITCHING AND MULTILINGUAL - POSITIVE
        multilingual_positive = [
            # Filipino mixed with Spanish
            "muy bien ang gawa mo", "excelente talaga", "perfecto naman", "que bonita nito",
            "magandang obra", "bien hecho talaga", "felicidades sa achievement",
            # Filipino mixed with Chinese/Hokkien
            "hopia mo po talaga", "pansit sa husay", "sosyal mo naman", "kiamkiam talaga to",
            # Filipino mixed with Arabic expressions
            "mashallah ang galing mo", "subhanallah ang ganda", "alhamdulillah napanood ko to",
            # Filipino mixed with Korean expressions
            "daebak talaga to", "jinjja maganda", "neomu joayo", "heol ang galing",
            "omo ang ganda talaga", "daebak to talaga", "aja fighting sa next video mo",
            # Filipino mixed with Japanese expressions
            "sugoi naman ito", "kawaii ng design", "kakkoii talaga", "subarashii ito",
            "omoshiroi ng content mo", "ganbareh sa next video", "arigatou sa tutorial"
        ]
        
        # DIALECTAL VARIATIONS - POSITIVE (MORE SPECIFIC REGIONS)
        dialectal_positive = [
            # Batangueno
            "ay sus maryosep ang galing", "aba'y magaling nga", "napakagaling mo ga", 
            "maganda ire", "masarap na luto", "aba'y mahusay",
            # Kapampangan
            "masanting ini", "mayap ya iti", "masarap ining lutung", "masanting ka talaga", 
            "masaling gawa", "manyaman ya iti",
            # Ilocano (Northern regions)
            "napintas unay", "nagsayaat", "nabuslon", "nalaing", "naimbag", "napintas nga",
            # Cebuano
            "nindot kini", "nindota ani", "perting nindota", "lami kaayo", "nindot jud kaayo", 
            "grabe ka nindot", "lami ni"
        ]
        
        # COMMON INTERNET COMMENT PATTERNS - POSITIVE
        internet_positive = [
            "First comment! Ang ganda talaga", "Grabe underrated creator to", 
            "Bakit di pa to viral?", "Deserve mo mag trending", "Pa-viral ng content ni kuya/ate",
            "Notification squad! Ang galing as always", "Deserved ng more views",
            "Bakit ang konti pa ng likes?", "Parang ang bilis ng 10 minutes",
            "Di ko namalayan 30 minutes na pala to", "Please make more vids like this",
            "Finally may tutorial na maayos", "Kaadik manood ng content mo",
            "Sasabihin ko lang ang galing", "You never disappoint", "Never failed to amaze me",
            "Always quality content", "Instant like once nakita ko name mo",
            "I've been a fan since day 1", "Automatic like pag ikaw", "Automatic click pag ikaw"
        ]
        
        # COMMON INTERNET COMMENT PATTERNS - NEGATIVE
        internet_negative = [
            "Bakit ko to pinanood", "Waste of 5 minutes", "Di worth it panoorin",
            "Click bait nanaman", "Puro ads lang to", "Skip to 3:45 wala kwenta intro",
            "Mas magaling pa 5-year old pamangkin ko", "Halatang di pinag-isipan",
            "Copy paste lang from other creators", "Bugok naman editing nito",
            "Mag-aral ka muna bago mag-tutorial", "Napaka-basic naman nito",
            "Humina na talaga content mo", "Remember when his/her content was good?",
            "Unsubscribe na ko", "Unfollowed", "Reported for misleading content",
            "Not watching your vids again", "Mas maganda pa content ng kabayo", 
            "Parang ginawa lang in 5 minutes", "Halatang di pinaghirapan"
        ]
        
        # DEMOGRAPHIC-SPECIFIC LANGUAGE - POSITIVE (YOUTH, ELDERLY, PROFESSIONALS)
        demographic_positive = [
            # Gen Z
            "no cap, ang lit nito", "sheeeeshh grabe husay", "main character energy", 
            "ate and left no crumbs", "fr fr ang galing", "lowkey obsessed na ko",
            "hits different talaga content mo", "rent free sa utak ko mga ideas mo",
            # Millennials
            "sobrang relatable nito", "adulting hack na needed ko", "self-care reminder na kailangan ko", 
            "finally something useful sa feed ko", "legit life-changing to",
            # Professionals
            "excellent methodology", "highly informative presentation", "comprehensive analysis", 
            "very professional execution", "technically advanced approach",
            "remarkable demonstration of skill", "commendable attention to detail", 
            "brilliant synthesis of complex ideas", "exceptional quality",
            # Parents
            "nakakatuwa naman ito para sa mga anak natin", "magandang resource para sa pamilya", 
            "educational para sa mga bata", "wholesome content para sa family bonding",
            "magandang ipakita sa mga anak", "safe content para sa buong pamilya"
        ]
        
        # EMOJI-HEAVY COMMENTS - POSITIVE
        emoji_positive = [
            "😍 ang ganda!!!", "❤️❤️❤️ grabeeee", "👏👏👏 saludo ako sayo", 
            "🔥🔥🔥 sobrang galing", "❤️🥰😍 love this!", "👌👌👌 perfect",
            "🙌🙌🙌 idol talaga kita", "😱😱😱 grabe ang galing", "💯💯💯 top tier content",
            "✨✨✨ sobrang inspiring", "🤩🤩🤩 nakakabilib", "💪💪💪 husay!",
            "🥺🥺🥺 ang gandaaaa", "🤗🤗🤗 nakakatuwa", "😘😘😘 love you bhie",
            "🌟🌟🌟 star quality", "💕💕💕 sobrang love ko to", "🙌💕😊 best content ever"
        ]
        
        # EMOJI-HEAVY COMMENTS - NEGATIVE
        emoji_negative = [
            "🙄 ang corny naman", "👎👎👎 di ko bet", "🤦‍♀️🤦‍♀️🤦‍♀️ nakakahiya to",
            "😒😒😒 nakaka-disappoint", "🚮🚮🚮 basura content", "😡😡😡 nakakabwisit",
            "🤢🤢🤢 kadiri", "💩💩💩 pangit", "🙄🙄🙄 overhyped",
            "😴😴😴 boring", "🥱🥱🥱 walang kwenta", "😬😬😬 cringe talaga",
            "😤😤😤 nakakaasar", "👀👀👀 sketchy content", "🧢🧢🧢 cap yan",
            "💀💀💀 kainis", "🙅‍♀️🙅‍♀️🙅‍♀️ no way", "🤮🤮🤮 nakakasuya"
        ]
        
        # COMBINE ALL CATEGORIES
        all_positive = (standard_positive + tiktok_positive + regional_positive + 
                    taglish_positive + short_positive + multilingual_positive +
                    dialectal_positive + internet_positive + demographic_positive +
                    emoji_positive)
        
        all_negative = (standard_negative + tiktok_negative + regional_negative + 
                        taglish_negative + short_negative + internet_negative +
                        emoji_negative)
        
        all_neutral = (standard_neutral + tiktok_neutral + regional_neutral + 
                    taglish_neutral + short_neutral)
        
        # SHUFFLE TO MIX DIFFERENT TYPES
        import random
        random.shuffle(all_positive)
        random.shuffle(all_negative)
        random.shuffle(all_neutral)
        
        # Create training data
        texts = all_positive + all_negative + all_neutral
        labels = (["Positive"] * len(all_positive) + 
                ["Negative"] * len(all_negative) + 
                ["Neutral"] * len(all_neutral))
        
        # Shuffle once more to mix categories
        combined = list(zip(texts, labels))
        random.shuffle(combined)
        texts, labels = zip(*combined)
        
        return list(texts), list(labels)

    # Augment data to create more training samples
def augment_tagalog_data(texts, labels, augmentation_factor=2):
        """
        Augment Tagalog training data by creating variations of existing examples.
        """
        import random
        from copy import deepcopy
        
        augmented_texts = deepcopy(texts)
        augmented_labels = deepcopy(labels)
        
        # Common typos and variations in Tagalog social media
        typo_map = {
            'a': ['ah', 'e'],
            'e': ['eh', 'i'],
            'i': ['e', 'ee'],
            'o': ['oh', 'u'],
            'u': ['o', 'oo'],
            'ng': ['nang', 'g'],
            'po': ['poh', 'pu'],
            'n': ['ng', 'm'],
            'm': ['n'],
            's': ['z', 'c'],
            'c': ['k', 's'],
            'q': ['k'],
            'w': ['u', 'v'],
            'y': ['i'],
            'b': ['v'],
            'v': ['b'],
        }
        
        # Common words that get modified in social media
        word_variations = {
            'ang': ['ung', 'yung', 'yng'],
            'nga': ['ngah', 'nge', 'ngi'],
            'lang': ['lng', 'lamg', 'lamang'],
            'talaga': ['tlga', 'talga', 'talge', 'tlge', 'talaaaga'],
            'sobra': ['sobrang', 'sobraaaa', 'sobraaa', 'sobrang sobra'],
            'maganda': ['mganda', 'ganda', 'gnda', 'magandaaaa'],
            'galing': ['galeng', 'gling', 'galeeeng', 'galinggg'],
            'salamat': ['tnx', 'thanks', 'salamuch', 'tnks', 'slmt'],
            'sana': ['sna', 'sanaaaa', 'sanaaa', 'sna sna'],
            'hindi': ['di', 'hinde', 'hinde', 'hindiii'],
            'grabe': ['grabeee', 'grabiii', 'grabeh', 'garbe']
        }
        
        # Common fillers added in social media comments
        fillers = [
            'hehe', 'haha', 'hihi', 'char', 'charot', 'lol', 'lmao', 'hahaha',
            'ahahaha', 'ngek', 'nyek', 'luh', 'ay', 'uy', 'hoy', 'omg', 'ay nako',
            'jusko', 'grabeeee', 'haysss', 'alams na', 'eme', 'eh', 'diba', 'teh',
            'bhie', 'mhie', 'siz', 'baks', 'mars', 'ghurl', 'beh', 'mamsh', 'cyst'
        ]
        
        # Punctuation and capitalization variations
        punct_variations = [
            lambda s: s + '!!!',
            lambda s: s + '!?!?',
            lambda s: s + '...',
            lambda s: s + '???',
            lambda s: s.upper(),
            lambda s: s.title(),
            lambda s: ''.join([c.upper() if i % 2 == 0 else c.lower() for i, c in enumerate(s)]),
            lambda s: s + '!!! 🙏',
            lambda s: s + ' 😭',
            lambda s: s + ' 💕',
            lambda s: s + ' 😍'
        ]
        
        # Functions to create variations
        augmentation_funcs = [
            # Add random filler at start
            lambda text: f"{random.choice(fillers)} {text}",
            
            # Add random filler at end
            lambda text: f"{text} {random.choice(fillers)}",
            
            # Add random capitalization/punctuation
            lambda text: random.choice(punct_variations)(text),
            
            # Repeat some letters (common in social media emphasis)
            lambda text: ''.join([c*random.randint(1, 3) if c in 'aeiou' and random.random() > 0.7 else c for c in text]),
            
            # Replace words with common variations
            lambda text: ' '.join([word_variations.get(word, [word])[0] if word in word_variations and random.random() > 0.5 
                                else word for word in text.split()]),
            
            # Add spacing variations
            lambda text: ' '.join([word if random.random() > 0.2 else word.replace('', ' ').strip() for word in text.split()]),
            
            # Introduce random typos
            lambda text: ''.join([typo_map.get(c, [c])[0] if c in typo_map and random.random() > 0.8 else c for c in text]),
            
            # Duplicate some words for emphasis (common in Filipino social media)
            lambda text: ' '.join([word + ' ' + word if len(word) > 3 and random.random() > 0.9 else word for word in text.split()]),
        ]
        
        for i, (text, label) in enumerate(zip(texts, labels)):
            # How many variations to create for each example
            variations_to_create = random.randint(1, augmentation_factor)
            
            for _ in range(variations_to_create):
                # Randomly select 1-3 augmentation functions to apply
                selected_funcs = random.sample(augmentation_funcs, k=random.randint(1, 3))
                
                # Apply the selected functions in sequence
                augmented_text = text
                for func in selected_funcs:
                    augmented_text = func(augmented_text)
                
                # Add to augmented dataset
                if augmented_text != text:  # Only add if different from original
                    augmented_texts.append(augmented_text)
                    augmented_labels.append(label)
        
        return augmented_texts, augmented_labels

    # Enhanced combined training function for multilingual sentiment model
def train_multilingual_sentiment_model():
        """
        Create and train an enhanced sentiment model that works well for both
        English and Tagalog, including code-switching and social media language.
        
        Returns the trained model pipeline ready for inference.
        """
        import pandas as pd
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.ensemble import VotingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import accuracy_score, classification_report
        import joblib
        from pathlib import Path
        
        # Import regular sentiment analysis functions
        from sentiment_analysis import generate_training_data
        
        # 1. Generate base training data
        # Get English training data
        english_texts, english_labels = generate_training_data()
        
        # Get enhanced Tagalog training data
        tagalog_texts, tagalog_labels = generate_enhanced_tagalog_training_data()
        
        # 2. Augment Tagalog data to create more variations
        augmented_tagalog_texts, augmented_tagalog_labels = augment_tagalog_data(
            tagalog_texts, tagalog_labels, augmentation_factor=3
        )
        
        # 3. Load user corrections if available
        try:
            corrections_df = pd.read_csv('sentiment_corrections.csv')
            correction_texts = corrections_df['Comment'].tolist()
            correction_labels = corrections_df['Corrected_Sentiment'].tolist()
            
            print(f"Loaded {len(correction_texts)} user corrections for training")
        except:
            correction_texts = []
            correction_labels = []
            print("No user corrections found, training without them")
        
        # 4. Combine all training data
        all_texts = english_texts + augmented_tagalog_texts + correction_texts
        all_labels = english_labels + augmented_tagalog_labels + correction_labels
        
        # 5. Split into training and testing sets (80/20 split)
        X_train, X_test, y_train, y_test = train_test_split(
            all_texts, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )
        
        # 6. Create feature extraction with TF-IDF
        # Add ngrams up to 4 words to capture phrases
        tfidf = TfidfVectorizer(
            max_features=10000,  # Increased features to capture more vocabulary
            ngram_range=(1, 4),  # Extended to 4-grams to capture more phrases
            min_df=2,
            use_idf=True,
            sublinear_tf=True,
            strip_accents='unicode'
        )
        
        # 7. Build an ensemble model with multiple classifiers
        classifiers = [
            ('nb', MultinomialNB()),
            ('lr', LogisticRegression(C=1.0, class_weight='balanced', solver='liblinear', max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
        ]
        
        ensemble = VotingClassifier(estimators=classifiers, voting='soft')
        
        # 8. Create and train the pipeline
        pipeline = Pipeline([
            ('tfidf', tfidf),
            ('classifier', ensemble)
        ])
        
        # Print training information
        print(f"Training multilingual sentiment model with {len(X_train)} examples")
        print(f"English examples: {len(english_texts)}")
        print(f"Tagalog examples: {len(tagalog_texts)}")
        print(f"Augmented Tagalog examples: {len(augmented_tagalog_texts)}")
        print(f"User corrections: {len(correction_texts)}")
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # 9. Evaluate model
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model accuracy on test set: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # 10. Save the model
        try:
            model_path = Path("multilingual_sentiment_model.joblib")
            joblib.dump(pipeline, model_path)
            print(f"Model saved to {model_path}")
        except Exception as e:
            print(f"Could not save model: {e}")
        
        return pipeline

    # Function to predict sentiment using the multilingual model
def predict_multilingual_sentiment(text_series):
        """
        Predict sentiment using the multilingual model (English + Tagalog).
        
        Args:
            text_series: Pandas Series or string containing text
        
        Returns:
            Series of sentiment predictions with confidence
        """
        import re
        import joblib
        from pathlib import Path
        import pandas as pd
        import numpy as np
        
        # Convert to list if it's a single string
        single_input = False
        if isinstance(text_series, str):
            text_series = [text_series]
            single_input = True
        
        # Ensure all inputs are strings
        text_series = [str(text) if text is not None else "" for text in text_series]
        
        # Load the multilingual model
        model_path = Path("multilingual_sentiment_model.joblib")
        
        if model_path.exists():
            try:
                model = joblib.load(model_path)
            except:
                # If loading fails, train a new model
                model = train_multilingual_sentiment_model()
        else:
            # If model doesn't exist, train a new model
            model = train_multilingual_sentiment_model()
        
        # Predict classes
        predictions = model.predict(text_series)
        
        # Get probabilities
        try:
            probabilities = model.predict_proba(text_series)
            confidence_scores = np.max(probabilities, axis=1)
            
            # Format results with confidence scores
            result = [f"{pred} ({conf:.2f})" for pred, conf in zip(predictions, confidence_scores)]
        except:
            # If predict_proba fails, use fixed confidence
            result = [f"{pred} (0.85)" for pred in predictions]
        
        if single_input:
            return result[0]
        
        return pd.Series(result)

    # Function to integrate with existing sentiment analysis
def tagalog_enhanced_sentiment_analysis(text_series):
        """
        Enhanced sentiment analysis with support for both English and Tagalog.
        Uses a combined approach for best results across languages.
        
        Args:
            text_series: Pandas Series or string containing text
        
        Returns:
            Combined sentiment results
        """
        # Import required functions from sentiment_analysis.py
        from sentiment_analysis import (
            analyze_sentiment_vader, 
            analyze_emoji_sentiment,
            analyze_lexicon_sentiment,
            preprocess_for_sentiment
        )
        
        # Convert to list if it's a single string
        single_input = False
        if isinstance(text_series, str):
            text_series = [text_series]
            single_input = True
        
        results = []
        
        for text in text_series:
            if not isinstance(text, str) or not text:
                results.append("Neutral (0.00)")
                continue
            
            # Process text for analysis
            processed = preprocess_for_sentiment(text)
            
            # Get language detection
            is_tag = is_tagalog(text)
            language = "Tagalog" if is_tag else "English"
            
            # Use multilingual prediction model as the primary sentiment analyzer
            try:
                ml_sentiment = predict_multilingual_sentiment(text)
                primary_score_match = re.search(r'\(([-+]?\d+\.\d+)\)', ml_sentiment)
                primary_score = float(primary_score_match.group(1)) if primary_score_match else 0.0
                
                if "Positive" in ml_sentiment:
                    primary_score = abs(primary_score)
                elif "Negative" in ml_sentiment:
                    primary_score = -abs(primary_score)
                else:
                    primary_score = 0.0
            except Exception as e:
                print(f"Error in multilingual model: {e}")
                primary_score = 0.0
            
            # Add emoji analysis for all languages
            emoji_score = analyze_emoji_sentiment(processed['emojis'])
            
            # For English, add VADER analysis
            vader_score = 0.0
            if not is_tag:
                try:
                    vader_sentiment = analyze_sentiment_vader(text)
                    vader_score = float(re.search(r'\(([-+]?\d+\.\d+)\)', vader_sentiment).group(1))
                except:
                    vader_score = 0.0
            
            # For Tagalog, use Tagalog lexicon
            tagalog_score = 0.0
            if is_tag:
                tagalog_score = analyze_tagalog_lexicon_sentiment(text)
            
            # Lexicon sentiment for TikTok-specific terms (works for both languages)
            lexicon_score = analyze_lexicon_sentiment(text)
            
            # Weight the scores based on language
            if is_tag:
                weights = {
                    'primary': 0.55,   # Multilingual model
                    'emoji': 0.15,     # Emoji sentiment
                    'tagalog': 0.20,   # Tagalog lexicon
                    'lexicon': 0.10    # TikTok lexicon
                }
                
                final_score = (
                    primary_score * weights['primary'] +
                    emoji_score * weights['emoji'] +
                    tagalog_score * weights['tagalog'] +
                    lexicon_score * weights['lexicon']
                )
            else:
                weights = {
                    'primary': 0.40,   # Multilingual model
                    'vader': 0.25,     # VADER (English only)
                    'emoji': 0.15,     # Emoji sentiment
                    'lexicon': 0.20    # TikTok lexicon
                }
                
                final_score = (
                    primary_score * weights['primary'] +
                    vader_score * weights['vader'] +
                    emoji_score * weights['emoji'] +
                    lexicon_score * weights['lexicon']
                )
            
            # Determine sentiment category
            if final_score >= 0.05:
                results.append(f"Positive ({final_score:.2f})")
            elif final_score <= -0.05:
                results.append(f"Negative ({final_score:.2f})")
            else:
                results.append(f"Neutral ({final_score:.2f})")
        
        if single_input:
            return results[0]
        
        return pd.Series(results)

    # Updated breakdown function to include multilingual capabilities
def get_tagalog_sentiment_breakdown(text):
        """
        Get detailed breakdown of sentiment scores from different methods.
        Now supports both English and Tagalog text with language detection.
        
        Args:
            text: String text to analyze
        
        Returns:
            Dictionary with sentiment scores from each method
        """
        # Import required functions from sentiment_analysis.py
        from sentiment_analysis import (
            analyze_sentiment_vader, 
            analyze_emoji_sentiment,
            analyze_lexicon_sentiment,
            preprocess_for_sentiment
        )
        
        if not isinstance(text, str) or not text:
            return {
                "vader": 0.0,
                "emoji": 0.0,
                "lexicon": 0.0,
                "tagalog": 0.0,
                "multilingual": 0.0,
                "final": 0.0,
                "language": "unknown",
                "sentiment": "Neutral"
            }
        
        # Process text
        processed = preprocess_for_sentiment(text)
        
        # Detect language
        is_tag = is_tagalog(text)
        language = "Tagalog" if is_tag else "English"
        
        # Multilingual model prediction (primary predictor)
        try:
            ml_sentiment = predict_multilingual_sentiment(text)
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
        
        # VADER sentiment (for English only)
        vader_score = 0.0
        if not is_tag:
            try:
                sid = SentimentIntensityAnalyzer()
                vader_scores = sid.polarity_scores(text)
                vader_score = vader_scores['compound']
            except:
                vader_score = 0.0
        
        # Emoji sentiment (universal)
        emoji_score = analyze_emoji_sentiment(processed['emojis'])
        
        # Lexicon sentiment (language specific)
        lexicon_score = analyze_lexicon_sentiment(text)
        
        # Tagalog lexicon (for Tagalog only)
        tagalog_score = 0.0
        if is_tag:
            tagalog_score = analyze_tagalog_lexicon_sentiment(text)
        
        # Calculate final score based on language
        if is_tag:
            weights = {
                'multilingual': 0.55,
                'emoji': 0.15, 
                'tagalog': 0.20,
                'lexicon': 0.10
            }
            
            final_score = (
                ml_score * weights['multilingual'] +
                emoji_score * weights['emoji'] +
                tagalog_score * weights['tagalog'] +
                lexicon_score * weights['lexicon']
            )
        else:
            weights = {
                'multilingual': 0.40,
                'vader': 0.25,
                'emoji': 0.15,
                'lexicon': 0.20
            }
            
            final_score = (
                ml_score * weights['multilingual'] +
                vader_score * weights['vader'] +
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
            "tagalog": tagalog_score,
            "multilingual": ml_score,
            "final": final_score,
            "language": language,
            "sentiment": sentiment
        }
