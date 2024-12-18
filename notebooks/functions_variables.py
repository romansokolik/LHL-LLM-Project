# Description: Functions for text transformation and analysis.
import re
from pprint import pprint
from datasets import load_dataset, DatasetDict
import nltk
from nltk import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
# import the specific tokenizer and model
from transformers import BertTokenizer, BertModel

# Variables
set_names = ['train', 'test', 'unsupervised']
limit = 10  # 25000 for the full dataset
sample_size = 1


def view_apply_function(ds, lim=1, fun=None, names=set_names, pp=False):
    '''
    View sample data from a dataset and optionally apply a function.
    '''
    for name in names:
        if fun: ds[name] = ds[name].map(fun)
        if pp:
            print(f'\n{name} DataSet:')
            pprint(ds[name][:lim])
        else:
            print(f'\n{name} DataSet:\n', ds[name][:lim])


def remove_html_tags(ds):
    # Replace 'text' with the actual field name in your dataset
    ds['text'] = re.sub(r'<.*?>', ' ', ds['text'])
    return ds


def remove_urls(ds):
    ds['text'] = re.sub(r'http\S+', ' ', ds['text'])
    return ds


def remove_punctuation(ds):
    ds['text'] = re.sub(r'[^\w\s]', ' ', ds['text'])
    return ds


def preprocess_chat_text(ds):
    # Expand common abbreviations
    abbreviation_mapping = {
        'lol': 'laugh out loud',
        'brb': 'be right back',
        'omg': 'oh my god'
        # Add more mappings as needed
    }
    # Replace abbreviations with their expanded forms
    for abbreviation, expansion in abbreviation_mapping.items():
        ds['text'] = ds['text'].replace(abbreviation, expansion)
    # Remove emoticons
    ds['text'] = re.sub(r':[a-z]+:', ' ', ds['text'])
    # Normalize common misspellings
    misspelling_mapping = {
        'u': 'you',
        'gr8': 'great',
        # Add more mappings as needed
    }
    # Replace misspelled words with their correct forms
    for misspelling, correction in misspelling_mapping.items():
        ds['text'] = re.sub(r'\b{}\b'.format(misspelling), correction, ds['text'])
    return ds


def remove_stopwords(ds):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(ds['text'])
    words = [word for word in words if word not in stop_words]
    ds['text'] = ' '.join(words)
    return ds


def remove_emojis(ds):
    emoji_pattern = re.compile(
        '['  # Match ranges of emoji Unicode blocks
        '\U0001F600-\U0001F64F'  # Emoticons
        '\U0001F300-\U0001F5FF'  # Symbols & Pictographs
        '\U0001F680-\U0001F6FF'  # Transport & Map Symbols
        '\U0001F1E0-\U0001F1FF'  # Flags
        '\U00002700-\U000027BF'  # Miscellaneous Symbols
        '\U00002600-\U000026FF'  # Dingbats
        '\U000024C2-\U0001F251'  # Enclosed characters
        '\U0001F900-\U0001F9FF'  # Supplemental Symbols and Pictographs
        '\U0001F200-\U0001F2FF'  # Enclosed Ideographic Supplement
        ']+',  # Close the character class
        flags=re.UNICODE,
    )
    # Remove emojis from the text column in the dataset
    ds['text'] = emoji_pattern.sub(r' ', ds['text'])
    return ds


def correct_spelling(ds):
    # Correct the spelling of words in the 'text' column
    try:
        ds['text'] = TextBlob(ds['text']).correct().string
        # print('\n', type(ds['text']), ds['text'])
    except Exception as e:
        print(e)
    return ds


def tokenize_text(ds):
    # return {'text': [set(nltk.word_tokenize(s)) for s in ds['text']]}
    return {
        # 'tokens': [text.split() for text in ds['text']]
        'tokens': set(ds['text'].split())
    }


def tokenize_regex(ds):
    return {
        'tokens': re.findall(r'\w+', ds['text'])
    }
