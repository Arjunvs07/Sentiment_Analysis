import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import spacy
import string
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


df = pd.read_csv("C:\\Users\\ADMIN\\New folder\\healthcare_reviews (1).csv")

most_frequent_sentence = df['Review_Text'].mode()[0]
df['Review_Text'].fillna(most_frequent_sentence, inplace=True)


# Load the English language model in spaCy
nlp = spacy.load('en_core_web_sm')

# Function to process text using spaCy
def process_text(text):
    # Process the text using spaCy
    doc = nlp(text)
    # Remove stop words, punctuation, and convert to lowercase
    # Lemmatize each token and join them back into a sentence
    lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.text.lower()]
    return lemmatized_tokens

# Apply text processing to the Sentences column
df['processed_text'] = df['Review_Text'].apply(process_text)
    

def analyze_sentiment(text):
    
    positive_words = ['good', 'great', 'excellent','happy','satisfied']
    negative_words = ['bad', 'terrible', 'awful','disappointing']
    
    tokens = process_text(text)
    positive_count = sum(1 for word in tokens if word in positive_words)
    negative_count = sum(1 for word in tokens if word in negative_words)
    
    if positive_count > negative_count:
        return 'positive'
    elif negative_count > positive_count:
        return 'negative'
    else:
        return 'neutral'

# Apply sentiment analysis to DataFrame
df['sentiment'] = df['Review_Text'].apply(analyze_sentiment)

df['tokenized_text_str'] = df['processed_text'].apply(lambda tokens: ' '.join(tokens))


st.set_option('deprecation.showPyplotGlobalUse', False)

def plots_of_reviews():
    # Plot a histogram of ratings
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x='Rating', bins=5, kde=True)
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    st.pyplot()

    # Plot a countplot of sentiment
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='sentiment')
    plt.title('Count of Sentiments')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    st.pyplot()

    # Plot a barplot of mean ratings by sentiment
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='sentiment', y='Rating', errorbar=None)
    plt.title('Mean Ratings by Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Mean Rating')
    st.pyplot()


    plt.figure(figsize=(8, 6))
    scatter_plot = sns.scatterplot(data=df, x='tokenized_text_str', y='Rating')
    plt.title('Tokenized Text Length vs. Rating')
    plt.xlabel('Tokenized Text')
    plt.ylabel('Rating')

    # Rotate x-axis labels
    scatter_plot.set_xticklabels(df['tokenized_text_str'].unique(), rotation=45, ha='right')
    st.pyplot()

# Streamlit part

st.set_page_config(layout = "wide")
st.title("HEALTHCARE REVIEWS")
with st.sidebar:
    select = option_menu("Main Menu",["HOME","INSIGHTS"])
if select == "HOME":
    col1,col2 = st.columns(2)
    with col1:
        st.write("")
        st.write("")
        st.image(r"C:\Users\ADMIN\New folder\download (1).jfif")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.image(r"C:\Users\ADMIN\New folder\download (2).jfif")
    with col2:
        st.write("### Health care, or healthcare, is the improvement of health via the prevention, diagnosis, treatment, amelioration or cure of disease, illness, injury, and other physical and mental impairments in people. Health care is delivered by health professionals and allied health fields.")
        st.write("### Healthcare reviews allow the people to select the most appropriate hospitals to get admitted to when fallen ill or affected by a disease. Hence improving the efficiency and care given by the hospitals")

elif select == "INSIGHTS":
    plots_of_reviews()


