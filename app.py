import streamlit as st
import pandas as pd
import nltk
from textblob import TextBlob
import plotly.express as px
import io

from nltk.sentiment import SentimentIntensityAnalyzer


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download('vader_lexicon')

# Download NLTK data
nltk.download('punkt')

# Set page title
st.set_page_config(page_title="Document Sentiment Analysis App")


# Define function to calculate sentiment polarity
def get_sentiment(text):
    blob = TextBlob(str(text))
    return blob.sentiment.polarity


# Define main function
def main():
    st.title("Document Sentiment Analysis App")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read CSV file
        df = pd.read_csv(uploaded_file)

        # Calculate sentiment polarity
        df['Sentiment Polarity'] = df.apply(lambda row: get_sentiment(' '.join(map(str, row))), axis=1)

        # Display dataframe
        st.write(df)

        # Create histogram of sentiment polarity
        fig = px.histogram(df, x='Sentiment Polarity', nbins=20)
        st.plotly_chart(fig)

        # Create scatter plot of sentiment polarity vs. row index
        fig2 = px.scatter(df, x=df.index, y='Sentiment Polarity')
        st.plotly_chart(fig2)

        # Create bar chart of number of positive, negative, and neutral rows
        sent_counts = df['Sentiment Polarity'].value_counts()
        labels = ['Positive', 'Negative', 'Neutral']
        values = [sent_counts[x] if x in sent_counts else 0 for x in [1, -1, 0]]
        fig3 = px.bar(x=labels, y=values)
        st.plotly_chart(fig3)

        # Create pie chart of proportion of positive, negative, and neutral rows
        prop_pos = len(df[df['Sentiment Polarity'] > 0]) / len(df)
        prop_neg = len(df[df['Sentiment Polarity'] < 0]) / len(df)
        prop_neu = len(df[df['Sentiment Polarity'] == 0]) / len(df)
        labels2 = ['Positive', 'Negative', 'Neutral']
        values2 = [prop_pos, prop_neg, prop_neu]
        fig4 = px.pie(names=labels2, values=values2)
        st.plotly_chart(fig4)

    # Input from text file
    text_file = st.file_uploader("Upload a text file", type=["txt"])

    def get_emotions(text):
        sid = SentimentIntensityAnalyzer()
        scores = sid.polarity_scores(text)
        emotions = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
        return emotions

    if text_file is not None:
        text = io.TextIOWrapper(text_file)
        text = text.read()

        # Get sentiment polarity
        sentiment = get_sentiment(text)
        st.write("Sentiment Polarity: ", sentiment)

        # Create histogram of sentiment polarity
        fig5 = px.histogram(x=[sentiment], nbins=20)
        st.plotly_chart(fig5)

        # Create scatter plot of sentiment polarity vs. word count
        words = text.split()
        word_count = len(words)
        df_text = pd.DataFrame({'Sentiment Polarity': [sentiment], 'Word Count': [word_count]})
        fig6 = px.scatter(df_text, x='Word Count', y='Sentiment Polarity')
        st.plotly_chart(fig6)

        # Create pie chart of sentiment polarity
        labels3 = ['Positive', 'Negative', 'Neutral']
        values3 = [int(sentiment > 0), int(sentiment < 0), int(sentiment == 0)]
        fig7 = px.pie(names=labels3, values=values3)
        st.plotly_chart(fig7)

        # Get top 10 most frequent emotions words
        emotions = get_emotions(text)
        top_emotions = list(emotions.keys())[:10]
        top_emotions_values = [emotions[e] for e in top_emotions]
        top_emotions_df = pd.DataFrame({'Emotion': top_emotions, 'Score': top_emotions_values})

        # Create bar chart of top 10 most frequent emotions words
        fig8 = px.bar(top_emotions_df, x='Emotion', y='Score', color='Emotion')
        st.plotly_chart(fig8)

    # Input from user
    user_input = st.text_input("Enter text:")
    if user_input:
        sentiment = get_sentiment(user_input)
        st.write("Sentiment Polarity: ", sentiment)

        # Create histogram of sentiment polarity
        fig6 = px.histogram(x=[sentiment], nbins=20)
        st.plotly_chart(fig6)

        # Create scatter plot of sentiment polarity vs. word count
        words = user_input.split()
        word_count = len(words)
        df_text = pd.DataFrame({'Sentiment Polarity': [sentiment], 'Word Count': [word_count]})
        fig6 = px.scatter(df_text, x='Word Count', y='Sentiment Polarity')
        st.plotly_chart(fig6)

        # Create pie chart of sentiment polarity
        labels3 = ['Positive', 'Negative', 'Neutral']
        values3 = [int(sentiment > 0), int(sentiment < 0), int(sentiment == 0)]
        fig7 = px.pie(names=labels3, values=values3)
        st.plotly_chart(fig7)

        # Get top 10 most frequent emotions words
        emotions = get_emotions(user_input)
        top_emotions = list(emotions.keys())[:10]
        top_emotions_values = [emotions[e] for e in top_emotions]
        top_emotions_df = pd.DataFrame({'Emotion': top_emotions, 'Score': top_emotions_values})

        # Create bar chart of top 10 most frequent emotions words
        fig8 = px.bar(top_emotions_df, x='Emotion', y='Score', color='Emotion')
        st.plotly_chart(fig8)


if __name__ == "__main__":
    main()
