import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter

# Download NLTK resources
nltk.download('vader_lexicon')
def apply_styles():
    with open("styles.css", "r") as f:
        styles = f.read()
    st.markdown(f"<style>{styles}</style>", unsafe_allow_html=True)

def main():
    apply_styles() 
    # Main app title with styling
    st.markdown("<h1 style='text-align: center; color: #188a30;'>WhatsApp Chat Analyzer</h1>", unsafe_allow_html=True)
# Styling the sidebar
    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-color: #188a30;
            color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0px 0px 10px 0px rgba(0, 0, 0, 0.1);
        }
        .sidebar .sidebar-content .block-container {
            color: white !important;
        }
        .sidebar .sidebar-content .block-container hr {
            border-top-color: rgba(255, 255, 255, 0.1) !important;
        }
        .sidebar .sidebar-content .block-container .block-container {
            color: #188a30!important;
        }
        .sidebar .sidebar-content .block-container .block-container hr {
            border-top-color: rgba(255, 255, 255, 0.1) !important;
        }
        .sidebar .sidebar-content .block-container .icon_container {
            color: #188a30!important;
        }
        .sidebar .sidebar-content .block-container .stButton {
            background-color: #188a30 !important;
            color: #4CAF50 !important;
            border-radius: 5px;
            border: 1px solid #188a30;
        }
        .sidebar .sidebar-content .block-container .stButton:hover {
            background-color: #4CAF50 !important;
            color: #188a30 !important;
        }
        </style>
        """
        , unsafe_allow_html=True
    )

    # Sidebar content          
    uploaded_files = st.sidebar.file_uploader("Upload WhatsApp Chat Export Files", accept_multiple_files=True, help="Please upload your WhatsApp chat files in .txt format.")


    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            # Process each uploaded file
            data = process_whatsapp_chat(uploaded_file)

            # Displaying results...
            # Chat Summary with styling
            st.subheader('Chat Summary')
            st.write(f"Total number of messages: **{data.shape[0]}**")
            st.write(f"Total number of users: **{data['Author'].nunique()}**")
            st.write(f"Total number of words: **{data['Message'].apply(lambda x: len(x.split())).sum()}**")
            st.write(f"Total number of media shared: **{data['Message'].apply(lambda x: '<Media omitted>' in x).sum()}**")

            # Messages per Author with styling
            st.subheader('Messages per Author')
            st.bar_chart(data['Author'].value_counts(), use_container_width=True)

            # Average Message Length per Author with styling
            st.subheader('Average Message Length per Author')
            st.write(data.groupby('Author')['Message'].apply(lambda x: x.str.len().mean()))

            # Total Messages per Date with styling
            st.subheader('Total Messages per Date')
            messages_per_date = data.groupby(data['Date'].dt.date)["Message"].count()
            st.line_chart(messages_per_date)
            st.write(messages_per_date)

            # Clean the text
            data['Clean_Message'] = data['Message'].apply(clean_text)

            # Split each message into words
            words = data['Clean_Message'].str.split(expand=True).unstack().value_counts()

            # Display the most common words in the chat
            st.subheader('Most Common Words in the Chat:')
            st.write(words.head(10))

            # Word Frequency Analysis with styling
            st.subheader('Word Frequency Analysis')
            words = data['Message'].str.lower().str.split().explode().value_counts()
            st.bar_chart(words.head(20), use_container_width=True)

            # Word Cloud Visualization
            st.subheader('Word Cloud Visualization')
            text = " ".join(message for message in data['Message'].str.lower())
            wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110, background_color='white').generate(text)
            st.image(wordcloud.to_array(), caption='Word Cloud')

            # Extracting emojis from messages
            if 'Message' in data.columns:
                data['Emojis'] = data['Message'].apply(extract_emojis)

                # Flattening the list of emojis
                emojis_list = [emoji for sublist in data['Emojis'] for emoji in sublist]

                # Counting the occurrence of each emoji
                emoji_counts = Counter(emojis_list)

                # Getting the most common emojis
                most_common_emojis = emoji_counts.most_common(10)

                # Plotting the most used emojis with styling
                if most_common_emojis:
                    emojis, frequencies = zip(*most_common_emojis)
                else:
                    emojis, frequencies = [], []

                plt.figure(figsize=(10, 6))
                plt.bar(emojis, frequencies)
                plt.title('Top 10 Emojis Used')
                plt.xlabel('Emoji')
                plt.ylabel('Frequency')
                plt.xticks(rotation=45)
                st.pyplot(plt)

            # Sentiment Analysis
            st.subheader('Sentiment Analysis Results')
            st.write(data.head(20))

            # Plot sentiment analysis
            plot_sentiment_analysis(data)

            # Top 10 users with the most number of messages with styling
            st.subheader('Top 10 Users with Maximum Number of Messages')
            top_10_users = data['Author'].value_counts().head(10)
            st.bar_chart(top_10_users, use_container_width=True)

            # Grouping data by date with styling
            st.subheader('Number of Messages by Date')
            messages_by_date = data.groupby(data['Date'].dt.date).size()
            st.line_chart(messages_by_date)

            # Finding the days with the most number of messages with styling
            st.write("\n")
            st.write(f"The day with the maximum number of messages was **{messages_by_date.idxmax()}** with **{messages_by_date.max()}** messages.")

            # Plotting number of messages by date with styling
            plt.figure(figsize=(14, 6))
            messages_by_date.plot(kind='line', color='skyblue')
            plt.title('Number of Messages by Date')
            plt.xlabel('Date')
            plt.ylabel('Number of Messages')
            plt.xticks(rotation=45)
            plt.grid(True)
            st.pyplot(plt)

            # Plotting the top 10 users with styling
            st.subheader('Top 10 Users with Maximum Number of Messages')
            plt.figure(figsize=(10, 6))
            top_10_users.plot(kind='bar', color='skyblue')
            plt.title('Top 10 Users with Maximum Number of Messages')
            plt.xlabel('Users')
            plt.ylabel('Number of Messages')
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(plt)

            # Plotting most busy day in a week with styling
            st.subheader('Most Busy Day in a Week')
            plt.figure(figsize=(10, 6))
            data['Day_of_week'].value_counts().plot(kind='bar', color='skyblue')
            plt.xlabel('Day of the Week')
            plt.ylabel('Number of Messages')
            plt.xticks(rotation=45)
            st.pyplot(plt)

            # Plotting most busy month with styling
            st.subheader('Most Busy Month')
            fig, ax = plt.subplots(figsize=(10, 6))
            data['Month'].value_counts().plot(kind='bar', color='salmon', ax=ax)
            ax.set_xlabel('Month')
            ax.set_ylabel('Number of Messages')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)

# Function to process WhatsApp chat file
def process_whatsapp_chat(uploaded_file):
    # Read the WhatsApp chat file line by line
    conversation = uploaded_file.readlines()

    # Define functions to process each line of the chat
    def date_time(s):
        pattern = '^([0-9]+)(\/)([0-9]+)(\/)([0-9]+), ([0-9]+):([0-9]+)(?:\s?(am|pm|AM|PM))? -'
        result = re.match(pattern, s)
        if result:
            return True
        return False

    def find_author(s):
        s = s.split(":")
        if len(s) == 2:
            return True
        else:
            return False

    def messages(line):
        splitline = line.split(' - ')
        dateTime = splitline[0]
        date, time = dateTime.split(",")
        message = " ".join(splitline[1:])

        if find_author(message):
            splitmessage = message.split(": ")
            author = splitmessage[0]
            message = " ".join(splitmessage[1:])
        else:
            author = None

        return date, time, author, message

    # Process each line of the conversation
    data = []
    messageBuffer = []
    for line in conversation:
        line = line.decode("utf-8").strip()  # Decode bytes to string and remove leading/trailing whitespace
        if date_time(line):
            if len(messageBuffer) > 0:
                data.append([date, time, author, ' '.join(messageBuffer)])
            messageBuffer.clear()
            date, time, author, message = messages(line)
            messageBuffer.append(message)
        else:
            messageBuffer.append(line)

    # Create a DataFrame from the processed data
    df = pd.DataFrame(data, columns=["Date", 'Time', 'Author', 'Message'])
    df['Date'] = pd.to_datetime(df['Date'])

    # Add day of the week column
    df['Day_of_week'] = df['Date'].dt.day_name()

    # Add month column
    df['Month'] = df['Date'].dt.month_name()

    # Remove rows with missing values
    data = df.dropna()

    # Perform sentiment analysis
    sentiments = SentimentIntensityAnalyzer()
    data["Positive"] = data["Message"].apply(lambda msg: sentiments.polarity_scores(msg)["pos"])
    data["Negative"] = data["Message"].apply(lambda msg: sentiments.polarity_scores(msg)["neg"])
    data["Neutral"] = data["Message"].apply(lambda msg: sentiments.polarity_scores(msg)["neu"])

    return data

# Function to clean text
def clean_text(text):
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Function to extract emojis from messages
def extract_emojis(text):
    emoji_list = []
    emoji_regex = r'[\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251\U0001F004\U0001F0CF\U0001F170-\U0001F251\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+'
    emojis = re.findall(emoji_regex, text)
    for emoji in emojis:
        emoji_list.append(emoji)
    return emoji_list

# Plot sentiment analysis
def plot_sentiment_analysis(data):
    x = sum(data["Positive"])
    y = sum(data["Negative"])
    z = sum(data["Neutral"])

    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [x, y, z]
    colors = ['gold', 'lightcoral', 'lightskyblue']
    explode = (0.1, 0, 0)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    ax.axis('equal')
    ax.set_title('Sentiment Analysis')
    st.pyplot(fig)

# Main function

# Run the app
if __name__ == '__main__':
    main()
