import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords,wordnet
from nltk.stem import WordNetLemmatizer
import re
import nltk
import spacy
import string
pd.options.mode.chained_assignment = None
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

st.set_page_config(page_title="Message Classification")


# Removal of Punctuations
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

# Removal of URLs & Numbers
def remove_urls(text):
    return re.sub('http[s]?://\S+', '', text)

# Removal of Numbers
def remove_numbers(text):
    url_pattern = re.compile(r'[0-9]+')
    return url_pattern.sub('', text)

# Removal of stopwords
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

# Lemmatization
lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

# Implementation of Msg Word Count logic
def Word_Count(data_frame,counter_obj):
    for text in data_frame["Message_body"].values:
        for word in text.split():
            counter_obj[word] += 1
    return counter_obj



def main():
    st.title("Spam Message Classification")

    st.write('Created By: Muhammad Hamza A Khan')
    st.text("")

    # Upload File From System
    uploaded_file = st.file_uploader("Upload CSV", type=".csv")
    use_example_file = st.checkbox("Use example file", False, help="Use in-built example file to demo the app")

    if use_example_file:
        uploaded_file = "SMS_data.csv"
        
    if uploaded_file:
        # Read File
        df = pd.read_csv(uploaded_file, encoding= 'unicode_escape')
        
        df["Message_body"] = df[["Message_body"]].astype(str)
        df['Date_Received'] = pd.to_datetime(df['Date_Received']).dt.date


        df["Message_body"] = (
            df["Message_body"].str.lower() # Lower Casing
            .apply(lambda text: remove_punctuation(text)) #REMOVE PUNCTUATION
            .apply(lambda text: remove_urls(text)) #URLS REMOVED
            .apply(lambda text: remove_numbers(text)) #NUMBERS REMOVED
            )

        # Option for showing dataset
        option_1 = st.checkbox("Show CSV File")
        
        if option_1:
           st.write(df)

        # Creating same type of Label data into different variables  
        spam_Message = df[df['Label']=='Spam']
        nonSpam_Message = df[df['Label']=='Non-Spam']


        cnt_spam = Word_Count(spam_Message,Counter())
        cnt_nonSpam = Word_Count(nonSpam_Message,Counter())

        # Creating word count dataframe, for visualization
        spamMsgWC_df = pd.DataFrame(cnt_spam.most_common(15),columns=['words', 'count'])
        nonSpamMsgWC_df = pd.DataFrame(cnt_nonSpam.most_common(15),columns=['words', 'count'])


        # Removing Stop Words
        df["Message_body"] = df["Message_body"].apply(lambda text: remove_stopwords(text))

        # Creating same type of Label data into different variables for no stop words
        spam_Message_nsw = df[df['Label']=='Spam']
        nonSpam_Message_nsw = df[df['Label']=='Non-Spam']


        cnt_spam_nsw = Word_Count(spam_Message_nsw,Counter())
        cnt_nonSpam_nsw = Word_Count(nonSpam_Message_nsw,Counter())

        # Creating word count dataframe (no stop words), for visualization
        spamMsgWC_nsw_df = pd.DataFrame(cnt_spam_nsw.most_common(15),columns=['words', 'count'])
        nonSpamMsgWC_nsw_df = pd.DataFrame(cnt_nonSpam_nsw.most_common(15),columns=['words', 'count'])


        # Applying Lemmatization
        df["Message_body"] = df["Message_body"].apply(lambda text: lemmatize_words(text))

        # Creating same type of Label data into different variables (applied Lemmatization)
        spam_Message_Lem = df[df['Label']=='Spam']
        nonSpam_Message_Lem = df[df['Label']=='Non-Spam']


        cnt_spam_Lem = Word_Count(spam_Message_Lem,Counter())
        cnt_nonSpam_Lem = Word_Count(nonSpam_Message_Lem,Counter())

        # Creating word count dataframe (applied Lemmatization), for visualization
        spamMsgWC_Lem_df = pd.DataFrame(cnt_spam_Lem.most_common(15),columns=['words', 'count'])
        nonSpamMsgWC_Lem_df = pd.DataFrame(cnt_nonSpam_Lem.most_common(15),columns=['words', 'count'])
        

        msgType = st.selectbox('Message Type',['','Spam','Non-Spam'], format_func=lambda x: 'Select an option' if x == '' else x )
        st.markdown("***")

        spamMsgCount = spam_Message.groupby(['Date_Received'])['Message_body'].count()
        nonSpamMsgCount = nonSpam_Message.groupby(['Date_Received'])['Message_body'].count()

        # Side by Side visualization 
        column_left, column_right = st.columns(2)

        if msgType == 'Spam':

            with column_left:
                # Visualization for common words found in spam msgs (including all words)
                fig, ax = plt.subplots(figsize=(6, 8))
                spamMsgWC_df.sort_values(by='count').plot.barh(x='words',y='count',ax=ax,color="purple")
                ax.grid(b = True, color ='grey',linestyle ='-', linewidth = 0.5,alpha = 2)
                ax.set_title("Common Words Found in Messages (Including All Words)")
                st.pyplot(fig)


                # Visualization for common words found in spam msgs (Applied Lemmatization)
                fig, ax = plt.subplots(figsize=(6, 8))
                spamMsgWC_Lem_df.sort_values(by='count').plot.barh(x='words',y='count',ax=ax,color="purple")
                ax.grid(b = True, color ='grey',linestyle ='-', linewidth = 0.5,alpha = 2)
                ax.set_title("Common Words Found in Messages (Applied Lemmatization)")
                st.pyplot(fig)


            with column_right:
                # Visualization for common words found in spam msgs (Without Stop Words)
                fig, ax = plt.subplots(figsize=(6, 8))
                spamMsgWC_nsw_df.sort_values(by='count').plot.barh(x='words',y='count',ax=ax,color="purple")
                ax.grid(b = True, color ='grey',linestyle ='-', linewidth = 0.5,alpha = 2)
                ax.set_title("Common Words Found in Messages (Without Stop Words)")
                st.pyplot(fig)

                
                # Visualization for no of msgs received over dates
                st.line_chart(spamMsgCount, width=450, height=400,use_container_width=False)
                


        elif msgType == 'Non-Spam':

            with column_left:
                # Visualization for common words found in non-spam msgs (including all words)
                fig, ax = plt.subplots(figsize=(6, 8))
                nonSpamMsgWC_df.sort_values(by='count').plot.barh(x='words',y='count',ax=ax,color="purple")
                ax.grid(b = True, color ='grey',linestyle ='-', linewidth = 0.5,alpha = 2)
                ax.set_title("Common Words Found in Messages (Including All Words)")
                st.pyplot(fig)

                # Visualization for common words found in non spam msgs (Applied Lemmatization)
                fig, ax = plt.subplots(figsize=(6, 8))
                nonSpamMsgWC_Lem_df.sort_values(by='count').plot.barh(x='words',y='count',ax=ax,color="purple")
                ax.grid(b = True, color ='grey',linestyle ='-', linewidth = 0.5,alpha = 2)
                ax.set_title("Common Words Found in Non Spam Msgs (Applied Lemmatization)")
                st.pyplot(fig)


            with column_right:

                # Visualization for common words found in non-spam msgs (Without Stop Words)
                fig, ax = plt.subplots(figsize=(6, 8))
                nonSpamMsgWC_nsw_df.sort_values(by='count').plot.barh(x='words',y='count',ax=ax,color="purple")
                ax.grid(b = True, color ='grey',linestyle ='-', linewidth = 0.5,alpha = 2)
                ax.set_title("Common Words Found in Msgs (Without Stop Words)")
                st.pyplot(fig)

                
                # Visualization for no of msgs received over dates
                st.line_chart(nonSpamMsgCount, width=450, height=400,use_container_width=False)


        st.markdown("***")

if __name__ == '__main__':
    main()
