import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
import re
import nltk
import spacy
import string
pd.options.mode.chained_assignment = None

st.set_page_config(page_title="Message Classification")

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
        df['Date_Received'] = pd.to_datetime(df['Date_Received'])

        # Lower Casing
        df["Message_body"] = df["Message_body"].str.lower()

        # Removal of Punctuations
        PUNCT_TO_REMOVE = string.punctuation
        def remove_punctuation(text):
            return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
        df["Message_body"] = df["Message_body"].apply(lambda text: remove_punctuation(text))

        option_1 = st.checkbox("Show CSV File")
        
        if option_1:
           st.write(df)

        # Creating same type of Label data into different variables  
        spam_Message = df[df['Label']=='Spam']
        nonSpam_Message = df[df['Label']=='Non-Spam']

        # Implementation of Msg Word Count logic
        cnt_spam = Counter()
        cnt_nonSpam = Counter()

        def Word_Count(data_frame,counter_obj):
            for text in data_frame["Message_body"].values:
                for word in text.split():
                    counter_obj[word] += 1

        Word_Count(spam_Message,cnt_spam)
        Word_Count(nonSpam_Message,cnt_nonSpam)

        # Creating word count dataframe, for visualization
        spamMsgWC_df = pd.DataFrame(cnt_spam.most_common(15),columns=['words', 'count'])
        nonSpamMsgWC_df = pd.DataFrame(cnt_nonSpam.most_common(15),columns=['words', 'count'])


        # Removal of stopwords
        STOPWORDS = set(stopwords.words('english'))
        def remove_stopwords(text):
            return " ".join([word for word in str(text).split() if word not in STOPWORDS])

        df["Message_body"] = df["Message_body"].apply(lambda text: remove_stopwords(text))

        # Creating same type of Label data into different variables for no stop words
        spam_Message_nsw = df[df['Label']=='Spam']
        nonSpam_Message_nsw = df[df['Label']=='Non-Spam']

        cnt_spam_nsw = Counter()
        cnt_nonSpam_nsw = Counter()

        Word_Count(spam_Message_nsw,cnt_spam_nsw)
        Word_Count(nonSpam_Message_nsw,cnt_nonSpam_nsw)

        # Creating word count dataframe (no stop words), for visualization
        spamMsgWC_nsw_df = pd.DataFrame(cnt_spam_nsw.most_common(15),columns=['words', 'count'])
        nonSpamMsgWC_nsw_df = pd.DataFrame(cnt_nonSpam_nsw.most_common(15),columns=['words', 'count'])
        

        msgType = st.selectbox('Message Type',['','Spam','Non-Spam'], format_func=lambda x: 'Select an option' if x == '' else x )
        st.markdown("***")

        spamMsgCount = spam_Message.groupby('Date_Received')['Message_body'].count()
        nonSpamMsgCount = nonSpam_Message.groupby('Date_Received')['Message_body'].count()

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

                # Visualization for common words found in spam msgs (Without Stop Words)
                fig, ax = plt.subplots(figsize=(6, 8))
                spamMsgWC_nsw_df.sort_values(by='count').plot.barh(x='words',y='count',ax=ax,color="purple")
                ax.grid(b = True, color ='grey',linestyle ='-', linewidth = 0.5,alpha = 2)
                ax.set_title("Common Words Found in Msgs (Without Stop Words)")
                st.pyplot(fig)

            # Visualization for no of msgs received over dates
            with column_right:
                fig, ax = plt.subplots(figsize=(12, 17))
                spamMsgCount.plot.line(y='Message_body',ax=ax,color="red",marker='o', markerfacecolor='blue')
                ax.grid(b = True, color ='grey',linestyle ='-', linewidth = 0.5,alpha = 2)
                ax.set_title("Number of Messages Recieved over Dates", fontdict={'fontsize': 20, 'fontweight': 'medium'})
                st.pyplot(fig)


        elif msgType == 'Non-Spam':

            with column_left:
                # Visualization for common words found in non-spam msgs (including all words)
                fig, ax = plt.subplots(figsize=(6, 8))
                nonSpamMsgWC_df.sort_values(by='count').plot.barh(x='words',y='count',ax=ax,color="purple")
                ax.grid(b = True, color ='grey',linestyle ='-', linewidth = 0.5,alpha = 2)
                ax.set_title("Common Words Found in Messages (Including All Words)")
                st.pyplot(fig)

                # Visualization for common words found in non-spam msgs (Without Stop Words)
                fig, ax = plt.subplots(figsize=(6, 8))
                nonSpamMsgWC_nsw_df.sort_values(by='count').plot.barh(x='words',y='count',ax=ax,color="purple")
                ax.grid(b = True, color ='grey',linestyle ='-', linewidth = 0.5,alpha = 2)
                ax.set_title("Common Words Found in Msgs (Without Stop Words)")
                st.pyplot(fig)

            # Visualization for no of msgs received over dates
            with column_right:
                fig, ax = plt.subplots(figsize=(12, 17))
                nonSpamMsgCount.plot.line(y='Message_body',ax=ax,color="red",marker='o', markerfacecolor='blue')
                ax.grid(b = True, color ='grey',linestyle ='-', linewidth = 0.5,alpha = 2)
                ax.set_title("Number of Messages Recieved over Dates", fontdict={'fontsize': 20, 'fontweight': 'medium'})
                st.pyplot(fig)

        st.markdown("***")

if __name__ == '__main__':
    main()
