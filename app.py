#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  29 11:48:08 2020

@author: nella
"""
#imports
import pandas as pd
import numpy as np
import re
import twitter_scraper
import nltk
import datetime as dt
import altair as alt
import matplotlib.pyplot as plt
import streamlit as st
from twitter_scraper import get_tweets
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models import LdaModel, CoherenceModel
from nltk.corpus import wordnet,stopwords
from gensim.corpora import Dictionary
from wordcloud import WordCloud

def main():

    def Homepage():
        st.image('https://pbs.twimg.com/profile_banners/1252962110335844352/1588007342/600x200',use_column_width=True)
        st.markdown("### Dashboard for the Finance In Common 2020 Summit")
        st.write('')
        st.write('This Dashbord aimed to provide statistics and key informations on the event by collecting contents on the twitter account for the event. Please check the menu on the left side to display data.')
        st.write('The Finance in Common Summit is taking place on 12 November 2020 during the Paris Peace Forum, an annual event focused on improving global governance.')
        st.write('ln the context of the Covid-19 pandemic and subsequent global socio-economic crisis, the Finance in Common Summit will stress the crucial role of Public Development Banks in reconciling short-term counter-cyclical responses to the crisis with sustainable recovery measures that will have a long-term impact on the planet and societies.')
        return
    def text_treat(x,y):
        """
        fonction to strip brut words from tweet texts
        """
        z=[t for t in x if t not in y]
        return z
    def build_tweets_data(page_name):
        """
        Input: Twitter page name
        Output: dataframe with tweets details
        """

        tweet_dic = get_tweets(page_name, pages=100)
        tweet_df = pd.DataFrame(tweet_dic)
        # collect in different columns hashtags, urls, photos, videos,
        tweet_df['hashtags'] = tweet_df['entries'].apply(lambda x:x['hashtags'])
        tweet_df['urls'] = tweet_df['entries'].apply(lambda x:x['urls'])
        tweet_df['photos'] = tweet_df['entries'].apply(lambda x:x['photos'])
        tweet_df['videos'] = tweet_df['entries'].apply(lambda x:x['videos'])

        # collect date
        tweet_df['date'] = tweet_df['time'].apply(lambda x:dt.datetime.date(x))
        # collect tags inside tweet text
        tweet_df['tags'] = tweet_df['text'].apply(lambda x:[t for t in x.split() if re.search('^@', t)])
        tweet_df['various links']= tweet_df['hashtags'] + tweet_df['urls'] + tweet_df['photos'] + tweet_df['videos'] + tweet_df['tags']
        # collect token in tweet text
        tweet_df['text_list'] = tweet_df['text'].apply(lambda x: x.split())
        # retrieve brut tweet text(only words)
        tweet_df['text_brut'] = tweet_df.apply(lambda x: text_treat(x['text_list'], x['various links']),axis = 1)

        return tweet_df
    def today_tweets_stats(df):
        """
        Input: dataframe
        Output:
        """
        #
        df_today = df[df['date']==dt.date.today()]
        df_today['hour'] = df_today['time'].apply(lambda x: x.hour)

        if df_today.empty:
            st.title("Today's Tweet Statistics")
            st.write('')
            st.markdown('**No tweets have been posted yet. Kindly check the global statistics for more details**')
        else:
            st.title("Today Tweet Statistics")
            st.markdown('* Total tweets, likes and retweets')
            st.image("https://img.icons8.com/material-sharp/48/000000/twitter-squared.png",caption=f'{df_today.shape[0]} Tweets', use_column_width=False)
            nb_likes = df_today['likes'].sum()
            st.image("https://img.icons8.com/material-sharp/48/000000/like.png", caption=f'{nb_likes} Favorites', use_column_width=False)
            nb_retweets = df_today['retweets'].sum()
            st.image("https://img.icons8.com/material/48/000000/retweet.png", caption=f'{nb_retweets} Retweets', use_column_width=False)
            st.markdown('* Favourite Tweet')
            nb_maxlikes = df_today['likes'].max()
            nb_maxretweet = df_today['retweets'].max()
            st.image("https://img.icons8.com/bubbles/120/000000/facebook-like.png",caption=f'{nb_maxlikes} likes & {nb_maxretweet} retweets', use_column_width=False)
            maxtweet_ind = df_today['text'].iloc[df_today['likes'].argmax()]
            st.write(f'**{maxtweet_ind}**')
            # Display Tweet volume per day
            st.markdown('* Tweet Volume')
            df_stats = df_today.groupby(['hour']).count()
            df_stats['index'] = df_stats.index.astype('str')
            tweet_vol = alt.Chart(df_stats).mark_line().encode(x = alt.X('index', axis=alt.Axis(title='Hour')),
            y = alt.Y('tweetId', axis=alt.Axis(title='Tweet Volume')), color = alt.value('Orange')).properties(width = 800, height=500, title = 'Tweet Volume per Hour').interactive()
            st.altair_chart(tweet_vol, use_container_width=True)

        return

    def global_tweets_stats(df):
        """
        Input: dataframe
        Output: display global stats
        """
        #

        st.title("Global Tweet Statistics")
        st.markdown('* Total tweets, likes and retweets')
        st.image("https://img.icons8.com/material-sharp/48/000000/twitter-squared.png",caption=f'{df.shape[0]} Tweets', use_column_width=False)
        nb_likes = df['likes'].sum()
        st.image("https://img.icons8.com/material-sharp/48/000000/like.png", caption=f'{nb_likes} Favorites', use_column_width=False)
        nb_retweets = df['retweets'].sum()
        st.image("https://img.icons8.com/material/48/000000/retweet.png", caption=f'{nb_retweets} Retweets', use_column_width=False)
        st.markdown('* Favourite Tweet')
        nb_maxlikes = df['likes'].max()
        nb_maxretweet = df['retweets'].max()
        st.image("https://img.icons8.com/bubbles/120/000000/facebook-like.png",caption=f'{nb_maxlikes} likes & {nb_maxretweet} retweets', use_column_width=False)
        maxtweet_ind = df['text'].iloc[df['likes'].argmax()]
        st.write(f'**{maxtweet_ind}**')
        # Display Tweet volume per day
        st.markdown('* Tweet Volume')
        df_stats = df.groupby(['date']).count()
        df_stats['index'] = df_stats.index.astype('str')
        tweet_vol = alt.Chart(df_stats).mark_line().encode(x = alt.X('index', axis=alt.Axis(title='Date')),
        y = alt.Y('tweetId', axis=alt.Axis(title='Tweet Volume')), color = alt.value('green')).properties(width = 800, height=500, title = 'Tweet Volume per Day').interactive()
        st.altair_chart(tweet_vol, use_container_width=True)
        return
    def display_partner(df):
        # concatenate list of tags to create partners list
        partners = df['tags'].sum()

        # Some partners name ends with punct like "," or ".".
        #They need to be removed
        partners_clean = []
        for x in partners:
            if re.search("\W$", x):
                partners_clean.append(x[:-1])
            else:
                partners_clean.append(x)
        #remove heading @
        partners_name = [x.replace('@','',1) for x in partners_clean]
        #remove FinanceInCommon from partner list
        partners_list = list(filter(lambda a: a != 'FinanceInCommon', partners_name))
        #create word cloud dictionnary
        partner_dict = {k:partners_list.count(k) for k in set(partners_list)}
        #Word Cloud display_
        wc = WordCloud(max_font_size=70, background_color="white", max_words=100)
        # generate word cloud
        wc.generate_from_frequencies(partner_dict)
        # show
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title('Key partners for the Finance In Common 2020 Summit')
        st.pyplot()
    def display_hashtags(df):
        # concatenate list of tags to create partners list
        hashtags = df['hashtags'].sum()
        #remove heading #
        hashtags_name = [x.replace('#','',1) for x in hashtags]
        #remove FinanceInCommon
        hashtags_clean = [x for x in hashtags_name if not re.search('^Finance',x)]
        #create word cloud dictionnary
        hashtags_dict = {k:hashtags_clean.count(k) for k in set(hashtags_clean)}
        #Word Cloud display_
        wc = WordCloud(max_font_size=70, background_color="white", max_words=100)
        # generate word cloud
        wc.generate_from_frequencies(hashtags_dict)
        # show
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title('Key Words for the Finance In Common 2020 Summit')
        st.pyplot()
        return

    def get_wordnet_pos(pos_tag):
        # pos_tag translator dor Lemmatization
        output = np.asarray(pos_tag)
        for i in range(len(pos_tag)):
            if pos_tag[i][1].startswith('J'):
                output[i][1] = wordnet.ADJ
            elif pos_tag[i][1].startswith('V'):
                output[i][1] = wordnet.VERB
            elif pos_tag[i][1].startswith('R'):
                output[i][1] = wordnet.ADV
            else:
                output[i][1] = wordnet.NOUN
        return output
    def preprocess(text):
        # Clean and process Tweets raw words
        stop_words = stopwords.words('english')
        text_clean =[word.lower() for word in text if word.isalpha() and word.lower() not in stop_words]
        text_tag = nltk.pos_tag(text_clean)
        lemmatizer = WordNetLemmatizer()
        text_lem = [lemmatizer.lemmatize(word[0],word[1]) for word in get_wordnet_pos(text_tag)]
        text_len = [word for word in text_lem if len(word) > 3] #due to the financial context.. need to purify corpus with impactful word
        # remove frequent words related to the context
        context_stop_words=['summit', 'support', 'collective', 'global', 'development','meet', 'meets', 'meetings', 'pour', 'enfin', 'novembre', 'november', 'event', 'évènement', 'publique', 'meeting', 'public', 'banks', 'pdb', 'pdbs', 'finance', 'finances', 'financial', 'financials', 'sommets', 'bank', 'develop', 'développement', 'banques', 'sommet']
        text_fin = [word for word in text_len if word not in context_stop_words]
        return text_fin

    def event_insight(df):
        """
        Input: dataframe
        Output: display event insight from word analysis
        """
        st.title('Finance In Common Summit Insights')
        st.markdown('* Partners')
        display_partner(df)
        st.write('The Top partners are: **IDFC_Network, ParisPeaceForum, RiouxRemy, AFD_France, UNEP_FI, EDFInetwork**')
        st.markdown('* Hashtags')
        display_hashtags(df)
        st.write("The Top words referencing the event talks are: **COVID19, PDB's, SDGs, ParisAgreement, GreenRecovery**")
        st.markdown('* Topics')
        ## LDA model Building
        #Prepare corpus
        df['text_process'] = df['text_brut'].apply(lambda x: preprocess(x))
        corpus = tweet_df['text_process']
        # Computing the bow (word to ind for each doc)
        id2word = Dictionary(corpus)
        bow = [id2word.doc2bow(line) for line in corpus]
        #perform LDA(previous running state best topic at 2)
        lda = LdaModel(bow, num_topics=2, alpha=0.01, id2word=id2word, passes=10,random_state=0)
        #display word cloud from topic model
        wc = WordCloud(max_font_size=50, background_color="white", max_words=100)
        # generate word cloud
        #wc.generate_from_frequencies(dict(lda.show_topics()[0]))
        wc.fit_words(dict(lda.show_topic(0,100)))
        # show
        plt.imshow(wc, interpolation="bilinear")
        plt.title('First topic: Relative to the Event Interest ')
        plt.axis("off")
        st.pyplot()

        wc.fit_words(dict(lda.show_topic(1,100)))
        # show
        plt.imshow(wc, interpolation="bilinear")
        plt.title('Second topic: Relative to the Summit Objective ')
        plt.axis("off")
        st.pyplot()

        st.write('Topic modelliing perform on the tweets revealed two main topics. The first one refering to the interest around the event(Good, Full, Mondial, Interest, thanks).')
        st.write('The second topic reveals the objective of the summit which is to Build a Bommon Action against the Crisis')
        return

    st.sidebar.title('Finance In Common Twitter Page Insights')
    st.sidebar.image('https://pbs.twimg.com/profile_images/1254533514910982144/ebgtPzI__400x400.jpg', use_column_width=True)
    #st.sidebar.title('Insight on the 1st global meeting of all Public Development Banks')
    #st.sidebar.title(' ')
    st.sidebar.title("Menu")
    app_mode = st.sidebar.selectbox("Please select a page", ["Homepage","Today's Statistics",
                                                                 "Global Statistics",
                                                                 "Event Insights"])
    tweet_df = build_tweets_data('FinanceInCommon')

    if app_mode == 'Homepage':
        Homepage()
    elif app_mode == "Today's Statistics":
        today_tweets_stats(tweet_df)
    elif app_mode == "Global Statistics":
        tweet_df = build_tweets_data('FinanceInCommon')
        global_tweets_stats(tweet_df)
    elif app_mode == "Event Insights":
        event_insight(tweet_df)

    return
if __name__ == "__main__":
    main()
