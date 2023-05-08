#import library
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora
import gensim
import numpy as np
from collections import Counter
from pprint import pprint
import os
import csv
from cosine_similarity import calculate_cosine_similarity

file_path_separator = '/' if os.name == 'posix' else '\\'
base_dir = 'test_dataset' + file_path_separator

tokenizer = RegexpTokenizer(r'\w+')
# create English stop words list
en_stop = get_stop_words('en')
# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

def stem_string(string):
    # clean and tokenize document string
    # return list of stemmed strings/words splitted from the main large input string
    raw = string.lower()
    tokens = tokenizer.tokenize(raw)
    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    return stemmed_tokens


def generate_lda_model(data_frames, lda, settings):
    # For LDA make the text with tokenized, normalized and stemmed.
    texts = []
    for doc in data_frames['text']:
        stemmed_doc = stem_string(doc)
        texts.append(stemmed_doc)

    if(settings.debug):
        print("#################### Printing corpus ####################")
        for i in range(len(texts)):
            print('### words in file #', i, '\n', texts[i], '\n')
    
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # Assuming we can split the text into 3 groups. (good text for stock, bad text for stock, meaningless text for stock)
    # We can get split them into more than 3 groups if needed.
    ldamodel = gensim.models.ldamodel.LdaModel(
        corpus, 
        num_topics=lda.num_topics, 
        id2word=dictionary, 
        update_every=lda.update_every, 
        chunksize=lda.chunksize, 
        passes=lda.passess)

    topics_matrix = ldamodel.show_topics(formatted=False, num_words=lda.num_of_words_topic_description)
    
    # each file should be classified for their each topic.
    data_frames['lda_topic'] = -1
    data_frames['lda_pro'] = -1.0
    for i,one_doc in enumerate(texts):
        bow = dictionary.doc2bow(one_doc)
        t = ldamodel.get_document_topics(bow)
        category=0
        max_pro=0.0
        for j in t:
            if max_pro<j[1]:
                category = j[0]
                max_pro = j[1]

        data_frames.at[data_frames.index[i],'lda_topic'] = category
        data_frames.at[data_frames.index[i], 'lda_pro'] = max_pro
    
    print("#################### Printing LDA Results ####################")
    if settings.debug:
        print('\n#################### Printing top ', lda.num_of_words_topic_description, ' words of each topic ####################\n')
        for i in range(lda.num_topics):
            topics_array = np.array(topics_matrix[i][1])
            pri
            nt("Topic #", i, ': ', [str(word[0]) for word in topics_array], '\n')
        # print documents topics
        for doc, doc_bow in zip(texts, corpus):
            print(doc)
            print(ldamodel.get_document_topics(doc_bow))
        
        # print topics words
        top_topics = ldamodel.top_topics(corpus, topn=lda.num_of_words_topic_description)
        print("### topics")
        pprint(top_topics)
    
    export_lda_results_into_excel(data_frames, topics_matrix, texts, lda)
    print("#################### DONE | Generating LDA ####################")
    return ldamodel

def export_lda_results_into_excel(data_frames, topics_matrix, texts, lda):
    file_tites = data_frames['title']
    lda_categories = data_frames['lda_topic']
    lda_pros = data_frames['lda_pro']

    with open(base_dir + 'lda_results.csv', 'w', encoding="utf-8") as out:
        writer = csv.writer(out)
        row = ["" for x in range(5)]
        row[0] = 'Video File Title'
        # row[1] = 'Cosine Similarity between Video Title and LDA Topic'
        row[1] = 'Cosine Similarity between Video Title and Video Content'
        row[2] = 'LDA Topic Number'
        row[3] = 'LDA Topic Description'
        row[4] = 'LDA Pro'
        writer.writerow(row)

        for i in range(len(file_tites)):
            row[0] = file_tites[i]
            stemmed_video_title = stem_string(row[0])
            row[1] = calculate_cosine_similarity(stemmed_video_title, summarize(texts[i], lda.num_of_words_topic_description))
            # row[1] = calculate_cosine_similarity(stemmed_video_title, summarize(texts[i], len(stemmed_video_title)))
            row[2] = lda_categories[i]
            row[3] = [str(word[0]) for word in np.array(topics_matrix[lda_categories[i]][1])]
            row[4] = lda_pros[i]
            writer.writerow(row)

def summarize(stemmed_tokens,n):
    counters = Counter(stemmed_tokens)
    top_n = []
    for pair in counters.most_common(10):
        top_n.append(pair[0])
    return top_n
    