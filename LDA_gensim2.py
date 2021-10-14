#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 11:04:42 2021

@author: appleuser
"""

import pandas as pd
import numpy as np
dataset = pd.read_csv('/Users/appleuser/Documents/Thesis/db.csv')
dataset.headline[0]
dataset.title[0]
dataset.date[0]
dataset[:1]
dataset["news"] = dataset["title"] + " " + dataset["headline"]
dataset.news[0]
text = dataset[['news']]
text.head(5)



# 특수문자 제거
text['clean_doc'] = text['news'].str.replace("[^a-zA-Z]", " ")
print(text['clean_doc'].head(5))

# 길이가 3이하인 단어는 제거 (길이가 짧은 단어)
text['clean_doc'] = text['clean_doc'].apply(lambda x : ' '.join([w for w in x.split() if len(w)>3]))
print(text['clean_doc'].head(5))

# 전체 단어에 대한 소문자 변환
text['clean_doc'] = text['clean_doc'].apply(lambda x: x.lower())
print(text['clean_doc'].head(5))

# 3인칭 단수를 1인칭으로.. 표제어 추출.
from nltk.stem import WordNetLemmatizer
text['clean_doc'] = text['clean_doc'].apply(lambda x: [WordNetLemmatizer().lemmatize(word, pos='v') for word in x])
print(text['clean_doc'].head(5))

#불용어 제거
from nltk.corpus import stopwords
stop = stopwords.words('english')
text['clean_doc'] = text['clean_doc'].apply(lambda x: [word for word in x if word not in (stop)])
tokenized_doc = text['clean_doc']

print(tokenized_doc.head(5))


# 3인칭 단수를 1인칭으로.. 표제어 추출.
from nltk.stem import WordNetLemmatizer
text['news'] = text['news'].apply(lambda x: [WordNetLemmatizer().lemmatize(word, pos='v') for word in x])
print(text.head(5))





# 각 단어 정수 인코딩 + 빈도수 기록
pip install gensim 
from gensim import corpora
dictionary = corpora.Dictionary(tokenized_doc)
corpus = [dictionary.doc2bow(text) for text in tokenized_doc]
print(corpus[1]) # 수행된 결과에서 두번째 뉴스 출력. 첫번째 문서의 인덱스는 0
print(dictionary[6])
len(dictionary)
print(corpus)


# LDA 모델 훈련시키기
import gensim
NUM_TOPICS = 6 #20개의 토픽, k=20
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
topics = ldamodel.print_topics(num_words=5)
for topic in topics:
    print(topic)


print(ldamodel.print_topics())

# LDA 시각화하기

pip install pyLDAvis

import pyLDAvis.gensim
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
pyLDAvis.save_html(vis, 'lda4.html')
pyLDAvis.display(vis)

import os
os.getcwd()
os.chdir('/Users/appleuser/Documents/Thesis')

# Compute Coherence Score using 
coherence_model_lda = CoherenceModel(model=ldamodel, texts=tokenized_doc, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)



# 문서별 토픽 분포 보기

for i, topic_list in enumerate(ldamodel[corpus]):
    if i==5:
        break
    print(i,'번째 문서의 topic 비율은',topic_list)

# 데이터프레임 만들기
def make_topictable_per_doc(ldamodel, corpus):
    topic_table = pd.DataFrame()

    # 몇 번째 문서인지를 의미하는 문서 번호와 해당 문서의 토픽 비중을 한 줄씩 꺼내온다.
    for i, topic_list in enumerate(ldamodel[corpus]):
        doc = topic_list[0] if ldamodel.per_word_topics else topic_list            
        doc = sorted(doc, key=lambda x: (x[1]), reverse=True)
        # 각 문서에 대해서 비중이 높은 토픽순으로 토픽을 정렬한다.
        # EX) 정렬 전 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (10번 토픽, 5%), (12번 토픽, 21.5%), 
        # Ex) 정렬 후 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (12번 토픽, 21.5%), (10번 토픽, 5%)
        # 48 > 25 > 21 > 5 순으로 정렬이 된 것.

        # 모든 문서에 대해서 각각 아래를 수행
        for j, (topic_num, prop_topic) in enumerate(doc): #  몇 번 토픽인지와 비중을 나눠서 저장한다.
            if j == 0:  # 정렬을 한 상태이므로 가장 앞에 있는 것이 가장 비중이 높은 토픽
                topic_table = topic_table.append(pd.Series([int(topic_num), round(prop_topic,4), topic_list]), ignore_index=True)
                # 가장 비중이 높은 토픽과, 가장 비중이 높은 토픽의 비중과, 전체 토픽의 비중을 저장한다.
            else:
                break
    return(topic_table)


topictable = make_topictable_per_doc(ldamodel, corpus)
topictable = topictable.reset_index() # 문서 번호을 의미하는 열(column)로 사용하기 위해서 인덱스 열을 하나 더 만든다.
topictable.columns = ['문서 번호', '가장 비중이 높은 토픽', '가장 높은 토픽의 비중', '각 토픽의 비중']
topictable[:10]

pd.set_option('display.max_columns', None)

print(dictionary[4])

# wordcloud for each models
# lda is assumed to be the variable holding the LdaModel object
import matplotlib.pyplot as plt
pip install wordcloud
from wordcloud import WordCloud

import PIL
PIL.image.open('*.png')


for t in range(ldamodel.num_topics):
    plt.figure()
    plt.imshow(WordCloud().fit_words(dict(ldamodel.show_topic(t, 200))))
    plt.axis("off")
    plt.title("Topic #" + str(t))
    plt.show()
    
    
### Determine the number of topics

## 1. Prepare Corpus
def prepare_corpus(tokenized_doc):
    """
    Input  : clean document
    Purpose: create term dictionary of our courpus and Converting list of documents (corpus) into Document Term Matrix
    Output : term dictionary and Document Term Matrix
    """
    # Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(tokenized_doc)
    dictionary = corpora.Dictionary(tokenized_doc)
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(text) for text in tokenized_doc]
    # generate LDA model
    return dictionary,doc_term_matrix



## 2. Determine the number of topics

num_topics = 10

def compute_coherence_values(dictionary, doc_term_matrix, tokenized_doc, stop, start=2, step=3):
    """
    Input   : dictionary : Gensim dictionary
              corpus : Gensim corpus
              texts : List of input texts
              stop : Max num of topics
    purpose : Compute c_v coherence for various number of topics
    Output  : model_list : List of LSA topic models
              coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, stop, step):
        # generate LDA model
        model = gensim.models.ldamodel.LdaModel(doc_term_matrix, num_topics=num_topics, id2word = dictionary)  # train model
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=tokenized_doc, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values


## 3. Plot
def plot_graph(tokenized_doc,start, stop, step):
    dictionary,doc_term_matrix=prepare_corpus(tokenized_doc)
    model_list, coherence_values = compute_coherence_values(dictionary, doc_term_matrix,tokenized_doc,
                                                            stop, start, step)
    # Show graph
    x = range(start, stop, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()

start,stop,step=2,12,1
plot_graph(tokenized_doc,start,stop,step)


# evaluate the model : coherence score

from gensim.models import CoherenceModel

coherence_model_lda = CoherenceModel(model=ldamodel, texts=tokenized_doc, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)



print('\nCoherence Score: ', coherence_model_lda)

print('\nPerplexity: ', model.log_perplexity(corpus))



