import numpy as np
import pandas as pd
from os import chdir, getcwd
#import matplotlib.pyplot as plt
from wordcloud import WordCloud

working_dir = "C:\\github_issues\\"
chdir(working_dir)
getcwd()

import gensim
print(dir(gensim))
from nltk.tokenize import  word_tokenize
import nltk.tag


#Creatig Reference Documents Term frequencies
mobility_df = pd.read_csv('C:\\github_issues\\Rules\\Mobility.csv')

gen_mobility_docs = [[w.lower() for w in word_tokenize(text)] 
                        for text in (mobility_df["Description"].values)]

mobility_dictionary = gensim.corpora.Dictionary(gen_mobility_docs)

mobility_corpus = [mobility_dictionary.doc2bow(gen_doc) for gen_doc in gen_mobility_docs]

mobility_tf_idf = gensim.models.TfidfModel(mobility_corpus)

mobility_sims = gensim.similarities.Similarity('Mobility_temp',mobility_tf_idf[mobility_corpus],
                                      num_features=len(mobility_dictionary))


Cryptography_df = pd.read_csv('C:\\github_issues\\Rules\\Cryptography.csv')

gen_Cryptography_docs = [[w.lower() for w in word_tokenize(text)] 
                        for text in (Cryptography_df["Description"].values)]

Cryptography_dictionary = gensim.corpora.Dictionary(gen_Cryptography_docs)

Cryptography_corpus = [Cryptography_dictionary.doc2bow(gen_doc) for gen_doc in gen_Cryptography_docs]

Cryptography_tf_idf = gensim.models.TfidfModel(Cryptography_corpus)

Cryptography_sims = gensim.similarities.Similarity('Cryptography_temp',Cryptography_tf_idf[Cryptography_corpus],
                                      num_features=len(Cryptography_dictionary))


Portability_df = pd.read_csv('C:\\github_issues\\Rules\\Portability.csv')

gen_Portability_docs = [[w.lower() for w in word_tokenize(text)] 
                        for text in (Portability_df["Description"].values)]

Portability_dictionary = gensim.corpora.Dictionary(gen_Portability_docs)

Portability_corpus = [Portability_dictionary.doc2bow(gen_doc) for gen_doc in gen_Portability_docs]

Portability_tf_idf = gensim.models.TfidfModel(Portability_corpus)

Portability_sims = gensim.similarities.Similarity('Portability_temp',Portability_tf_idf[Portability_corpus],
                                      num_features=len(Portability_dictionary))

Maintainability_df = pd.read_csv('C:\\github_issues\\Rules\\Maintainability.csv')

gen_Maintainability_docs = [[w.lower() for w in word_tokenize(text)] 
                        for text in (Maintainability_df["Description"].values)]

Maintainability_dictionary = gensim.corpora.Dictionary(gen_Maintainability_docs)

Maintainability_corpus = [Maintainability_dictionary.doc2bow(gen_doc) for gen_doc in gen_Maintainability_docs]

Maintainability_tf_idf = gensim.models.TfidfModel(Maintainability_corpus)

Maintainability_sims = gensim.similarities.Similarity('Maintainability_temp',Maintainability_tf_idf[Maintainability_corpus],
                                      num_features=len(Maintainability_dictionary))

Reliability_df = pd.read_csv('C:\\github_issues\\Rules\\Reliability.csv')

gen_Reliability_docs = [[w.lower() for w in word_tokenize(text)] 
                        for text in (Reliability_df["Description"].values)]

Reliability_dictionary = gensim.corpora.Dictionary(gen_Reliability_docs)

Reliability_corpus = [Reliability_dictionary.doc2bow(gen_doc) for gen_doc in gen_Reliability_docs]

Reliability_tf_idf = gensim.models.TfidfModel(Reliability_corpus)

Reliability_sims = gensim.similarities.Similarity('Reliability_temp',Reliability_tf_idf[Reliability_corpus],
                                      num_features=len(Reliability_dictionary))



Globalization_df = pd.read_csv('C:\\github_issues\\Rules\\Globalization.csv')

gen_Globalization_docs = [[w.lower() for w in word_tokenize(text)] 
                        for text in (Globalization_df["Description"].values)]

Globalization_dictionary = gensim.corpora.Dictionary(gen_Globalization_docs)

Globalization_corpus = [Globalization_dictionary.doc2bow(gen_doc) for gen_doc in gen_Globalization_docs]

Globalization_tf_idf = gensim.models.TfidfModel(Globalization_corpus)

Globalization_sims = gensim.similarities.Similarity('Globalization_temp',Globalization_tf_idf[Globalization_corpus],
                                      num_features=len(Globalization_dictionary))



Interoperability_df = pd.read_csv('C:\\github_issues\\Rules\\Interoperability.csv')

gen_Interoperability_docs = [[w.lower() for w in word_tokenize(text)] 
                        for text in (Interoperability_df["Description"].values)]

Interoperability_dictionary = gensim.corpora.Dictionary(gen_Interoperability_docs)

Interoperability_corpus = [Interoperability_dictionary.doc2bow(gen_doc) for gen_doc in gen_Interoperability_docs]

Interoperability_tf_idf = gensim.models.TfidfModel(Interoperability_corpus)

Interoperability_sims = gensim.similarities.Similarity('Interoperability_temp',Interoperability_tf_idf[Interoperability_corpus],
                                      num_features=len(Interoperability_dictionary))



Performance_df = pd.read_csv('C:\\github_issues\\Rules\\Performance.csv')

gen_Performance_docs = [[w.lower() for w in word_tokenize(text)] 
                        for text in (Performance_df["Description"].values)]

Performance_dictionary = gensim.corpora.Dictionary(gen_Performance_docs)

Performance_corpus = [Performance_dictionary.doc2bow(gen_doc) for gen_doc in gen_Performance_docs]

Performance_tf_idf = gensim.models.TfidfModel(Performance_corpus)

Performance_sims = gensim.similarities.Similarity('Performance_temp',Performance_tf_idf[Performance_corpus],
                                      num_features=len(Performance_dictionary))



Usage_df = pd.read_csv('C:\\github_issues\\Rules\\Usage.csv')

gen_Usage_docs = [[w.lower() for w in word_tokenize(text)] 
                        for text in (Usage_df["Description"].values)]

Usage_dictionary = gensim.corpora.Dictionary(gen_Usage_docs)

Usage_corpus = [Usage_dictionary.doc2bow(gen_doc) for gen_doc in gen_Usage_docs]

Usage_tf_idf = gensim.models.TfidfModel(Usage_corpus)

Usage_sims = gensim.similarities.Similarity('Usage_temp',Usage_tf_idf[Usage_corpus],
                                      num_features=len(Usage_dictionary))


Naming_df = pd.read_csv('C:\\github_issues\\Rules\\Naming.csv')

gen_Naming_docs = [[w.lower() for w in word_tokenize(text)] 
                        for text in (Naming_df["Description"].values)]

Naming_dictionary = gensim.corpora.Dictionary(gen_Naming_docs)

Naming_corpus = [Naming_dictionary.doc2bow(gen_doc) for gen_doc in gen_Naming_docs]

Naming_tf_idf = gensim.models.TfidfModel(Naming_corpus)

Naming_sims = gensim.similarities.Similarity('Naming_temp',Naming_tf_idf[Naming_corpus],
                                      num_features=len(Naming_dictionary))


Security_df = pd.read_csv('C:\\github_issues\\Rules\\Security.csv')

gen_Security_docs = [[w.lower() for w in word_tokenize(text)] 
                        for text in (Security_df["Description"].values)]

Security_dictionary = gensim.corpora.Dictionary(gen_Security_docs)

Security_corpus = [Security_dictionary.doc2bow(gen_doc) for gen_doc in gen_Security_docs]

Security_tf_idf = gensim.models.TfidfModel(Security_corpus)

Security_sims = gensim.similarities.Similarity('Security_temp',Security_tf_idf[Security_corpus],
                                      num_features=len(Security_dictionary))


Design_df = pd.read_csv('C:\\github_issues\\Rules\\Design.csv')

gen_Design_docs = [[w.lower() for w in word_tokenize(text)] 
                        for text in (Design_df["Description"].values)]

Design_dictionary = gensim.corpora.Dictionary(gen_Design_docs)

Design_corpus = [Design_dictionary.doc2bow(gen_doc) for gen_doc in gen_Design_docs]

Design_tf_idf = gensim.models.TfidfModel(Design_corpus)

Design_sims = gensim.similarities.Similarity('Design_temp',Design_tf_idf[Design_corpus],
                                      num_features=len(Design_dictionary))




# getting target data to to analysis similarity of issues with description of code analysis rules
csharp_df = pd.read_csv('C:\\github_issues\\Csharp\\Csharp_issues.csv',encoding = "ISO-8859-1")
csharp_df['body'] = csharp_df["body"].apply(lambda x: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))

java_df = pd.read_csv('C:\\github_issues\\Java\\Java_issues.csv',encoding = "ISO-8859-1")
java_df['body'] = java_df["body"].apply(str).apply(lambda x: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))


#Creating a function to extract nouns from issues data
def extractEntity(issue_body):
    tagged_names = nltk.tag.pos_tag(issue_body.split())
    edited_words = [word for word,tag in tagged_names if tag == 'NN' or tag == 'NNP' or tag == 'NNPS']
    return ' '.join(edited_words)


## Functions for each category to analyze similarity 
def getSimilarityArrayCryptography(query_text):
    query_doc = [w.lower() for w in word_tokenize(query_text)]
    print(query_doc)
    query_doc_bow = Cryptography_dictionary.doc2bow(query_doc)
    print(query_doc_bow)
    query_doc_tf_idf = Cryptography_tf_idf[query_doc_bow]
    print(query_doc_tf_idf)
    return max(Cryptography_sims[query_doc_tf_idf])

def getSimilarityArrayPortability(query_text):
    query_doc = [w.lower() for w in word_tokenize(query_text)]
    print(query_doc)
    query_doc_bow = Portability_dictionary.doc2bow(query_doc)
    print(query_doc_bow)
    query_doc_tf_idf = Portability_tf_idf[query_doc_bow]
    print(query_doc_tf_idf)
    return max(Portability_sims[query_doc_tf_idf])


def getSimilarityArrayMaintainability(query_text):
    query_doc = [w.lower() for w in word_tokenize(query_text)]
    print(query_doc)
    query_doc_bow = Maintainability_dictionary.doc2bow(query_doc)
    print(query_doc_bow)
    query_doc_tf_idf = Maintainability_tf_idf[query_doc_bow]
    print(query_doc_tf_idf)
    return max(Maintainability_sims[query_doc_tf_idf])


def getSimilarityArrayReliability(query_text):
    query_doc = [w.lower() for w in word_tokenize(query_text)]
    print(query_doc)
    query_doc_bow = Reliability_dictionary.doc2bow(query_doc)
    print(query_doc_bow)
    query_doc_tf_idf = Reliability_tf_idf[query_doc_bow]
    print(query_doc_tf_idf)
    return max(Reliability_sims[query_doc_tf_idf])


def getSimilarityArrayGlobalization(query_text):
    query_doc = [w.lower() for w in word_tokenize(query_text)]
    print(query_doc)
    query_doc_bow = Globalization_dictionary.doc2bow(query_doc)
    print(query_doc_bow)
    query_doc_tf_idf = Globalization_tf_idf[query_doc_bow]
    print(query_doc_tf_idf)
    return max(Globalization_sims[query_doc_tf_idf])


def getSimilarityArrayInteroperability(query_text):
    query_doc = [w.lower() for w in word_tokenize(query_text)]
    print(query_doc)
    query_doc_bow = Interoperability_dictionary.doc2bow(query_doc)
    print(query_doc_bow)
    query_doc_tf_idf = Interoperability_tf_idf[query_doc_bow]
    print(query_doc_tf_idf)
    return max(Interoperability_sims[query_doc_tf_idf])


def getSimilarityArrayPerformance(query_text):
    query_doc = [w.lower() for w in word_tokenize(query_text)]
    print(query_doc)
    query_doc_bow = Performance_dictionary.doc2bow(query_doc)
    print(query_doc_bow)
    query_doc_tf_idf = Performance_tf_idf[query_doc_bow]
    print(query_doc_tf_idf)
    return max(Performance_sims[query_doc_tf_idf])


def getSimilarityArrayUsage(query_text):
    query_doc = [w.lower() for w in word_tokenize(query_text)]
    print(query_doc)
    query_doc_bow = Usage_dictionary.doc2bow(query_doc)
    print(query_doc_bow)
    query_doc_tf_idf = Usage_tf_idf[query_doc_bow]
    print(query_doc_tf_idf)
    return max(Usage_sims[query_doc_tf_idf])




def getSimilarityArrayNaming(query_text):
    query_doc = [w.lower() for w in word_tokenize(query_text)]
    print(query_doc)
    query_doc_bow = Naming_dictionary.doc2bow(query_doc)
    print(query_doc_bow)
    query_doc_tf_idf = Naming_tf_idf[query_doc_bow]
    print(query_doc_tf_idf)
    return max(Naming_sims[query_doc_tf_idf])



def getSimilarityArraySecurity(query_text):
    query_doc = [w.lower() for w in word_tokenize(query_text)]
    print(query_doc)
    query_doc_bow = Security_dictionary.doc2bow(query_doc)
    print(query_doc_bow)
    query_doc_tf_idf = Security_tf_idf[query_doc_bow]
    print(query_doc_tf_idf)
    return max(Security_sims[query_doc_tf_idf])

def getSimilarityArrayDesign(query_text):
    query_doc = [w.lower() for w in word_tokenize(query_text)]
    print(query_doc)
    query_doc_bow = Design_dictionary.doc2bow(query_doc)
    print(query_doc_bow)
    query_doc_tf_idf = Design_tf_idf[query_doc_bow]
    print(query_doc_tf_idf)
    return max(Design_sims[query_doc_tf_idf])

def getSimilarityArrayMobility(query_text):
    query_doc = [w.lower() for w in word_tokenize(query_text)]
    print(query_doc)
    query_doc_bow = mobility_dictionary.doc2bow(query_doc)
    print(query_doc_bow)
    query_doc_tf_idf = mobility_tf_idf[query_doc_bow]
    print(query_doc_tf_idf)
    return max(mobility_sims[query_doc_tf_idf])



## Applying similarity test function for both Java and Csharp based data set
##       and creating a new column for each category to store similarity score

csharp_df["Mobility_Arr"] = csharp_df["body"].astype(str).apply(getSimilarityArrayMobility)
csharp_df["Cryptography_Arr"] = csharp_df["body"].astype(str).apply(getSimilarityArrayCryptography)
csharp_df["Portability_Arr"] = csharp_df["body"].astype(str).apply(getSimilarityArrayPortability)
csharp_df["Maintainability_Arr"] = csharp_df["body"].astype(str).apply(getSimilarityArrayMaintainability)
csharp_df["Reliability_Arr"] = csharp_df["body"].astype(str).apply(getSimilarityArrayReliability)
csharp_df["Globalization_Arr"] = csharp_df["body"].astype(str).apply(getSimilarityArrayGlobalization)
csharp_df["Interoperability_Arr"] = csharp_df["body"].astype(str).apply(getSimilarityArrayInteroperability)
csharp_df["Performance_Arr"] = csharp_df["body"].astype(str).apply(getSimilarityArrayPerformance)
csharp_df["Naming_Arr"] = csharp_df["body"].astype(str).apply(getSimilarityArrayNaming)
csharp_df["Usage_Arr"] = csharp_df["body"].astype(str).apply(getSimilarityArrayUsage)
csharp_df["Security_Arr"] = csharp_df["body"].astype(str).apply(getSimilarityArraySecurity)
csharp_df["Design_Arr"] = csharp_df["body"].astype(str).apply(getSimilarityArrayDesign)


java_df["Mobility_Arr"] = java_df["body"].astype(str).apply(getSimilarityArrayMobility)
java_df["Cryptography_Arr"] = java_df["body"].astype(str).apply(getSimilarityArrayCryptography)
java_df["Portability_Arr"] = java_df["body"].astype(str).apply(getSimilarityArrayPortability)
java_df["Maintainability_Arr"] = java_df["body"].astype(str).apply(getSimilarityArrayMaintainability)
java_df["Reliability_Arr"] = java_df["body"].astype(str).apply(getSimilarityArrayReliability)
java_df["Globalization_Arr"] = java_df["body"].astype(str).apply(getSimilarityArrayGlobalization)
java_df["Interoperability_Arr"] = java_df["body"].astype(str).apply(getSimilarityArrayInteroperability)
java_df["Performance_Arr"] = java_df["body"].astype(str).apply(getSimilarityArrayPerformance)
java_df["Naming_Arr"] = java_df["body"].astype(str).apply(getSimilarityArrayNaming)
java_df["Usage_Arr"] = java_df["body"].astype(str).apply(getSimilarityArrayUsage)
java_df["Security_Arr"] = java_df["body"].astype(str).apply(getSimilarityArraySecurity)
java_df["Design_Arr"] = java_df["body"].astype(str).apply(getSimilarityArrayDesign)


# testing the modified data frames
csharp_df.head(5)
java_df.head(5)


### creating a plot to compare the average score for a category of both datasets
import matplotlib.pyplot as plt
#from basic_units import cm, inch

N = 12
csharpMeans = (csharp_df["Mobility_Arr"].mean(), 
csharp_df["Cryptography_Arr"].mean(), 
csharp_df["Portability_Arr"].mean(), 
csharp_df["Maintainability_Arr"].mean(), 
csharp_df["Reliability_Arr"].mean(), 
csharp_df["Globalization_Arr"].mean(), 
csharp_df["Interoperability_Arr"].mean(), 
csharp_df["Performance_Arr"].mean(), 
csharp_df["Naming_Arr"].mean(), 
csharp_df["Usage_Arr"].mean(), 
csharp_df["Security_Arr"].mean(), 
csharp_df["Design_Arr"].mean())


fig, ax = plt.subplots()

ind = np.arange(N)    # the x locations for the groups
width = 0.35         # the width of the bars
p1 = ax.bar(ind, csharpMeans, width, alpha=0.5, color='#F78F1E', bottom=0)


javaMeans = (java_df["Mobility_Arr"].mean(), 
java_df["Cryptography_Arr"].mean(), 
java_df["Portability_Arr"].mean(), 
java_df["Maintainability_Arr"].mean(), 
java_df["Reliability_Arr"].mean(), 
java_df["Globalization_Arr"].mean(), 
java_df["Interoperability_Arr"].mean(), 
java_df["Performance_Arr"].mean(), 
java_df["Naming_Arr"].mean(), 
java_df["Usage_Arr"].mean(), 
java_df["Security_Arr"].mean(), 
java_df["Design_Arr"].mean())

p2 = ax.bar(ind + width, javaMeans, width, alpha=0.5,
            color='#FFC222', bottom=0)

ax.set_title('Scores by C# and Java')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('Mobility','Cryptographic', 'Portability', 'Maintainability', 'Reliability', 'Globalization'
                    , 'Interoperability', 'Performance', 'Naming', 'Usage', 'Security', 'Design'))

ax.legend((p1[0], p2[0]), ('C#', 'Java'))
#ax.yaxis.set_units(inch)
ax.autoscale_view()
plt.xticks(rotation=45)
plt.show()










