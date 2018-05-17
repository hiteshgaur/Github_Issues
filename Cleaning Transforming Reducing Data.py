import string
import re
import pandas as pd
from os import chdir, getcwd
from bs4 import BeautifulSoup
from nltk import sent_tokenize
#import nltk

working_dir = "C:\\github_issues\\"
chdir(working_dir)
getcwd()



#pip install langdetect


from langdetect import detect 

#to testlanguge detection feature
detect("War doesn't show who's right, just who's left.")# 'en' 
detect("Ein, zwei, drei, vier") #'de' `
detect("elke account die nu word gecreert in de register pagina krijgt nu een unique id, waardoor je weet wie wie precies is.dit moest ik maken omdat je anders nie wist wie welke dagen heeft gekozen. hoe ik het heb gemaakt is door ene foreign key te gebruiken in het sql server zodat ik de data op het login tabel exact kon overnemen naar het weekstaat tabel.")
detect("1. go to http://www.spurri.com/defining-the-word-makerspace/ 2. click on learn more button in any of the image and go to that learn more page 3. scroll down in that page and click on settings and observe it is not working drop down is kept it is not working")
#detect(" 0")

import datetime

print(datetime.datetime.now())

default_url = "https://github.com/"
default_url_length = len(default_url)
def processUrl(issue_url):
    print(issue_url)
    issue_url = issue_url[default_url_length+1:-1]
    print(issue_url)
    slash_index = issue_url.find('/')
    print(slash_index)
    repo_owner = issue_url[0:slash_index]
    issue_url = issue_url[slash_index+1:]
    print(issue_url)
    slash_index = issue_url.find('/')
    print(slash_index)
    repo_name = issue_url[0:slash_index]
    issue_number = issue_url[(issue_url.rfind('/') + 1):]
    print(repo_owner)
    print(repo_name)
    print(issue_number)
    return repo_owner, repo_name, issue_number


def cleanMe(html):
    soup = BeautifulSoup(html, "html.parser") # create a new bs4 object from the html data loaded
    for script in soup(["script", "style"]): # remove all javascript and stylesheet code
        script.extract()
    # get text
    text = soup.get_text()
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text

table = str.maketrans('', '', string.punctuation)

for x in range(100, 107):
    df = pd.read_csv("multiplefiles\\subset_{}.csv".format(x), encoding = "ISO-8859-1")
    ## peeking into top 5 and bottom 5 rows
    print(df.head(5))
    ## peeking into columns
    df.columns 
    print(df.dtypes)
    df.rename(columns={'Unnamed: 0':'row_index'}, inplace=True)
    df["IsLanguageEnglish"] = False
    df["Owner"] = ""
    df["RepoName"] = ""
    df["IssueNumber"] = ""
    print(df.dtypes)
    ## getting length of data type
    print(len(df))
    start_index = -1
    end_index = -1
    #url_arr = []
    ## Iterating through each record
    for row_index, row in df.iterrows():
        if(row_index < start_index):
            continue;
        if(end_index > -1 and row_index > end_index):
            break;
        print('\n################################################################################')
        #looging row index, needed to resume manually
        print(row_index)
        #process(row)
        try:
            titleLang = detect(row["issue_title"])
            print(titleLang)
            if(titleLang == "en"):
                bodyLang = detect(row["body"])
                print(bodyLang)
                if(bodyLang == "en"):
                    df.loc[row_index, 'IsLanguageEnglish'] = True
                    repo_owner, repo_name, issue_number = processUrl(str(row["issue_url"]))
                    df.loc[row_index, 'Owner'] = repo_owner
                    df.loc[row_index, 'RepoName'] = repo_name
                    df.loc[row_index, 'IssueNumber'] = issue_number
                    #removing urls
                    body_text_processed = re.sub(r"http\S+", "", row["body"], flags=re.MULTILINE)
                    #body_text_processed = re.sub(r'^https?:\/\/.*[\r\n]*', '', row["body"], flags=re.MULTILINE)
                    #clean html
                    body_text_processed = cleanMe(body_text_processed)
                    #removing punctuations
                    body_text_processed = body_text_processed.translate(table).lower() 
                    # split into sentences
                    #body_text_processed = sent_tokenize(body_text_processed)
                    print(body_text_processed)
                    #body_text_processed = " ".join(body_text_processed)
                    df.loc[row_index, 'body'] = body_text_processed
        except Exception as e:
            print(e)
        print(datetime.datetime.now())
    df = df.drop(['issue_url'], axis=1)   
    df = df.loc[df['IsLanguageEnglish'] == True]
    try:
        df.to_csv("multiplefilesWithLang\\subset_{}.csv".format(x), index=False, encoding = "utf-8")
    except Exception as ex:
        print(ex)
        df.to_csv("multiplefilesWithLang\\subset_{}.csv".format(x), index=False, encoding = "ISO-8859-1")
    df = []
    print(datetime.datetime.now())
