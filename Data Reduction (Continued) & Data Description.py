
import numpy as np
import pandas as pd
from os import chdir, getcwd

working_dir = "C:\\github_issues\\"
chdir(working_dir)
getcwd()

## following code can generate error, if computer does not have at least 32GB memory
df = pd.read_csv('github_issues.csv')
len(df)
df.dtypes
df.head(10)

#spilitting the large dataset into smaller chunks
for i,chunk in enumerate(pd.read_csv('github_issues.csv', chunksize=50000)):
    print(i)
    chunk.to_csv('multiplefiles\\subset_{}.csv'.format(i))
    print("done")


#reading only last subset
df = pd.read_csv('multiplefiles\\subset_106.csv', encoding = "ISO-8859-1")
len(df)
df.dtypes
df.head(10)
df.tail(10)


#getting total records count before cleaning data
total_records_count = 0    
for x in range(0, 37):
    tempdf = pd.read_csv("multiplefiles\\subset_{}.csv".format(x))
    total_records_count += len(tempdf)

print(total_records_count)


#reading only last subset after cleaning, transforming and reducing for spoken language detection
df_clean = pd.read_csv('multiplefilesWithLang\\subset_106.csv', encoding = "ISO-8859-1")
len(df_clean)
df_clean.dtypes
df_clean.head(5)
df_clean.tail(5)

#getting total records count after cleaning data
total_records_count = 0    
for x in range(0, 37):
    tempdf = pd.read_csv("multiplefilesWithLang\\subset_{}.csv".format(x))
    total_records_count += len(tempdf)

print(total_records_count)



#Execute it after nodejs has scrapped the languages for all the repositories
#In 75 days, only 37 files were processed
total_records_count = 0    
for x in range(0, 37):
    df = pd.read_csv('languageStats\\repoLanguages_{}.csv'.format(x), header=None, names = ['row_number', 'repo_owner', 'repo_name', 'language', 'weightage'])
    total_records_count += len(df)
print(total_records_count)

#getting info on programming languages
unique_languages = df.language.unique()
print(len(unique_languages))
print(unique_languages)

df.head(5)


#Getting info only for C# vs Java
total_c_sharp_repositories = 0
total_java_repositories = 0    
for x in range(0, 37):
    df = pd.read_csv('languageStats\\repoLanguages_{}.csv'.format(x), header=None, names = ['row_number', 'repo_owner', 'repo_name', 'language', 'weightage'])
    grouped_data = df.groupby(['language'])['row_number'].count()
    total_c_sharp_repositories += grouped_data['C#']
    total_java_repositories += grouped_data['Java']
print(total_c_sharp_repositories)
print(total_java_repositories)

#plotting a graph for only C# and Java, with plotly
import plotly.plotly as py
import plotly.graph_objs as go

data = [go.Bar(
            x=['C#', 'Java'],
            y=[total_c_sharp_repositories, total_java_repositories],
            text=[total_c_sharp_repositories, total_java_repositories],
            textposition = 'auto',
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),
            ),
            opacity=0.6
    )]

py.iplot(data, filename='basic-bar')



#plotting a graph for only C# and Java, with matplotlib 
import matplotlib.pyplot as plt

#function to get weighted share in dataframe
def weighted_sum(group, w, length):
    d = group[w]
    return d.sum()/length

#For each subset getting weighted sum and get average for all csv files
list_languages = df.groupby("language").apply(weighted_sum, "weightage", len(df)).sort_values(ascending=False)
type(list_languages)

counts = list_languages.value_counts()
ax = list_languages.iloc[:10].plot(kind="barh")
ax.invert_yaxis()
ax.get_xaxis().set_visible(False)


objects = ('C#', 'Java')
y_pos = np.arange(len(objects))
performance = [total_c_sharp_repositories, total_java_repositories]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Number of Repositories')
plt.title('Repository comparision')
 
plt.show()






#Joining languages and issue datasets
# FOR only C# and Java based repositories
total_c_sharp_issues = 0
total_java_issues = 0    
for y in range(0, 37):
    language_df = pd.read_csv('languageStats\\repoLanguages_{}.csv'.format(x), header=None, 
                     names = ['row_index', 'Owner', 'RepoName', 'language', 'weightage'])
    language_df["weightage"] = language_df["weightage"].str.replace('%', '')
    #transforming weightage column
    language_df["weightage"] = language_df['weightage'].astype(str).astype(float)
    #filtering out dataset for java and c# with greater than 33% weightage
    java_df = language_df.loc[(language_df['weightage'] >= 33) & (language_df['language'] == 'Java')]
    c_sharp_df = language_df.loc[(language_df['weightage'] >= 33) & (language_df['language'] == 'C#')]
    #reading corresponding issues data set
    issue_df = pd.read_csv("multiplefilesWithLang\\subset_{}.csv".format(x))
    #joining both data set
    joined_java_df = pd.merge(issue_df, java_df, how='inner', on=['row_index', 'Owner', 'RepoName'],
                              left_on=None, right_on=None,
         left_index=False, right_index=False, sort=True,
         suffixes=['_i', '_l'], copy=True, indicator=False)
    joined_c_sharp_df = pd.merge(issue_df, c_sharp_df, how='inner', on=['row_index', 'Owner', 'RepoName'],
                                 left_on=None, right_on=None,
         left_index=False, right_index=False, sort=True,
         suffixes=['_i', '_l'], copy=True, indicator=False)
    #only body of issue is required, filtering out other columns
    joined_java_df = joined_java_df.filter(['Owner', 'RepoName', 'IssueNumber', 'body'], axis=1)
    joined_c_sharp_df = joined_c_sharp_df.filter(['Owner', 'RepoName', 'IssueNumber', 'body'], axis=1)
    #Saving it to a new csv file for analysis
    joined_java_df.to_csv('Java\\issues_data_{}.csv'.format(y), index=False )
    joined_c_sharp_df.to_csv('Csharp\\issues_data_{}.csv'.format(y), index=False )
    #adding counts for all files
    total_c_sharp_issues += len(joined_c_sharp_df)
    total_java_issues += len(joined_java_df)


print(total_c_sharp_issues)
print(total_java_issues)


#Getting a barchart with matplotlib
objects = ('C#', 'Java')
y_pos = np.arange(len(objects))
performance = [total_c_sharp_issues, total_java_issues]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Number of issues')
plt.title('Comparision of Issues')
 
plt.show()



#combining C# and Java based issues in one file
from os import chdir, getcwd, listdir, path

def combineResults(language):
    current_directory = path.join(r'C:\\github_issues',language)
    chdir(current_directory)
    print(getcwd())
    directory = listdir(current_directory)
    combined_results = pd.DataFrame([])
    for counter, file in enumerate(directory):
        temp_df = pd.read_csv(file,encoding = "ISO-8859-1")
        combined_results = combined_results.append(temp_df)
    combined_results.to_csv(language+'_issues.csv', index=False, encoding = "ISO-8859-1")
    
combineResults('Java')
combineResults('Csharp')       


#Getting Descriptive Stats
df = pd.read_csv('C:\\github_issues\\Java\\Java_issues.csv',encoding = "ISO-8859-1")
df.columns
df.head(10)
len(df.IssueNumber.unique())
len(df.RepoName.unique())
len(df.Owner.unique())



df = pd.read_csv('C:\\github_issues\\Csharp\\Csharp_issues.csv',encoding = "ISO-8859-1")
df.columns
df.head(10)
len(df.IssueNumber.unique())
len(df.RepoName.unique())
len(df.Owner.unique())




