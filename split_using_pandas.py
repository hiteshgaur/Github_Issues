for i,chunk in enumerate(pd.read_csv('F:\\Projects\\github-issues\\github_issues.csv', chunksize=500000)):
    chunk.to_csv('F:\\Projects\\github-issues\\multiple\\subset_{}.csv'.format(i))
