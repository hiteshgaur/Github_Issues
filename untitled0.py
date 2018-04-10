# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 01:38:23 2018

@author: Admin
"""



import pandas as pd


working_dir = "F:\\Projects\\github-issues\\"



for i,chunk in enumerate(pd.read_csv('F:\\Projects\\github-issues\\github_issues.csv', chunksize=500000)):
    chunk.to_csv('F:\\Projects\\github-issues\\multiple\\subset_{}.csv'.format(i))
