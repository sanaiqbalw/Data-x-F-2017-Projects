# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

input_file = open('C:/Users/jgond_000/test/yfcc100m_autotags/yfcc100m_autotags','r')
output_file = open('output.txt','w')
 
for lines in range(500):
    line = input_file.readline()
    output_file.write(line)
    
df = pd.read_table('output.txt')

input_file2 = open('C:/Users/jgond_000/test/yfcc100m_dataset/yfcc100m_dataset','r')
output_file2 = open('dataset_sample.txt','w')
 
for lines in range(500):
    line1 = input_file2.readline()
    output_file2.write(line1)
    
df2 = pd.read_table('dataset_sample.txt')