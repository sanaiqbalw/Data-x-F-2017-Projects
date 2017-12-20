from nltk import word_tokenize
from csv_to_dict import csv2dict

def Read_Abbrev():
    global abb_forwards
    abb_forwards = csv2dict('resources/Abbreviations.csv')

def Abb_Replace(string, start=0):
    '''This function replaces words in the input string with abbreviations from the abb_forward dictionary'''

    split_string = word_tokenize(string)

    #Compares the words from the string to the abb_forwards dictionary
    #If one of the words is in the dictionary it replaces it and exits
    output = []
    i=start
    for x in split_string[start:]:
        i+=1
        if x in abb_forwards:
            new_string = string.replace(x, abb_forwards[x])
            output = output + Abb_Replace(new_string, i)
        
    output.append(string)
    return output