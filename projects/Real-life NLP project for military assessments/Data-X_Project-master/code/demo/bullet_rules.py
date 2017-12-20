import re
def text2int(textnum, numwords={}):
    '''
    This function takes a text string as input (textnum) and returns the string with plain english numbers replaced
    with numerals (ex: one hundred and two -> 102). It should be able to handle hyphens such as forty-six -> 46. It 
    walks through the string recursively. It was adapted from the code found here:
    https://stackoverflow.com/questions/493174/is-there-a-way-to-convert-number-words-to-integers
    '''
    if not numwords:
        #this section will only execute if numwords is not passed into the function.
        units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
        ]

        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

        scales = ["hundred", "thousand", "million", "billion", "trillion"]

        numwords["and"] = (1, 0)
        for idx, word in enumerate(units):    numwords[word] = (1, idx)
        for idx, word in enumerate(tens):     numwords[word] = (1, idx * 10)
        for idx, word in enumerate(scales):   numwords[word] = (10 ** (idx * 3 or 2), 0)

    #start at zero
    current = result = 0
    #we need to keep track of the words at the end of the string
    tail = textnum.copy()
    for word in textnum:
        if word not in numwords:
        #if the number is not a numeral convertable word:
            if len(word.split('-')) > 1:  
                for subword in word.split('-'):
                #we know that there are multiple subwords that need to be sorted
                    if subword not in numwords:
                    #if the subword is not a number convertable word:
                        if subword == word.split('-')[-1]:
                        #if the subword is the last word in the hyphen sequence
                            if len(tail)>1:
                            #if there are still words left to walk through: 
                                tail.pop(0)
                                #return the converted number if there were previous conversions and the rest of the converted string
                                if result + current>0: return str(result + current) + '-' + subword + ' ' + text2int(tail, numwords)
                                #else just return the word and the rest of the converted string
                                else: return subword + ' ' + text2int(tail, numwords)

                            else:
                            #else we are at the end of the original string so just return the last piece
                                #return the converted numerals if there were some before the current word
                                if result + current >0: return str(result + current) + '-' + subword
                                #else just return the last word
                                else: return subword
                        elif subword == word.split('-')[0]:
                        #the subword is the first word in the sequence
                            tail.pop(0)
                            tail.insert(0, '-'.join(word.split('-')[1:]))
                            if result + current>0: return str(result + current) + ' ' + subword + '-' + text2int(tail, numwords)
                            #else just return the word and the rest of the converted string
                            else: return subword + '-' + text2int(tail, numwords)
                        else:
                        #the subword is somewhere in the middle of the sequence
                            tail.pop(0)
                            idx = word.split('-').index(subword)
                            tail.insert(0, '-'.join(word.split('-')[idx+1:]))
                            if result + current>0: return str(result + current) + '-' + subword + '-' + text2int(tail, numwords)
                            #else just return the word and the rest of the converted string
                            else: return subword + '-' + text2int(tail, numwords)

                    else:
                    #in this case we have found number convertable words around a hyphen
                        #find the numeral that corresponds to the subword
                        scale, increment = numwords[subword]
                        current = current * scale + increment
                        if scale > 100:
                            result += current
                            current = 0
                #this should only execute if both subwords are in number words   
                tail.pop(0)
            else:
                if len(tail)>1: 
                    tail.pop(0)
                    if result + current>0: return str(result + current) + ' ' + word + ' ' + text2int(tail, numwords)
                    else: return word + ' ' + text2int(tail, numwords)
                else: 
                    if result + current >0: return str(result + current) + ' ' + tail[0]
                    else: return tail[0]
            
        elif word == 'and':
        #if we run into an 'and' we have to be careful because if it is in the middle of a number we need to get rid of it
        #otherwise we should ignore it
            if result+current>0 and tail[0] in numwords:
            #if and is in the middle of a number:
                tail.pop(0)
                scale, increment = numwords[word]
                current = current * scale + increment
                if scale > 100:
                    result += current
                    current = 0
            else:
            #'and' is not in the middle of a number:
                #these next few lines should match the lines where subword not in numwords, refer there for extra comments
                if len(tail)>1: 
                    tail.pop(0)
                    if result + current>0: return str(result + current) + ' ' + word + ' ' + text2int(tail, numwords)
                    else: return word + ' ' + text2int(tail, numwords)
                else: 
                    if result + current >0: return str(result + current) + ' ' + tail[0]
                    else: return tail[0]
        else:
        #if word is in numwords:
            #we effectively remove it from the string by removing it from tail
            tail.pop(0)
            #we then increment our numeral
            scale, increment = numwords[word]
            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0
    
    #again I think this only executes if word is in numwords
    if len(tail) > 0:
        return str(result + current) + ' ' + text2int(tail, numwords)
    else: return str(result + current)
    
def Num_to_Num(string, numwords={}):
    '''This function takes in a string as input and returns the string with all plain english numbers changed to numerals
    It replaces the words thousand, million, and bilion with K, M, and B respectively
    '''
    #we define numwords here without the thousand, million, etc because we will convert those to K, M, etc
    if not numwords:
        units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
        ]

        tens = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

        scales = ["hundred"]

        numwords["and"] = (1, 0)
        for idx, word in enumerate(units):    numwords[word] = (1, idx)
        for idx, word in enumerate(tens):     numwords[word] = (1, (idx+2) * 10)
        for idx, word in enumerate(scales):   numwords[word] = (10 ** (idx * 3 or 2), 0)
    
    #the number converter has trouble with zeros, so we do those manually
    string = string.replace('zero', '0')
    #split the string into a list
    split_string = string.split()
    #run the converter
    string = text2int(split_string, numwords)

    #replace large magnitude descriptors with appropriate abbreviations
    string = string.replace(' thousand', 'K')
    string = string.replace(' million', 'M')
    string = string.replace(' billion', 'B')
    return string

def Common_Sub(string):
    '''This function substitutes common symbols (%, $) for the words they represent'''

    #replace percent with '%'
    string = string.replace(' percent', '%')
    #replace pounds with 'lbs'
    string = string.replace(' pounds', ' lbs')
    #replace '___ dollars' with '$___'
    string = re.sub(r'(\w+) [Dd]ollars', r'$\1', string)

    return string


def BulletRules(string):
    '''This function performs basic substitutions on a string according to the AMC writer's guidance.
    The order these functions are executed is important, as the results from one will be input to the next.
    '''
    #run number converter
    string = Num_to_Num(string)
    #capitalize the first letter of the first word
    string = string[0].upper() + string[1:]
    #run common substitution function
    string = Common_Sub(string)
    #add a bullet to the beginning
    string = '- ' + string
    return string

#define the funciton that will return length of string in Times 12
def CalcStrLen(string):
    '''Function takes string as input and returns the length of the string when rendered in Times 12pt font in MS word or pdf'''
    tot_len = 0
    for char in string:
        tot_len += chardict[char]
    return(tot_len) #should probably be in [565, 576] range