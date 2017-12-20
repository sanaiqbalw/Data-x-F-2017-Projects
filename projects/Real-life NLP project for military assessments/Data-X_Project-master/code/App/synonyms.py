from nltk.corpus import wordnet as wn
from nltk import pos_tag, word_tokenize
from pattern.en import conjugate, tenses, pluralize
import re

def synonym_wn (input, output, form = None):
    '''
    Function to provide synonyms for words from the Wordnet corpus.
    
    Input should be a string
    Form refers to the parts of speech, which by default is None. Options include:
    'n'    NOUN 
    'v'    VERB 
    'a'    ADJECTIVE 
    's'    ADJECTIVE SATELLITE 
    'r'    ADVERB 
    
    output takes an integer denoting number of synonyms to output
    '''
    a = []
    synon = []
    counter = 0
    for i,synset in enumerate(wn.synsets(input, form)):
        a.append(synset.lemma_names())

    for lis in a:
        for item in lis:
            if item not in synon: 
                if counter < output: 
                    synon.append(item)
                    counter += 1
    return(synon)

def Bullet_Replace(old_word, new_word, bullet, POS_tag):
    '''
    This function replaces the old_word in bullet with new_word using POS_tag to make 
    the forms of the words match
    '''
    if POS_tag == 'VBD':
        #verb is past tense
        if '3sgp' in tenses(old_word):
            #3rd person singular past
            new_word = conjugate(new_word, '3sgp')
        else:
            #plural past
            new_word = conjugate(new_word, 'ppl')
    elif POS_tag == 'VBG':
        #gerund/present participle
        new_word = conjugate(new_word, 'part')
    elif POS_tag == 'VBN':
        #past participle
        new_word = conjugate(new_word, 'ppart')
    elif POS_tag == 'VBP':
        if '1sg' in tenses(old_word):
            #1st person singular
            new_word = conjugate(new_word, '1sg')
        else:
            #2nd person singular
            new_word = conjugate(new_word, '2sg')
    elif POS_tag == 'VBZ':
        if '3sg' in tenses(old_word):
            new_word = conjugate(new_word, '3sg')
        else:
            new_word = conjugate(new_word, 'pl')
    elif POS_tag in ['NNS', 'NNPS']:
        #need to make new word plural
        new_word = pluralize(new_word)
    
    #check for capitalization
    if old_word[0] != old_word[0].lower():
        new_word = new_word[0].upper() + new_word[1:]
    
    return(bullet.replace(old_word, new_word))

def Syn_Replace(bullet, syn_flag, syn_words=[], num_syn=5, start=0):
    '''
    This function takes in a bullet and the desired maximum number of synonyms to return for each word
    For now it just uses wordnet to get synonyms
    '''
#     parsed_string = pattern.en.parse(bullet).split()[0]
    parsed_string = word_tokenize(bullet)
    parsed_string = pos_tag(parsed_string)
    
    output = []
    i=start
    for x in parsed_string[start:]:
        i+=1
        word = x[0]
        if word in english_stops:
            #ignores stop words
            continue
        elif word == word.upper():
            #ignores acronyms
            continue
        elif x[1][:2] in ['JJ', 'NN', 'VB', 'RB']:
            #only find synonyms of adjectives, nouns, and verbs, and adverbs
            if (syn_flag==0) or (word in syn_words):
                #only find synonyms of the flagged words
                #get the POS letter to give to synonym_wn()
                POS = part_of_speech[x[1]]
                #gives list of synonyms from wordnet
                poss_syns = synonym_wn(word, num_syn, form=POS)
                #add these to output
                for syn in poss_syns:
                    new_string = Bullet_Replace(word, syn, bullet, x[1])
                    output = output + Syn_Replace(new_string, syn_flag, syn_words, num_syn, start=i)
        
    output.append(bullet)
    
    return output

def Syn_Wrapper(bullet, syn_flag):
    #find synonym words if syn_flag is on (=1)
    if syn_flag:
        syn_words = re.findall(r"#[a-zA-Z]+#", bullet)
        syn_words = [word.replace("#", "") for word in syn_words]
        
    #find synonyms
    output = Syn_Replace(bullet, syn_flag, syn_words)
    
    #replace underscores (_) with spaces and the synonym flags
    output = [re.sub(r"#([a-z_A-Z]+)#", r"\1", bullet) for bullet in output]
    output = [bullet.replace("_", " ") for bullet in output]
    
    return output

english_stops = ['i',
 'me',
 'my',
 'myself',
 'we',
 'our',
 'ours',
 'ourselves',
 'you',
 'your',
 'yours',
 'yourself',
 'yourselves',
 'he',
 'him',
 'his',
 'himself',
 'she',
 'her',
 'hers',
 'herself',
 'it',
 'its',
 'itself',
 'they',
 'them',
 'their',
 'theirs',
 'themselves',
 'what',
 'which',
 'who',
 'whom',
 'this',
 'that',
 'these',
 'those',
 'am',
 'is',
 'are',
 'was',
 'were',
 'be',
 'been',
 'being',
 'have',
 'has',
 'had',
 'having',
 'do',
 'does',
 'did',
 'doing',
 'a',
 'an',
 'the',
 'and',
 'but',
 'if',
 'or',
 'because',
 'as',
 'until',
 'while',
 'of',
 'at',
 'by',
 'for',
 'with',
 'about',
 'against',
 'between',
 'into',
 'through',
 'during',
 'before',
 'after',
 'above',
 'below',
 'to',
 'from',
 'up',
 'down',
 'in',
 'out',
 'on',
 'off',
 'over',
 'under',
 'again',
 'further',
 'then',
 'once',
 'here',
 'there',
 'when',
 'where',
 'why',
 'how',
 'all',
 'any',
 'both',
 'each',
 'few',
 'more',
 'most',
 'other',
 'some',
 'such',
 'no',
 'nor',
 'not',
 'only',
 'own',
 'same',
 'so',
 'than',
 'too',
 'very',
 's',
 't',
 'can',
 'will',
 'just',
 'don',
 'should',
 'now',
 'd',
 'll',
 'm',
 'o',
 're',
 've',
 'y',
 'ain',
 'aren',
 'couldn',
 'didn',
 'doesn',
 'hadn',
 'hasn',
 'haven',
 'isn',
 'ma',
 'mightn',
 'mustn',
 'needn',
 'shan',
 'shouldn',
 'wasn',
 'weren',
 'won',
 'wouldn',
 '']

part_of_speech = {
    'NN': 'n',
    'NNS': 'n',
    'NNP': 'n',
    'NNPS': 'n',
    'JJ': 'a',
    'JJR': 'a',
    'JJS': 'a',
    'RB': 'r',
    'RBR': 'r',
    'RBS': 'r',
    'VB': 'v',
    'VBD': 'v',
    'VBG': 'v',
    'VBN': 'v',
    'VBP': 'v',
    'VBZ': 'v'
}