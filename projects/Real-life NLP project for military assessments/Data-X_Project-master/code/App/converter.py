from bullet_rules import *
from abbreviations import *
from synonyms import Syn_Wrapper
from acronyms import ACR_Replace
def RunConvert(string, syn_flag=1):
    '''
    This function takes a string input and returns multiple options for a converted bullet (in a list),
    or it returns -1 to indicate that the input is too short to form a bullet, 
    or it returns -2 to indicate that the bullet is too long
    '''
    #Parameters
    #Upper length limit
    up_lim = 576
    #Lower length limit
    low_lim = 565

    #Initialize output
    output = []


    ##### Bullet Rules ########
    #First we make sure that the bullet is following typical bullet rules
    string = BulletRules(string)

    #Next we test the length of the bullet
    curr_len = CalcStrLen(string, syn_flag)
    #return -1 if too short
    if curr_len < low_lim: 
        return -1
    #if right length, add to output
    elif curr_len <= up_lim:
        output.append(string)


    ##### Acronyms ########
    #We convert acronyms which gives list of possible bullets with acronyms converted
    acronym_output = ACR_Replace(string)
    
    #Add bullets of correct length to output
    j=0
    for i, bullet in enumerate(acronym_output.copy()):
        len_bullet = CalcStrLen(bullet, syn_flag)
        if (len_bullet <= up_lim) and (len_bullet >= low_lim):
            output.append(bullet)
        elif len_bullet < low_lim:
            #if the the bullet is too short, get rid of it
            acronym_output.pop(i-j)
            j+=1
    
    #send acronym_output to smart theaurus to find synonyms


    ##### Synonyms ########
    #We find synonyms which gives us a list of possible options
    syn_output = []
    for bullet in acronym_output:
        syn_output = syn_output + Syn_Wrapper(bullet, syn_flag)

    #Add bullets of correct length to output
    j=0
    for i, bullet in enumerate(syn_output.copy()):
        len_bullet = CalcStrLen(bullet, syn_flag)
        if (len_bullet <= up_lim) and (len_bullet >= low_lim):
            output.append(bullet)
            syn_output.pop(i-j)
            j+=1
        elif len_bullet < low_lim:
            syn_output.pop(i-j)
            j+=1

    #send syn_output to abbreviator to shorten bullets to try to fit into length requirment


    ##### Abbreviations ########
    #We try abbreviations
    Read_Abbrev()
    abbrev_output = []
    for bullet in syn_output:
        abbrev_output = abbrev_output + Abb_Replace(bullet)
    
    #check length of abbreviation outputs
    for bullet in abbrev_output:
        if (CalcStrLen(bullet, syn_flag) <= up_lim) and (CalcStrLen(bullet, syn_flag) >= low_lim):
            output.append(bullet)


    ##### Output ########
    #We now hopefully have an output that we can return.
    #If not, this means the original string was too long.
    if output == []:
        return -2
    else:
        #remove any remaining synonym flags
        if syn_flag:
            output = [bullet.replace("#", "") for bullet in output]
        return list(set(output))