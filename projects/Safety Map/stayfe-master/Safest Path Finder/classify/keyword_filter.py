import pandas as pd

weapons = {
    'gun':['gun','shooting', 'shot', 'armed', 'firearm'], 
    'bomb':['explode', 'explosion', 'bomb']
}

crimes = {
    'robbery': ['robbery','robbed', 'robber', 'break'], 
    'murder': ['murder', 'massacre', 'homicide', 'kill', 'stab', 'assassin'], 
    'rape': ['rape', 'rapist', 'sexual assault'],
    'theft': ['theft', 'pick-pocket', 'steal', 'burglar', 'stolen'],
    'assault': ['assault', 'hit', 'punch', 'kick'],
    'kidnap': ['kidnap']
}

def get_keys(text, dic):
    keys = []
    for k, v in dic.items():
        try:
            if any([V in text for V in v]):
                keys.append(k)
        except(TypeError):
            return ''
    return ','.join(keys)

def if_court_case(text):
    return any([v in text for v in ('court', 'pleaded', 'guilt', 'charge')])


if __name__ == '__main__':
    filename = '../data/abc7news_initial_parsed.csv'
    df = pd.read_csv(filename)
    df['Crime_Type'] = [get_keys(text, crimes) for text in df['Body_Text']]
    df['Weapon'] = [get_keys(text, weapons) for text in df['Body_Text']]
    df.to_csv('../data/keyword_filtered.csv')