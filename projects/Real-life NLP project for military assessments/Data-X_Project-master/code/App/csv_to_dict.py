import csv

def csv2dict(csv_file):
    '''
    This function takes a string location of a csv file, reads the csv file, and converts it into a dictionary object
    '''
    new_dict = {}
    input_file = open(csv_file, 'r', encoding='ANSI', newline = '\n')
    reader = csv.reader(input_file)
    for row in reader:
        temp_dict = {row[0]:row[1]}
        new_dict.update(temp_dict)
    input_file.close()

    return(new_dict)