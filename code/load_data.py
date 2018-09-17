import csv

def load_list(path):
    file = open(path,'r')
    list = file.readlines()
    list = [a.strip('\n') for a in list]
    file.close()
    return list

def save_list(path,list):
    file = open(path,'w')
    for a in list:
        file.write(a+'\n')
    file.close()

def load_csv(path):
    dict = {}
    with open(path) as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                if dict.has_key(row['Image Index']):
                    dict[row['Image Index']] += [row]
                else:
                    dict[row['Image Index']] = [row]
            except:
                None
    return dict
