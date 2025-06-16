import json
from tools.date import sec2date, date_formt

# load data from csv
def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        ranks = f.readlines()

    res = []
    for line in ranks:
        line = line.replace('\n', '')
        res.append(json.loads(line))

    return res


# reload for update
def reload(data):
    temp = dict()

    for record in data:
        key = record.pop('TIME')
        temp[key] = record

    return temp


def read_real_loc(path, begin_date):
    with open(path, 'r', encoding="utf-8") as f:
        data = f.readlines()

    res = []
    for line in data:
        temp_dic = {}

        line = line.replace('\n', '').split(',')
        temp_dic['time'], *temp_dic['loc'] = list(map(float, line))
        temp_dic['time'] = date_formt(begin_date) + sec2date(temp_dic['time'])

        res.append(temp_dic)

    return res

if __name__ == '__main__':
    print(read_real_loc(r'../data/real_loc.csv', '2024-10-11 14:00:00'))