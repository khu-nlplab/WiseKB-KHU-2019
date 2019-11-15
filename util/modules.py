from typing import List, TextIO, Dict
def log(tag, text)-> None:
    if (tag == 'i')  : print("[INFO] " + text,flush=True)
    elif (tag == 'e'): print("[ERROR] " + text,flush=True)
    elif (tag == 's'): print("[SUCCESS] " + text,flush=True)

def get_data_from_txt(_filename)-> List[str]:
    """
    :return: data List
    """
    dataList :List[str] = []
    f : TextIO = open(file=_filename, mode='r', encoding='utf-8-sig')
    for line in f.readlines():
        line = line.strip()
        if line != '':
            dataList.append(line)
    f.close()
    log('s', "Get {} lines, at {}".format(len(dataList), _filename))
    return dataList
