import ntpath
import datetime

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def log(msg):
    with open('log.txt', 'a') as fp:
        timestamp =  datetime.datetime.now()
        fp.write(f"[{str(timestamp)}] {msg}\n")
