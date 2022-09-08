import ntpath
import datetime

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def log(fp, msg):
    timestamp =  datetime.datetime.now()
    fp.write(f"[{str(timestamp)}] {msg}")
