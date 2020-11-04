from os import makedirs
from os.path import isdir

def secure_dir(folder):
    if not isdir(folder):
        makedirs(folder)
