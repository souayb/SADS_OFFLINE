from collections.abc import MutableMapping
from contextlib import suppress
import os
import csv 

class Logfiles(MutableMapping):
    """
    Fil base dictionary for storing the app activities information 

    Using dictionay as base function 
    
    """
    def __init__(self, dirname, pairs=(), **kwargs):
        self.dirname = dirname
        with suppress(FileExistsError):
            os.mkdir(self.dirname)
        self.update(pairs, **kwargs)

    def __getitem__(self, key):
        fullname = os.path.join(self.dirname, key)
        try:
            with open(fullname) as file:
                return file.read()
        except FileNotFoundError:
            raise KeyError(key) from None
    def __setitem__(self, key, value, file_type='csv'):
        fullname = os.path.join(self.dirname, key)
        if file_type == 'csv':
            value.to_csv().encode('utf-8')
        

        with open(fullname, mode='w') as file:
            file.write(value)

    def __delitem__(self, key):
        fullname = os.path.join(self.dirname, key)
        try:
            os.remove(fullname)
        except FileNotFoundError:
            raise KeyError(key) from None

    def __len__(self):
        return len(os.listdir(self.dirname))

    def __iter__(self):
        return iter(os.listdir(self.dirname))
    def __repr__(self):

        return f"Lofiles{tuple(self.items())}"

if __name__ == '__main__':
    stark = Logfiles('starks')
    stark['arial'] = 'tomboy'
    stark['sansa'] = 'Bad taste in men'
    




