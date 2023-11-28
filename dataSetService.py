import os

class DataSetService():
    def __init__(self, directory: str, fileName: str) -> None:
        self.directory = directory
        self.filePath = os.path.join(directory, fileName)

        if (not os.path.isdir(self.directory)):
            os.path.os.mkdir(self.directory)
    
    def clear(self):
        pass
    
    def append(self, identifier, coordinates):
        f = open(self.filePath, 'a')
        f.write('hi there\n')
        f.close()