import os, pandas

class DataSetService():
    def __init__(self, directory: str, fileName: str) -> None:
        self.directory = directory
        self.filePath = os.path.join(directory, fileName)

        if (not os.path.isdir(self.directory)):
            os.path.os.mkdir(self.directory)

    def append(self, identifier: str, coordinatesList: [()]):
        """Appends a line of coordinates to the Data Set CSV. In format 'x0; y0; x1; y1; ...; identifier;'"""
        line = ""
        for coordinates in coordinatesList:
            line = f"{line};{coordinates[0]};{coordinates[1]}"
        line = f"{line};{identifier}"[1:]

        file = open(self.filePath, 'a')
        file.write(f"{line}\n")
        file.close()

    def load(self) -> pandas.DataFrame:
        """Loads the Pandas DataFrame from the Data Set CSV."""
        return pandas.read_csv(self.filePath)

    def clear(self):
        """Clears the Data Set CSV."""
        open(self.filePath, 'w').close()