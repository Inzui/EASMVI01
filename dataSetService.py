import os, pandas

class DataSetService():
    def __init__(self, directory: str, dataSetType: str) -> None:
        self.dataSetType = dataSetType
        self._directory = directory
        self._filePath = os.path.join(directory, dataSetType, f"{dataSetType}.csv")

        if (not os.path.isdir(self._directory)):
            os.path.os.mkdir(self._directory)

    def append(self, identifier: str, coordinatesList: [()]):
        """Appends a line of coordinates to the Data Set CSV. In format 'x0; y0; x1; y1; ...; identifier;'"""
        line = ""
        for coordinates in coordinatesList:
            line = f"{line}{coordinates[0]};{coordinates[1]};"
        line = f"{line}{identifier}"

        file = open(self._filePath, 'a')
        file.write(f"{line}\n")
        file.close()

    def load(self) -> pandas.DataFrame:
        """Loads the Pandas DataFrame from the Data Set CSV."""
        return pandas.read_csv(self._filePath, sep = ';')

    def clear(self):
        """Clears the Data Set CSV."""
        line = ""
        for i in range(21):
            line = f"{line}x{i};y{i};"
        file = open(self._filePath, 'w')
        file.write(f"{line}identifier\n")
        file.close()

    def exists(self) -> bool:
        """Returns if the data set CSV exists."""
        return os.path.isfile(self._filePath)