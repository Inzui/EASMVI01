import os, pandas

class DataSetService():
    def __init__(self, directory: str, dataSetType: str) -> None:
        self.dataSetType = dataSetType
        self._directory = directory
        self._filePath = os.path.join(directory, dataSetType, f"{dataSetType}.csv")

        if (not os.path.isdir(self._directory)):
            os.mkdir(self._directory)

    def append(self, identifier: str, coordinatesList: [()]):
        """Appends a line of coordinates to the Data Set CSV. In format 'x0; y0; x1; y1; ...; x20; y20; identifier;'"""
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
    
    def renumber(self, identifier):
        """Renumbers the pictures in the dataset from 0 to n."""
        picturesDir = os.path.join(self._directory, self.dataSetType, identifier)
        pictures = os.listdir(picturesDir)
        
        # First rename all files to prevent double names.
        for picture in pictures:
            os.rename(os.path.join(picturesDir, picture), os.path.join(picturesDir, f"x{picture}"))

        # Renumber all files.
        index = 0
        for picture in pictures:
            os.rename(os.path.join(picturesDir, f"x{picture}"), os.path.join(picturesDir, f"{identifier}.{index}.png"))
            index += 1
    
    @staticmethod
    def unpack(coordinatesList: [()]) -> []:
        """Unpacks a coordinates list from [()] to [x0, y0, x1, y, ...]"""
        return [coordinate for coordinates in coordinatesList for coordinate in coordinates]