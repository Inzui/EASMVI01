from photoProcessor import PhotoProcessor

class Main():
    def __init__(self, photoProcessor: PhotoProcessor) -> None:
        self.photoProcessor = photoProcessor

    def run(self):
        for i in range(50):
            try:
                print(self.photoProcessor.run(f"C:\\Users\\ianzu\\OneDrive - Hogeschool Rotterdam\\Machine Vision\\Dataset\\Training\\A\\A.{i}.png"))
            except:
                print(f"Rejected image '{i}'")

if __name__ == "__main__":
    main = Main(PhotoProcessor())
    main.run()