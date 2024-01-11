<h1>EASMVI01</h1>
Machine Vision Assignment for the Minor Mechatronics by I. Zuiderent (1004784) & J. van Wamelen (1009652). This application recognizes the right hand and the letters ‘e’, ‘n’, ‘a’, ‘t’, ‘i’, ‘r’, ‘o’, ‘d’, ‘s’, and ‘l’ of the Dutch sign language using your webcam.

[GitHub Repository Link](https://github.com/Inzui/EASMVI01)

<h2>Install required packages</h2>

```
$ python -m pip install requirements.txt
```


<h2>Run code</h2>

Code can be run by using the following command:

```
$ python main.py
```

Arguments can be passed by adding them after this command. Example to force the retraining of the Machine Learning Model:

```
$ python main.py -ft
```

A list of possible arguments can be requested by using the -h option:


```
$ python main.py -h
usage: main.py [-h] [-fc] [-ft] [-si]

EASMVI01 Assignment for recognizing Dutch sign language.

options:
  -h, --help           show this help message and exit
  -fc, --forceConvert  Force the conversion of training and test images
                       to CSV.
  -ft, --forceTrain    Force the training of the Machine Learning Model,
                       even if one already exists.
  -si, --showImages    Shows the detected hand with drawn landmarks
                       while running.
```

<b>Note: unzip Dataset.zip before running the forceConvert command!</b>