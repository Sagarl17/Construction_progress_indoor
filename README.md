# Construction_progress_indoor


# Construction progress-outdoor module

This repository contains the code for predicting the outdoor progress of each component in BIM

# Steps For Installation:

```bash
git clone https://git.xyzinnotech.com/gopinath/construction_progress_outdoor.git
cd construction_progress_outdoor
pip install -r requirements-cpu.txt (If GPU is not available)
               or
pip install -r requirements-gpu.txt (If GPU is available)

```


# Initial setup:
```bash
1.Download the stanford dataset from the following folder ,unzip it and place it the main folder:
    https://drive.google.com/open?id=1nXq75Aru5HouRyEmAWzHceMe4jRo3ptv

2.Download the bim folder from the following link and replace the bim folder in data:
    https://drive.google.com/open?id=1AwWjPi1SVuyxSOJALPJY8Wm6cMEkiwB6


```

# How to test for the new dataset:

```bash
There are two modes for main.py.One mode is just classification of any image. Another mode is extracting element from 3d mesh as image and classifying it
1.For classification of wall images, place images of wall in sep_images and run python main.py images

2.For classification of element from 3d mesh, open the "rgb.obj" in 3d folder with cloud compare and select four points.Enter the points manually into main.py in face array and run python main.py total
```
# Where are my results stored :

```bash
1)The classfied images will be in sepc_images and stages of wall will be generated in "sep_progress.json"

2)The extracted image from mesh will be in extracted_images, classified image will be in classified_images and stages of wall will be generated in "progress.json"
```



