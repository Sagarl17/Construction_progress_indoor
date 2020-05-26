import os
import sys
import json
import laspy
import runpy
from src import extract,classify

face=[[5.095,-16.676,2.179],[7.065,16.674,2.163,],[7.078,-16.691,1.136],[5.328,-16.710,1.047]]

mode=sys.argv[1]
if mode=='total':

    extract.extract_image(face)
    my_json=classify.image_classification()

    with open('progress.json', 'w') as outfile:                                                                                 #Save the json file
        json.dump(my_json, outfile)  
elif mode=='images':
    my_json=classify.image_classification2()
    with open('sep_progress.json', 'w') as outfile:                                                                                 #Save the json file
        json.dump(my_json, outfile)
