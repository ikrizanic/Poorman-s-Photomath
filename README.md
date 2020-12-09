# Photomath task
## Flask app url
https://poormans-photomath.herokuapp.com/

## Manual

Implementation of this solution comes with a simple Flask app that allows users to upload 
and evaluate their images. You can try this app on https://poormans-photomath.herokuapp.com/

Procution vesion of the flask app is on branch "heroku", while pure backend parts (detector, model and solver) are on the main branch.

It is also possible to use and change this code with just cloning this repository to
your machine. All necessary functions are uploaded here, and you can find a link to 
the subset of dataset used in this implementation saved in pickle format.

## Task
The task was to implement simple math problem solver. 
The solution consists of three main parts:
1. Character detector
2. Building and training model, and then predicting output for given image
3. Solving a simple mathematical expression using primitive parser and solver

## Solution
Tasks are solved as follows:
#### First task
OpenCV library was used to recognize contours on the given image and then to
crop areas of contours that are big enough.
Cropped images were scaled to fit wanted dimensions of 45pix*45pix and are returned 
as output of the first task.

#### Second task
Firstly sequential CNN model from Keras library was compiled, and then it was trained
for only one epoch on a subset of Kaggle dataset of handwritten symbols counting 
around 150k images. 
This extremely simple model gives more than good enough performance for a given task.

#### Third task
A very simple parser and solver were implemented. Parser is based on reverse polish notation 
and solver is based on simplified Shunting-yard algorithm.

## Conclusion
The solution of this task is capable of solving only very simple and pretty expression
and is not suitable for real life usage. There were many problems encountered during 
implementation of this solution, and usable solution would require much more complex
approach.

## Links
- dataset: https://www.kaggle.com/xainano/handwrittenmathsymbols
- dataset subset: https://drive.google.com/file/d/1g-BnJr-QiftCtWW_s4itgrCyfE7hsieD/view?usp=sharing
- Shunting-yard algorithm: https://brilliant.org/wiki/shunting-yard-algorithm/
- Keras library: https://keras.io/
- OpenCV library: https://opencv.org/
- scikit-learn library: https://scikit-learn.org/stable/
