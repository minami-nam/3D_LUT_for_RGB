 # **(Image - Adaptive) 3D-LUT-for-RGB-NIR**

 
## About this program
First, It's based on HuiZeng/Image-Adaptive-3DLUT, so if you want to make on your own, please visit his repository.

It's my first program using python. 

My Lab wanted to make program that has 2 inputs, RGB (PNG) file and NIR(High-light) file each and prints some results using 3D LUT method.

Caution : now I modified only about paired training. so unpaired one may have some problems to run directly.


## Usage

#### Requirements : 
```
Python 3.10.x, 
Cuda 12.1 or 12.6, 
Pytorch 2.7.1+cu121, 
Torchvision 0.16.0+cu121,
Numpy 1.26.4, 
Pillow 9.5.0, 
Scipy 1.15.3
ninja 1.11.1.4
(rawpy if you need to convert .dng files into 480p .jpg files using final.py.)
```

## How to Run

### making build files

#### before you run these .py files, please read this carefully.

you should make data/fiveK/ dir first, and some folders below to save your resized .jpg files.
and to train the model, put input files in data/fiveK/input, Highlighted NIR files in data/fiveK/NIR, Original files in data/fiveK/Target.


```
cd your/dir/Image-Adaptive-3DLUT-master
python final.py # if you need to make dataset files

python setup.py install

```

### Training and Evaluating the results

you should put test input files in data/fiveK/input, Higtlighted NIR files in data/fiveK/NIR (the program will not read this dir during eval), Target files in data/fiveK/Target.


```
python image_adaptive_lut_train_paired.py

python image_adaptive_lut_evaluation.py
```
