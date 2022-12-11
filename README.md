# KCF tracker in Python3

The project is based on [the project](https://github.com/uoip/KCFpy.git), a Python2 version implementation of 
> [High-Speed Tracking with Kernelized Correlation Filters](http://www.robots.ox.ac.uk/~joao/publications/henriques_tpami2015.pdf)<br>
> J. F. Henriques, R. Caseiro, P. Martins, J. Batista<br>
> TPAMI 2015

The version of Python2 is translated from [KCFcpp](https://github.com/joaofaro/KCFcpp) (Authors: Joao Faro, Christian Bailer, Joao F. Henriques), a C++ implementation of Kernelized Correlation Filters. Find more references and code of KCF at http://www.robots.ox.ac.uk/~joao/circulant/

### Requirements
* Python3 (3.8 or later)
* NumPy
* Numba (needed if you want to use the hog feature)
* OpenCV 

Here is an easy way to install, just execute the command under the project folder.

```shell
pip3 install -r requirements.txt
```

### Use
Download the sources and execute
```shell
git clone git@github.com:discipleofhamilton/KCF-python.git # ssh
git clone https://github.com/discipleofhamilton/KCF-python.git # https
cd KCFpy
python run.py
```
It will open the default camera of your computer, you can also open a different camera or a video
```shell
python run.py 2
```
```shell
python run.py ./test.avi  
```

### Limitation

KCF has 2 limitations:

1. Programming language problem: the execute speed of python is slower than C++.
2. Short-term tracking: KCF doesn't have fail-over strategy, it hard to re-track the object which it just lost.