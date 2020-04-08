TREE HEIGHT EXTRACTION IN SPARSE SCENES BASED ON UAV REMOTE SENSING

## Requirements

* python3
* pip install numpy scipy matplotlib plotly
* [python-pcl](https://github.com/strawlab/python-pcl)

If you have python-pcl install problems, you can try the ".whl" file in the floder "python-pcl/"

### Tested Environment

I tested the code in the following environment：

* Windows10-64bit 
* Anaconda python3.7

Make sure anaconda has been installed on your windows computer, use the following command:

> conda create -n py37 python=3.7
> pip install python_pcl-0.3.0rc1-cp37-cp37m-win_amd64.whl
> conda install -c plotly plotly-orca==1.2.1 psutil requests
> pip install numpy scipy matplotlib


## Run

> python main.py

## Description

A simple algorithm for extracting tree height in sparse scene from point cloud data.

<div align=left>
<img width="230" height="315" src="https://raw.githubusercontent.com/yzfly/SimpleTreeHeight/master/images/procedure.png"/>
</div>