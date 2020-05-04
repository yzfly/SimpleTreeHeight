TREE HEIGHT EXTRACTION IN SPARSE SCENES BASED ON UAV REMOTE SENSING

## Requirements

* python3
* pip install numpy scipy matplotlib plotly
* [python-pcl](https://github.com/strawlab/python-pcl)

If you have python-pcl installation problems, you can try the ".whl" file in the floder "python-pcl/"

### Tested Environment

I have tested the code in the following environment：

* Windows10-64bit 
* Anaconda python3.7

Make sure anaconda has been installed on your windows computer, use the following command:

```bash
conda create -n py37 python=3.7
conda activate py37
pip install python_pcl-0.3.0rc1-cp37-cp37m-win_amd64.whl
conda install -c plotly plotly-orca==1.2.1 psutil requests
pip install numpy scipy matplotlib
```

#### ImportError: DLL load failed

If you meet "ImportError: DLL load failed" problem when you try to "import pcl", that was because "OpenNI2.dll" missing. 

> To fix this, I provide you with "python-pcl/OpenNI2.dll", just copy "OpenNI2.dll" into folder "YOUR_ANACONDA3_FLODER\envs\py37\lib\site-packages\pcl".

## Run

> python main.py

## Description

A simple algorithm for extracting tree height in sparse scene from point cloud data.

<div align=left>
<img width="230" height="315" src="https://raw.githubusercontent.com/yzfly/SimpleTreeHeight/master/images/procedure.png"/>
</div>