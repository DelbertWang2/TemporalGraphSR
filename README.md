# TemporalGraphSR
This is the code of the algorithm proposed in “Temporal Graph Super Resolution on Power Distribution Network Measurements”. If this code is useful to you, we will be glad to see our paper being cited, thanks!


# 1	Requirements：
1.	Matlab (tested with Matlab 2018b on Windows 10)
2.	Matpower toolbox of Matlab (provided in the attachment)
3.	Python (tested with python 3.6, tested on Linux and Windows 10)
4.	Pytorch =1.1 (GPU version)
5.	Numpy = 1.16.3
6.	Scipy = 1.1.0
Others: math, os

# 2	How to run this code:
### Theoretically: 

0.	Initialization: Install the requirements and Run `/MatScripts/init.m` to initial the Matlab path
1.	Run `/MatScripts/gen_data/prepare_data.m` in order to
a)	generate the raw simulation datasets `/Data/rawdata.mat`
b)	calculate the modified adjacent matrix and incidence matrix
2.	Run `/PythonScripts/train_sepe.py` in order to
*This is the code for base case chapter 4.1*
a)	construct the mask M, the LTR features and the HTR labels with `rawdata.mat`, so, before run this script, make sure the `/Data/rawdata.mat` are correctly generated
b)	separate training set and test set, construct the training batches.
c)	implement the graph convolution layer and the 6-layer GCN model
d)	write the LTR features, HTR labels, GCN results and other details into file `/Data/trained_data.mat`
3.	Run `/PythonScripts/train_omittednodes.py` in order to
   *This is the code for chapter 4.3*
   a)	construct the mask M, the LTR features and the HTR labels with `rawdata.mat`, so, before run this script, make sure the `/Data/rawdata.mat` are correctly generated
   b)	separate training set and test set, construct the training batches.
   c)	implement the graph convolution layer and the 6-layer GCN model
   d)	write the LTR features, HTR labels, GCN results and other details into file `/Data/trained_data_omittednodes.mat`
4.	Run `/MatScripts/evaluation/run_SR.m` in order to
   a)	evaluate the proposed Super resolution method	
5.	Run `/MatScripts/evaluation/run_other_methods.m` in order to
   a)	evaluate the proposed other interpolation methods

### Actually:

For the convenience of the reviewers, most intermediate data files (.mat files) are also provided. Therefore, the reviewers can actually directly run step 4 or 5 after the initialization. If the .mat files are missing or changed, you will have to follow the steps above.


# 3	Why not all in python:
The models are implemented on both Matlab and python because the popular and stable power system analysis toolbox MATPOWER do not have python version. The PandaPower (a python power system analysis package) still has some bugs when performing state estimation. Therefore, we implement the power flow simulation and state estimation on Matlab. 
