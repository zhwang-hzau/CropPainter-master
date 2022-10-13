# **CropPainter**

## - Dependencies

CUDA
Cudnn
Python3.6
 - Anaconda create environment：
 - `conda create -n cp36 python=3.6`
 - `conda activate cp36`
 - `conda install pytorch=0.4.1 cuda92 -c pytorch`
 - `cd CropPainter-master`
 - `pip install -r requirements.txt`

**Mention:** The above code installs pytorch0.4.1 with the cuda9.2 and  cudnn7.1, they work successfully on 2070s, but may have some problems on 30 series Graphics Cards. So please make sure your CUDA, Cudnn, pytorch and torchvision versions match each other. This environment was tested successfully on Windows10 and Ubuntu16. 

## Testing

Before testing, download the [datasets_1](https://drive.google.com/file/d/1M_vB8OOB8K9jzF9-_A6P7oJ6dL7Jm-LP/view?usp=sharing) and [trained model](https://drive.google.com/file/d/1Y9dQJJNGOVWzu5Hkq65J4bZNruizr3Ak/view?usp=sharing), unpacked and placed in CropPainter-master. 

**Single image generate**
in `CropPainter_run.py` : 
set`cls` to different crop types you want test
set`run = "test"`	`single_generate=True`
set`single_traits=traits, single_image_name='test.png'` the value of the traits and the name of the generated image can be modified. However, if the input traits exceed the range of traits used for training, the performance of the generated images will deteriorate. the range are shown at range.csv 

	cd code
	python CropPainter_run.py

**Batch image generate**
 in `CropPainter_run.py` : 
set`run = "test"`	`single_generate=False`
set`traits_path='../data/Panicle/test/traits.csv'` testing set traits.csv 

In the project, we provide four crops of testing set .csv files in `/data/<cls>/test'`, and four trained models in `models/<cls>/Model`

	cd code
	python CropPainter_run.py

## Training

Training parameters set：
 in `CropPainter_run.py` :
set`run = "train"`
set`traits_path='../data/Panicle/train/traits.csv'` training set traits.csv 

the training set images need to be placed in the `/data/<cls>/images'`
and a `filenames.txt` file is needed in `/data/<cls>/train'`, which contains the names of the training set images. 

In [datasets_1](https://drive.google.com/file/d/1M_vB8OOB8K9jzF9-_A6P7oJ6dL7Jm-LP/view?usp=sharing), we provide **Panicle** training set and related files, so it is directly trainable. If you need to train other crop data, download the [datasets_all](https://drive.google.com/file/d/1Fi9MsPHdyMDYbOmnY21llpd5JdT7aZ1f/view?usp=sharing) and move the data to the appropriate place.

	cd code
	python CropPainter_run.py

## CropPainter visualizer
Based on Windows10, we provide a software: CropPainter visualizer
![enter image description here](https://raw.githubusercontent.com/zhwang-hzau/images/main/cp-visualizer.png)

Without deep learning environment, users are able to operate the software to visualize phenotypic information for four crops. 
By entering trait values, crop images can be generated in real time. The software also supports inputting batch trait information (excel file) to generate crop images in batch.
The executable application can be downloaded from [here](https://drive.google.com/file/d/1k1iybVKWfwLZBfQ49sktsv-FbLJWwiDz/view?usp=sharing). 

Please note: Older CPU may cause stuck when running this application. We ran it successfully on intel Core i5-6500 (4 cores and 4 threads at 3.2GHz) and AMD Ryzen 5 3500X (6 cores and 6 threads at 3.6GHz), so we recommend using a higher performance CPU.
