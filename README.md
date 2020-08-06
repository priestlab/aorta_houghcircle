# aorta_houghcircle
Generating aorta measurements from pulmonary artery view MRI using hough circles

## Set up (installing all python package dependencies)
To ensure all required packages are installed, run the following command:
```
pip install -r requirements.txt
```

Package List:
```
cycler==0.10.0
decorator==4.4.2
future==0.18.2
imagecodecs==2020.2.18
imageio==2.8.0
kiwisolver==1.2.0
matplotlib==3.2.1
networkx==2.4
numpy==1.18.4
opencv-contrib-python-headless==4.2.0.32
pandas==1.0.3
Pillow==7.1.2
pydicom==1.4.2
pyparsing==2.4.7
python-dateutil==2.8.1
pytz==2020.1
PyWavelets==1.1.1
scikit-image==0.17.2
scipy==1.4.1
six==1.14.0
tifffile==2020.5.11
torch==1.5.0
torchvision==0.6.0
```

## Running the code with an example
To see the process of segmentation and generating measurements in action, simply run the code:
```
python Ao_segment.py --root_dir test_data/zips --csv_data test_data/labels.csv --threads 1 --plot
```

The segmentation of the examples would be like the images shown below:

Example 1:
![Example 1](/images/segmentation_test01.png)

Example 2:
![Example 2](/images/segmentation_test03.png)

Example 3:
![Example 3](/images/segmentation_test03.png)
