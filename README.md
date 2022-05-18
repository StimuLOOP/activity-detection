# activity-detection
We provide utilities to apply models presented in 

* Accuracy of physical activity and posture classification using movement sensors in individuals with mobility impairment after stroke
* Classification of Functional And Non-Functional Arm Use by Inertial Measurement Units In Subjects With Upper Limb Impairment After Stroke

## Prerequisites
To use the provided code, please install the requirements given in ```requirements.txt``` with pip.
```
pip install -r requirements.txt
```
To prevent dependency issues, we recommend installing the packages in a conda or venv environment.

## Data Format
To use our models, make sure to format the data accordingly in csv format. 
Sensor data should be collected from the following set of sensors

* left wrist (```wrist_l```)
* right wrist (```wrist_r```)
* left ankle (```ankle_l```)
* right ankle (```ankle_r```)
* chest (```chest```)

The csv may also contain only a subset of these sensor locations.
Additionally, each sensor measurement must include the following measurements:
* Barometer (```press```)
* Gyroscope in x,y and z direction (```gyro_<x,y or z>```)
* Accelerometer in x,y and z direction (```acc_<x,y or z>```)

The CSV must then have columns in the format ```<sensor_pos>___<measurement>```.
Examples of how to format the data are given in ```examples/data.csv``` and ```examples/data_wrists.csv``` with only a subset of sensor locations.
Example code to export data collected by the Zurich Move sensor in matlab to csv can be found in ```MatlabToCSV.ipynb```.


## Gait and Activity predictions
This repository provides scripts to apply the SVM model for various sensor setups as explained in "_Accuracy of physical activity and posture classification using movement sensors in individuals with mobility impairment after stroke_".

### Usage
To generate gait and activity predictions for a specific set of sensors and a given csv file, you can use the scripts ```get_gait_predictions.py``` and ```get_activity_predictions.py```. 
Generally, the script can be called as follows:
```
python get_<gait/activity>_predictions.py [-s {wrists,ankles,all,no_chest,left,right}] [-a] [-o OUTPUT_LOCATION] file_path
```
Call ```python get_<gait/activity>_predictions.py -h``` for a more detailed description of each argument.

Some examples of how to use them are:
```
python get_gait_predictions.py example/data.csv
python get_activity_predictions.py -s wrists example/data.csv 
python get_gait_predictions.py -s left -a example/data.csv 
python get_activity_predictions.py -s ankles -o example/ example/data.csv 
```



## Functional predictions
In addition to the previous section, we provide the Logistic Regression models from "_Classification of Functional And Non-Functional Arm Use by Inertial Measurement Units In Subjects With Upper Limb Impairment After Stroke_".

### Usage
To generate functional arm predictions for a specific set of sensors and a given csv file, you can use the script ```get_functional_predictions.py```. 
Generally, the script can be called as follows:
```
python get_<gait/activity>_predictions.py [-s {wrists,ankles,all,no_chest,left,right}] [-a] [-o OUTPUT_LOCATION] file_path
```
Call ```python get_<gait/activity>_predictions.py -h``` for a more detailed description of each argument.

Some examples of how to use them are:
```
python get_functional_predictions.py example/data.csv
python get_functional_predictions.py -s wrist_r example/data.csv 
python get_functional_predictions.py -s wrist_r -a example/data.csv 
python get_functional_predictions.py -s wrist_l -a -o example/ example/data.csv 
```