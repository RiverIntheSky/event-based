# DVS SLAM
The master branch performs SLAM on planar scenes with an event-based camera using contrast maximization.
## Usage 
```
./planar_slam dataset_name window_size experiment_name
```
`dataset_name` is the dataset you want to test the algorithm, `window_size` is the number of events to be grouped into a frame, `experiment_name` creates a folder, where the estimation and ground truth pose is stored to .txt files. If there already exists a folder with the same name, the original folder will be **deleted**. You have to set `dataset_name`, the other two parameters can be left empty. `window_size` will default to 10 000, no folder will be created to store the optimized results.

example: 
```
./planar_slam ~/Documents/dataset/poster_translation 30000 test
```
## Note
You can choose whether or not to use numerical differentiation in the tracking phase by setting the variable `num_diff` in source file `num_diff` to `true` or `false`. The size of the map can also be changed in the same file. 

The author considers to wrap the parameters into a config file.

This project is part of the master thesis *Use of a DVS for High Speed Applications* by Weizhen Huang weizhen.huang@rwth-aachen.de
