# GNN_code

First thing you need to create the environment:

`source install_pyg.sh`

Then you need to download the trackml data from https://www.kaggle.com/c/trackml-particle-identification/data

unzip all the tarballs and place them all in one folder. Then we need to create the folder structure that pytorch-geometric expects. You will need to edit this file to the paths on your system

`source make_folder.sh test_run`

This will create a folder named test_run in your training folder, with subdirectories processed and raw. The raw folder should be sim linked to your trackml data folder (we dont want multiple copies of the raw data). The processed folder is where the graphs will be once built. This will also copy the config.yml file into the folder. This config file is where you make changes to how you want the graphs built.

The folder above test_run is the training folder. Edit env.sh to point to the training folder.

You are now ready to build and train a model.

`source run.sh`

This file calls scripts/heptrx_nnconv.py.  If you decided to name the folder under the training folder something other than test_run, you will need to edit run.sh to point to the correct folder, you can also call a different gnn model (see the comment out lines and mimic them). You will also need to change the path of the output folder inside heptrx_nnconv.py.


Once the graphs are built and the model is trained, you can calculate the tracking efficiencies by running:

`source calculate_efficiencies.py`

Again you will need to change the names of some paths so that it properly loads the model. This file calls scripts/track_efficiency.py
