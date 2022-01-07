conda create --name pyg python=3.6
conda activate pyg
conda install tqdm yaml
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install torch-geometric
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install matplotlib
pip install unionfind
pip install accelerate
pip install packaging
accelerate config
