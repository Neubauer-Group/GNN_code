conda create --name pyg python=3.6
conda activate pyg
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
conda install pandas matplotlib jupyter nbconvert
conda install tqdm yaml
conda install pyg -c pyg -c conda-forge
pip install unionfind
