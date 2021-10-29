qdPy is a package to compute helioseismic eigenfrequencies using
quasi-degenerate perturbation theory. 

# Forward model
Taken an input toroidal velocity profile in terms of $w_s(r)$
and computes the new eigenfrequencies using both 
degenerate perturbation theory (DPT) and quasi-degenrate
perturbation theory (QDPT)

# Inversions
-- yet to be implemented 

## How to use the package

Open the terminal and go to your local directory where you want to clone the repository and use ```git clone``` as follows
```
git clone https://github.com/samarth-kashypa/qdPy.git
```
Enter the directory ```qdPy```
```
cd qdPy
```
We recommend creating the ```env_qdPy``` environment from the ```env_qdPy.yml``` (incase all the dependencies are not already in your current conda environment), 
```
conda env create -f env_qdPy.yml
conda activate env_qdPy 
```
Run the configure script to download eigenfunctions and setup the directory structure.
You'll first be prompted to enter the location of the ```scratch``` directory (This is where all
output files will be stored. The default would be the package directory itself. The script will
then run the solar eigenfunction downloader. By default these will be installed in the 
directory that contains the package i.e. if package directory is given by ```/path/to/dir/qdPy```,
the eigenfunctions will be installed in ```/path/to/dir/get-solar-eigs``` by default.

