# Script for installing BBHx, nessai and their dependencies

# Needs conda environment for BBHx, which we will install first.
conda create -n solar_wind_env -y

eval "$(conda shell.bash hook)"
conda activate solar_wind_env

conda install -c conda-forge python lapack=3.6.1 gsl Cython -y  # not sure if the cython dep is needed

# now we switch to pip for everything else
pip install bbhx

# there is a bug in release nessai; for now, we will install it from the git repository.
pip install git+ssh://git@github.com/mj-will/nessai.git

pip install corner



# we should be done!