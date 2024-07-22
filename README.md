# SolarWindLISA
Investigating the impact of the solar wind on LISA data analysis

## Installation
Ensure that you have `conda` installed. Then run the script
```console
./install.sh
```
which will create a `conda` environment and place all necessary packages inside. If you cannot execute the script (e.g. `Permission denied`) then make it executable via 
```console
sudo chmod +x install.sh
```
and run it.

Once the script has finished running, simply activate the environment before running python scripts:
```console
conda activate solar_wind_env
```

## Usage
See the `examples/noiseless.py` script for an example parameter estimation. Copy and modify as required to include your own data.

## Problems?
Email me (c.chapman-bird.1@research.gla.ac.uk) or use the Github issue tracker if there are bugs.
