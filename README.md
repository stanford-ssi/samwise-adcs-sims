# SAMWISE ADCS sims
*(ADCS = Attitude Determination and Control System)*

6DOF python simulation for validating SAMWISE ADCS


## Installation
To install the contents of this repo as an editable python package run:
```
pip3 install -e . 
```
(making sure you are in the root folder)

## Directory Structure
* `attitude`: Attitude GNC algorithms
* `orbit`: Orbit GNC algorithms
* `world_model`: Code for modelling things external to the satellite (such as fields, gravity, etc)
* `data_structures`: Classes for holding and modeling data
* `math`: Purely math utils (e.g. quaternion functions)
* `utils`: General utilities (plotting, etc)
* `simulations`: Runnable code goes here - **each file must define a `run` method!**

## Running the Code
To run the code, type:
```
python -m simwise.main --run [name of file in simulations/operational folder]
```
i.e. to run the integrated sim, run the following command in the root directory:
```
python -m simwise.main --run nadir_pointing
```
*INSTALL MAGNETIC FIELD MODEL*
```
cd igrf
python -m pip install -e .
```

e.g. `python3 simwise/main.py --run simulate_attitude`


## References
> [!NOTE]
> See the rust code for the actual board here:
> https://github.com/polygnomial/adcs

Started 10/17/2024
Stanford Student Space Initiative
