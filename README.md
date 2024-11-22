# SAMWISE ADCS sims
*(ADCS = Attitude Determination and Control System)*

6DOF python simulation for validating SAMWISE ADCS

To install the contents of this repo as a python package run:
```
pip3 install .
```
*INSTALL MAGNETIC FIELD MODEL*
```
cd igrf
python -m pip install -e .
```

To run the code:
```
python3 simwise/simulation.py
```

> [!NOTE]
> See the rust code for the actual board here:
> https://github.com/polygnomial/adcs

Started 10/17/2024
Stanford Student Space Initiative
