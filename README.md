This repository contains the source code for the simulations performed in Part I and II of the "Realising Synthetic Active Inference Agents" paper series. Simulations are performed with Jupyter Notebook, Julia and RxInfer. Original figures can be regenerated from the `v0.1.0` release of this repository.

# Install Julia
In order to install the Julia language (`v1.10.4`), follow the platform-specific instructions at https://julialang.org/downloads/

# Install Jupyter Notebook
Jupyter notebook is a framework for running Julia scripts (among other languages). It is well-suited for showing demo applications and interactive experimentation. In order to install Jupyter Notebook, follow the instructions at https://jupyter.readthedocs.io/en/latest/install.html

# Install required packages
The simulation notebooks require several external packages. To install them, open Julia
```
$ julia
```
and enter the package prompt by typing a closing bracket
```
julia> ]
```
Next, activate the virtual environment
```
(v1.10) pkg> activate .
```
and instantiate the required packages
```
(LAIF) pkg> instantiate
```
This will download and install the required packages in the virtual environment named LAIF.

# Run the demos
Exit Julia, navigate to the root directory and start a Jupyter server
```
~/LAIF$ jupyter notebook
```
A browser window should open, and you can select the demo you wish to run.

# License
MIT License, Copyright (c) 2024 BIASlab http://biaslab.org
