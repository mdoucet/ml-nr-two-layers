# Machine Learning for Reflectometry
M. Doucet, R. Archibald, W. Heller

Machine learning code to determine a thin film structure from a 
two-layer thin film measured with neutron reflectometry.

From an R(Q) profile with a predefined Q binning, we predict the values
of the two layer thicknesses, their SLD, their roughness parameters, and
the substrate's roughness.

## What's in the repo?
The `data` directory includes the neural network model, and validation
data for testing. It also includes example data taken on the Magnetism
Reflectometer at SNS.

The `notebooks` directory contains example notebooks that loads and
uses the trained neural network.

## Dependencies
TensorFlow was used to handle the neural network. A conda environment
can easily be created using the `play_env.yaml` file using

    conda env create -f play_env.yml
