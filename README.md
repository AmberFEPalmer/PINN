# SEIR physics informed neural network
Python - Version 3.10.19
Tensorflow - Version 2.15.0

PINN files
1. PINN_simulated_data.py = PINN tested on simulated SIR models, beta learnt by the model, recovery rate and incubation parameters provided
2. PINN_all_learnable_parameters.py = PINN tested on simulated SIR models, beta, recovery rate and incubation parameters learnt by the model
3. PINN_SEIR.py = PINN on real world COVID-19 data
4. PINN_metapopulation = PINN on metapopulation SIR model with 5 patches

Data files
1. Data_processing.py = processing UKHSA COVID-19 case data
2. SEIR_model.py = developing SEIR model to generate synthetic data
3. PINN_transmission_changing.py = developing SEIR models with beta changing as a spline
4. SEIR_Gaussian_noise.py = gaussian noise added to infected compartment of SEIR model
