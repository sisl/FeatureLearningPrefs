# FeatureLearningPrefs
Companion code for "Preference-based Learning of Reward Function Features" by S.M. Katz, A. Maleki, E. Biyik, and M. J. Kochenderfer

This code was tested using Julia1.1 and Flux version 0.8.3 (note: it may not run on later versions of Flux)

## Data Collection
To collect data training data for a particular user, first open two terminal windows and navigate to the `python_model` directory in both. In one terminal window, run `python traj_A.py` and leave it running. In the other terminal window, run `python traj_B.py` and leave it running. These scripts will play videos of trajectories when they are created during the query process.
Next, run the following lines in Julia:
```
include("driving_simulate.jl")
num_queries = 100
reward_iteration(num_queries)
```

To collect testing data, ensure that the python scripts are still running and run the following lines in Julia:
```
include("driving_simulate.jl")
num_test_examples = 75
u_test = generate_heuristic_test_set(num_test_examples)
respond_to_test_set(u_test)
```

## Network Training
As an example, run these lines in Julia to train the network for user 1:
```
include("driving_simulate.jl")
include("driving_postprocessing.jl")

@load "TrainData/train1.jld2"
prefs_train_1 = copy(prefs)
final_weights_1 = copy(w_hist[end])

@load "TestData/test1.jld2"
prefs_test_1 = copy(prefs)

prefs_train_1, X_train_1, Y_train_1, X_test_1, Y_test_1, W_1 = 
					get_datasets(prefs_train_1, prefs_test_1, final_weights_1)

Random.seed!(24)
nn_ϕ = Chain(Dense(ns, nhidden, relu), Dense(nhidden, num_features_nn, tanh))
acc_train_vec_1, acc_eval_vec_1, acc_test_vec_1 = run_mixed_fancy_save(prefs_train_1, X_test_1, Y_test_1, W_1, "nn1.jld2")
```

## Network Visualization
To visualize the network for user 1, run the following lines in a Jupyter notebook:
```
include("driving_simulate.jl")
include("driving_viz.jl")

@load "FinalNetworks/nn1.jld2"

viz_features(nn)
```
An interactive window should pop up. Brighter regions indicate higher feature values.