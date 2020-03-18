using Flux

"""
----------------------------------------------
Design Parameters
----------------------------------------------
"""
mutable struct PARAMETERS
    pre_process_nn::Bool
    visualize::String
    allow_neg_w::Bool
    objective::String
end

# pre_process_nn
#     true:       states are pre-processed before being sent to nn
#                 that means we pass the difference in x,y,v,θ
#	  false:      states are not pre-processed
# visualize
#     "python-1": using the python visualization with one
#                 trajectory shown at the time
#     "python-2": using the python visualization with two
#                 simulataneous trajectories shown
#     "julia:"    using julia visualization
# allow_neg_w:
#     true:       allows sampling negative weights. this generally
#                 generates unrealistic trajectories
#     false:      does not allow negative weights, leading to more
#                 realistic trajectories
# ojbective:
#     "random":   no active learning. generate random queries.
#     "ig":       information gain
#     "vr":       volume removal
#     "heuristic": heuristic approach

P = PARAMETERS(true, "python-2", false, "heuristic")

"""
----------------------------------------------
Types
----------------------------------------------
"""
mutable struct Preference
	x₁::Array
	x₂::Array
	ψ::Vector
	pref::Int64
end

function preference(x₁ = zeros(2,2), x₂ = zeros(2,2), ψ=zeros(2), pref=1)
	return Preference(x₁, x₂, ψ, pref)
end

"""
----------------------------------------------
General Size Parameters
----------------------------------------------
"""
M = 100 #1000 # Number of weights to sample to estimate the objective function
num_features_hc = 4   # hand designed features
num_features_nn = 1	  # neaural network features
num_features_tot = num_features_nn + num_features_hc # total number of  features
num_steps = 5 # Number of times control input changes
step_time = 10 # Number of dt's per step
ctrl_size = 2 # Number of control inputs per time step

"""
----------------------------------------------
Variables Updated During Iterations
----------------------------------------------
"""
W_tot = zeros(num_features_hc, M)   # weights of handed coded feature vector
W_tot_mixed = zeros(num_features_tot, M)   # weights of total feature vector
prefs = Vector{Preference}()
w_hist = Vector{Array{Float64,2}}()

"""
----------------------------------------------
MCMC Parameters
----------------------------------------------
"""
burn = 5000 # length of burn-in period
T = 100 # thinning of samples (take every Tth sample after burn)

"""
----------------------------------------------
Optimization/Dynamics Parameters
----------------------------------------------
"""
lb = repeat([-1.0, -1.0], num_steps*2)
ub = repeat([1.0, 1.0], num_steps*2)

dt = 0.1
friction = 1

"""
----------------------------------------------
Neural Net Parameters
----------------------------------------------
"""
ns = 4
nhidden = 100
k = num_steps*step_time
nn_ϕ = Chain(Dense(2*ns, nhidden, relu), Dense(nhidden, num_features_nn, tanh))

