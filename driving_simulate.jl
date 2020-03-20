# Main file to include for FeatureLearningPrefs repository
# Includes all relevant files and packages

using Mamba
using Optim
using LinearAlgebra
using ForwardDiff
using LineSearches
using Distributions
using JLD2
using NearestNeighbors
using DelimitedFiles
using Flux
using BSON
using Plots
using PyCall
using Random

include("driving_const.jl")
include("driving_functions.jl")
include("driving_learn.jl")
include("driving_postprocessing.jl")
if P.visualize == "python-1"
	include("python_model/load_python_lib.jl")
end

# Function used to interact with user and collect their
# responses to preference queries using active learning
function reward_iteration(n_queries::Int64)
	u_tot = Array{Float64}(undef, 0)
	for i = 1:n_queries
		println("Query $i:")
		sample_w(M)

		# Just is case optimization fails and get a nan
		# reinitialize and try again
		itr_nan = 0
		while true
			u_tot = get_inputs()
			if isnan(u_tot[1])
				print("nan value found. trying again!")
				itr_nan += 1
				if itr_nan == 5
					throw(DomainError("nan values persist!"))
				end
			else
				break
			end
		end

		x₁, x₂, ψ_tot = get_ψ_hc(u_tot)

		# Obtain user feedback
		if P.visualize == "python-1"
			#
			ϕ₁, ϕ₂, pref = py"sim_trajectory"(u_tot[1:num_steps*ctrl_size],
											  u_tot[num_steps*ctrl_size+1:end])
		elseif P.visualize == "python-2"
			# dump trajectory files. python programs will visualize these.
			println("plotting!!!")
			user_input = "0"
			while !(user_input == "1") & !(user_input== "2")
				write_trajectory("python_model/traj_A.txt", u_tot[1:num_steps*ctrl_size])
				write_trajectory("python_model/traj_B.txt", u_tot[num_steps*ctrl_size+1:end])
				user_input = Input("enter 1 for A and 2 for B and 0 for repeat?\n")
			end
			pref = user_input == "1" ? 1 : -1
		end

		# Add to current set of preferences
		push!(prefs, Preference(x₁, x₂, ψ_tot, pref))
	end
	@save "my_data.jld2" w_hist prefs
end

# Function to response to a set of test queries and
# store the reponses. u_test can be created using the
# generate_heuristic_test_set function in the
# driving_functions.jl file
function respond_to_test_set(u_test)
	num_examples = size(u_test, 1)
	for i = 1:num_examples
		println("Test Set $(i):")
		u_tot = u_test[i, :]

		x₁, x₂, ψ_tot = get_ψ_hc(u_tot)

		# Obtain user feedback
		if P.visualize == "python-1"
			ϕ₁, ϕ₂, pref = py"sim_trajectory"(u_tot[1:num_steps*ctrl_size],
											  u_tot[num_steps*ctrl_size+1:end])
		elseif P.visualize == "python-2"
			# dump trajectory files. python programs will visualize these.
			println("plotting!!!")
			user_input = "0"
			while !(user_input == "1") & !(user_input== "2")
				write_trajectory("python_model/traj_A.txt", u_tot[1:num_steps*ctrl_size])
				write_trajectory("python_model/traj_B.txt", u_tot[num_steps*ctrl_size+1:end])
				user_input = Input("enter 1 for A and 2 for B and 0 for repeat?\n")
			end
			pref = user_input == "1" ? 1 : -1
		end

		# Add to current set of preferences
		push!(prefs, Preference(x₁, x₂, ψ_tot, pref))
	end
	@save "test_data.jld2" prefs
end

function write_trajectory(file_name, uvec)
	f = open(file_name, "w")
	writedlm(f, uvec, ", ")
	close(f)
end
