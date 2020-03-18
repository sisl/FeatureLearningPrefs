
"""
----------------------------------------------
MCMC Sampling
----------------------------------------------
"""

# Performs MCMC sampling using adaptive metropolis algorithm
# implemented in Mamba.jl package
function sample_w(M)
	w₀ = zeros(num_features_hc)
	Σ = Matrix{Float64}(I, num_features_hc, num_features_hc) ./ 10000 # From batch active code
	w = AMMVariate(w₀, Σ, logf)
	for i = 1:(M*T+burn)
		sample!(w, adapt = (i <= burn))
		if i > burn && mod(i,T) == 0
			ind = convert(Int64, (i-burn)/T)
			W_tot[:,ind] = w[:]
		end
	end
	push!(w_hist, copy(W_tot))
end

# Log-likelihood for MCMC
function logf(w::DenseVector)
	# Priors
	if norm(w) > 1.0
	 	return -Inf
	elseif any(w .< 0.0) & !P.allow_neg_w
		return -Inf
	# No prefs yet so uniform prior
	elseif isempty(prefs)
		return 0.0
	# Log of sigmoid likelihood
	else
		sum_over_prefs = 0
		for i = 1:length(prefs)
			sum_over_prefs += min.(prefs[i].pref*w'*prefs[i].ψ, 0)
		end
		return sum_over_prefs
	end
end


"""
----------------------------------------------
Update states
----------------------------------------------
"""

# Runs dynamics to get a trajectory from a set of control inputs
function get_x_mat(u; init_x = [0.17, 0, pi/2, 0.41])
	x = []
	ind = 0
	for i = 1:num_steps
		for j = 1:step_time
			ind += 1
			if ind == 1
				x = init_x + dxdt(init_x, u[ctrl_size*(i-1)+1:ctrl_size*i])*dt
			else
				x = hcat(x, x[:,end] + dxdt(x[:,end], u[ctrl_size*(i-1)+1:ctrl_size*i])*dt)
			end
		end
	end

	return x
end

# Actual vehicle dynamics
function dxdt(x, u)
	return [x[4]*cos(x[3]), x[4]*sin(x[3]), x[4]*u[1], u[2] - x[4]*friction]
end

"""
----------------------------------------------
Active query generation
----------------------------------------------
"""

# Get handed-coded feature difference by running trajectories from the inputs
# and obtaining the hand-coded features for each of them
function get_ψ_hc(u_tot)
	x₁ = get_x_mat(u_tot[1:num_steps*ctrl_size], init_x = [0, -0.3, π/2, 0.4])
	x₂ = get_x_mat(u_tot[num_steps*ctrl_size+1:end], init_x = [0, -0.3, π/2, 0.4])
	ϕ₁ = ϕ_hc(x₁)
	ϕ₂ = ϕ_hc(x₂)
	ψ_hc = ϕ₁ - ϕ₂
	return x₁, x₂, ψ_hc
end

# Get mixed feature difference by running trajectories from the inputs
# and obtaining the hand-coded features and mixed feature for each of them
function get_ψ_mixed(u_tot, eval_nn)
	x₁ = get_x_mat(u_tot[1:num_steps*ctrl_size], init_x = [0, -0.3, π/2, 0.4])
	x₂ = get_x_mat(u_tot[num_steps*ctrl_size+1:end], init_x = [0, -0.3, π/2, 0.4])
	ϕ₁ = ϕ_hc(x₁)
	ϕ₂ = ϕ_hc(x₂)
	ϕ₃ = ϕ_nn(x₁, eval_nn).data
	ϕ₄ = ϕ_nn(x₂, eval_nn).data
	ψ_hc = ϕ₁ - ϕ₂
	ψ_nn = ϕ₃ - ϕ₄
	ψ = vcat(ψ_hc, ψ_nn)
	return x₁, x₂, ψ
end

# Obtain the optimal inputs for the next query based on
# the current W samples and the specified active querying method
# For volume removal and information gain objectives, the optimizer
# L-BFGS implemented in the Optim package is used
function get_inputs()
	initvals = rand.(Distributions.Uniform.(lb,ub))
	if P.objective == "random"
		return initvals
	end
	if P.objective == "heuristic"
		return get_input_heuristic()
	end

	if P.objective == "ig"
		objec = object_ig_hc
	elseif P.objective == "vr"
		objec = object_vr_hc
	else
		throw(UndefVarError("$(P.objective) is not defined"))
	end

	inner_optimizer = LBFGS(linesearch = LineSearches.BackTracking())
	res = Optim.optimize(objec, lb, ub, initvals, Fminbox(inner_optimizer),
						Optim.Options(show_trace=false), autodiff=:forward)
	u = Optim.minimizer(res)
	return u
end

"""
----------------------------------------------
Heuristic method functions
----------------------------------------------
"""

# Heuristic query generation method that relies on
# multiobjective optimization method to select
# samples used to create query trajectories
function get_input_heuristic(;μ=0.1)
	best_val = 0
	best_candidate = hcat(W_tot[:,1], W_tot[:,2])
	for i = 1:M
		for j = i+1:M
			pa = siglike(W_tot[:,i])
			pb = siglike(W_tot[:,j])
			# normalizing weights when calculating distance
			distance = norm(W_tot[:,i]/norm(W_tot[:,i]) - W_tot[:,j]/norm(W_tot[:,j]))

			if pa*pb/(length(prefs)^2) + μ*distance > best_val
				best_val = pa*pb/(length(prefs)^2) + μ*distance
				best_candidate = hcat(W_tot[:,i], W_tot[:,j])
			end
		end
	end
	ua = get_optimal_inputs(best_candidate[:,1])
	ub = get_optimal_inputs(best_candidate[:,2])
	return [ua; ub]
end

# Takes in a sample of linear reward weights and returns
# the locally optimal trajectory for that particular set
# of weights; optimization is run num_runs times and the
# best control inputs are returned
function get_optimal_inputs(w, num_runs=10)
	lb_small = lb[1:num_steps*ctrl_size]
	ub_small = ub[1:num_steps*ctrl_size]
	initvals = rand.(Distributions.Uniform.(lb_small, ub_small))
	inner_optimizer = LBFGS(linesearch = LineSearches.BackTracking())
	objec(u) = -w'*get_ϕ(u)[2]

	res = Optim.optimize(objec, lb_small, ub_small, initvals,
		Fminbox(inner_optimizer), Optim.Options(show_trace=false), autodiff=:forward)
	u = Optim.minimizer(res)
	curr_min = Optim.minimum(res)

	for i = 1:num_runs-1
		res = Optim.optimize(objec, lb_small, ub_small, initvals,
			Fminbox(inner_optimizer), Optim.Options(show_trace=false), autodiff=:forward)
		if Optim.minimum(res) < curr_min
			curr_min = Optim.minimum(res)
			u = Optim.minimizer(res)
		end
	end
	return u
end

# Way to calculate likelihood for multiobjective optimization
# querying method
function siglike(w)
	sum_over_prefs = 0
	for i = 1:length(prefs)
		sum_over_prefs += min.(prefs[i].pref*w'*prefs[i].ψ, 0)
	end
	return sum_over_prefs
end

# Gets feature of one trajectory from a vector
# of control inputs
function get_ϕ(u::Vector)
	x = get_x_mat(u, init_x = [0, -0.3, π/2, 0.4])
	return x, ϕ_hc(x)
end

"""
----------------------------------------------
Feature Functions
----------------------------------------------
"""

# Hand-coded features
function ϕ_hc(x::Array)
	xpos = x[1,:]
	ypos = x[2,:]
	θ = x[3,:]
	v = x[4,:]

	staying_in_lane = mean(exp.(-30min.((xpos .- 0.17).^2, xpos.^2, (xpos .+ 0.17).^2)))
	keeping_speed = -mean((v .- 1).^2)
	heading = mean(sin.(θ))
	collision_avoidance = -mean(exp.(-(7*(xpos .- xpos_h).^2 + 3*(ypos .- ypos_h).^2)))

	return [staying_in_lane, keeping_speed, heading, collision_avoidance]
end

# Neural network features
function ϕ_nn(x::Array, eval_nn)

	# Flux and ForwardDiff are not working together.
	# When eval_nn is false, instead of directly evaluating nn,
	# we use a function for it. This is necessary for autodiff to
	# work, because ForwardDiff.jl is apparently not able to
	# differentiate a Flux.jl function.

	# update the function with the new weights
	if !eval_nn
		nn_ϕ_func = nn_to_func(nn_ϕ)
	end

	ϕ = zeros(num_features_nn)
	for i = 1:num_steps
		xvec = vcat(x[:,i], x_mat_h[:,i])
		if P.pre_process_nn
			xvec = pre_process_nn(xvec)
		end
		if eval_nn
			ϕ += nn_ϕ(xvec)
		else
			ϕ += nn_ϕ_func(xvec)
		end
	end
	return ϕ
end

# Gets hand-coded features from a flattened x-vector
# that contains states for both trajectories in a query
# Used in neural network feature learning functions
function get_features_hc(x)
	xr = reshape(x[1:k*ns], (ns,k))
	xr = hcat(xr, reshape(x[k*ns+1:end], (ns,k)))
	ϕ₁ = ϕ_hc(xr[:,1:k])
	ϕ₂ = ϕ_hc(xr[:,k+1:end])
	return ϕ₁, ϕ₂
end

# Gets neural network features from a flattened x-vector
# that contains states for both trajectories in a query
# Used in neural network feature learning functions
function get_features_nn(x)
	xr = reshape(x[1:k*ns], (ns,k))
	xr = hcat(xr, reshape(x[k*ns+1:end], (ns,k)))
	ϕ₁ = zeros(num_features_nn)
	ϕ₂ = zeros(num_features_nn)
	for i = 1:k
		xvec1 = vcat(xr[:,i], x_mat_h[:,i])
		xvec2 = vcat(xr[:,i+k], x_mat_h[:,i])
		if P.pre_process_nn
			xvec1 = pre_process_nn(xvec1)
			xvec2 = pre_process_nn(xvec2)
		end
		ϕ₁ += nn_ϕ(xvec1)
		ϕ₂ += nn_ϕ(xvec2)
	end
	return ϕ₁, ϕ₂
end

# A workaround for Flux and ForwardDiff not
# working together
function nn_to_func(nn)
	p = Flux.params(nn).order
	function func(x)
		return relu.(p[3]*relu.(p[1]*x+p[2])+ p[4])
	end
	return func
end

function pre_process_nn(x)
	xs = copy(x)

	# Replace y in the state with d
	xs[2] = √((x[1] - x[5])^2 + (x[2] - x[6])^2)

	# Return the first four values which will be
	# [xᵣ, d, θᵣ, vᵣ]
	return xs[1:4]
end

"""
----------------------------------------------
Optimization objectives
----------------------------------------------
"""

# Information gain objective using hand-coded features
function object_ig_hc(u::Vector)
	_, _, ψ_tot = get_ψ_hc(u)
	pq1 = f_tot(ψ_tot)
	pq2 = f_tot(-ψ_tot)
	return -(sum(pq1 .* log2.(M .* pq1 ./ sum(pq1))) + sum(pq2 .* log2.(M .* pq2 ./ sum(pq2))))
end

# Volume removal objective using hand-coded features
function object_vr_hc(u::Vector)
	_, _, ψ_tot = get_ψ_hc(u)
	pq1 = f_tot(ψ_tot)
	pq2 = f_tot(-ψ_tot)
	return -min(sum(1 .- pq1), sum(1 .- pq2))
end

# For optaining optimal trajectory with respect to
# hand-coded features
function neg_reward_hc(u::Vector)
	w_est = [mean(W_tot[i,:]) for i in 1:num_features_hc]
	x = get_x_mat(u, init_x = [0, -0.3, pi/2, 0.4])
	ϕ = ϕ_hc(x)
	return -w_est'*ϕ
end

"""
----------------------------------------------
Other Functions
----------------------------------------------
"""

# For information gain and volume removal objectives
function f_tot(ψ_tot)
	return 1 ./ (1 .+ exp.(-W_tot'*ψ_tot))
end

# Solves for the optimal trajectory based on current
# w estimate and hand-coded features
function solve_optimal_trajectory()
	lb_small = lb[1:num_steps*ctrl_size]
	ub_small = ub[1:num_steps*ctrl_size]
	initvals = rand.(Distributions.Uniform.(lb_small, ub_small))
	inner_optimizer = LBFGS(linesearch = LineSearches.BackTracking())
	objec = neg_reward_hc
	res = Optim.optimize(objec, lb_small, ub_small, initvals,
	                     Fminbox(inner_optimizer),
						 Optim.Options(show_trace=false), autodiff=:forward)
	u = Optim.minimizer(res)
	return u
end

# Function for generating a set of queries corresponding
# to random w pairs to use as a test set
# Output is a matrix of control inputs for the test set
function generate_heuristic_test_set(num_examples)
	u_mat = zeros(num_examples, length(lb))
	for i = 1:num_examples
		mod(i,10) == 0 ? println(i) : nothing
		# Sample two random W's
		w₁ = rand(num_features_hc)
		w₂ = rand(num_features_hc)
		w₁ ./= norm(w₁)
		w₂ ./= norm(w₂)

		# Get the corresponding optimal inputs
		u₁ = get_optimal_inputs(w₁)
		u₂ = get_optimal_inputs(w₂)
		u_mat[i,:] = [u₁; u₂]
	end
	return u_mat
end

# Obtains input from the user
function Input(prompt::String)
    print(prompt)
    return readline()
end

# Trajectory for human car
u_h = [0, 0.41, 1, 0.41, -1, 0.41, 0, 1.3*0.41, 0, 1.3*0.41]
x_mat_h = get_x_mat(u_h)
xpos_h = x_mat_h[1,:]
ypos_h = x_mat_h[2,:]
