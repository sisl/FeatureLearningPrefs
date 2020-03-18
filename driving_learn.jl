"""
----------------------------------------------
Loss functions and metrics for training neural network
----------------------------------------------
"""

# X is a matrix where each row is the flattened states for both trajectories
# First half is trajectory A, first k inputs are state 1, next k are state 2, etc.

# Loss over all training examples
function loss(X, Y, w; zero_out_inds = [])
	loss = 0
	for i = 1:size(X, 1)
		loss += indiv_loss(X[i,:], Y[i], w[:], zero_out_inds=zero_out_inds)
	end
	return loss/size(X, 1)
end

# Loss over one training example
function indiv_loss(x, y, W; ϵ = 1e-8, zero_out_inds = [])
	ϕ₁, ϕ₂ = get_features_hc(x)
	if length(zero_out_inds) > 0
		ϕ₁[zero_out_inds] = zeros(length(zero_out_inds))
		ϕ₂[zero_out_inds] = zeros(length(zero_out_inds))
	end
	ϕ₃, ϕ₄ = get_features_nn(x)
	ψ = vcat(ϕ₁, ϕ₃) - vcat(ϕ₂, ϕ₄)
	pa = 1/(1 + exp(-W'*ψ))
	return -y*log(pa + ϵ) - (1-y)*log(1 - pa + ϵ)
end

# hc_only will make a prediction using only the hand-coded features
# and weights, while nn_only will make a prediction using only the
# neural network features and weights - otherwise, it will use both
# zero_out_inds allows you to zero out the weights for certain
# features (effectively eliminating them)
function predict(x, weights; hc_only=false, nn_only=false, zero_out_inds=[])
	if hc_only
		ϕ₁, ϕ₂ = get_features_hc(x)
		if length(zero_out_inds) > 0
			ϕ₁[zero_out_inds] = zeros(length(zero_out_inds))
			ϕ₂[zero_out_inds] = zeros(length(zero_out_inds))
		end
		ψ = ϕ₁ - ϕ₂
	elseif nn_only
		ϕ₃, ϕ₄ = get_features_nn(x)
		ψ = ϕ₃ - ϕ₄
	else
		ϕ₁, ϕ₂ = get_features_hc(x)
		if length(zero_out_inds) > 0
			ϕ₁[zero_out_inds] = zeros(length(zero_out_inds))
			ϕ₂[zero_out_inds] = zeros(length(zero_out_inds))
		end
		ϕ₃, ϕ₄ = get_features_nn(x)
		ψ = vcat(ϕ₁, ϕ₃) - vcat(ϕ₂, ϕ₄)
	end
	pa = 1/(1 + exp(-weights'*ψ))
	ŷ = pa ≥ 0.5 ? 1 : 0
	return ŷ
end

# Evaluates prediction accuracy on a given dataset
# Inputs have same description as prediction function
function accuracy(X, Y, Weights; hc_only=false, nn_only=false, zero_out_inds=[])
	correct_list = []
	for i = 1:size(X, 1)
		ŷ = predict(X[i,:], Weights; hc_only=hc_only, nn_only=nn_only, zero_out_inds=zero_out_inds)
		ŷ == Y[i] ? correct = 1 : correct = 0
		push!(correct_list, correct)
	end
	return correct_list
end


"""
----------------------------------------------
Train the model and evaluate
----------------------------------------------
"""
# Used for updating linear reward weights
function simple_gd(f, w₀; α=1E-4, itr_max=5)
	w = w₀

	# the learning rate of nn features is 2.0 times more.
	fact = vcat(ones(num_features_hc), ones(num_features_nn)) #*3.0)
	for i=1:itr_max
		df = Flux.gradient(f, w)
		w_new = w - α*df[1].*fact
		(f(w)- f(w_new))/f(w) > 0.00 ? w = w_new : break
		w = w_new
	end
	try
		return w.data
	catch
		return w
	end
end

# Perform one epoch of training to update neural network
function my_train!(loss, nn_ps, w_ps, data, opt; α_w = 0.001, weight_update_every=5, print_every=15, zero_out_inds = [])
	nn_ps = Flux.Params(nn_ps)
	for (i,d) in enumerate(data)
		# Update neural net parameters
		loss_xy(x,y) = loss(x,y,w_ps,zero_out_inds=zero_out_inds)
		nn_gs = Flux.gradient(nn_ps) do
			loss_xy(d...)
		end
		Tracker.update!(opt, nn_ps, nn_gs)
		loss_w(w) = loss(d..., w,zero_out_inds=zero_out_inds)

		if mod(i, weight_update_every) == 0
			# Update weights
			w_ps_new = simple_gd(loss_w, w_ps; α=α_w, itr_max=5)
			w_ps_new == w_ps ? α_w = min(α_w*0.9, 1E-5) : w_ps = w_ps_new
			println(round.(w_ps; digits=3))
		end
		if mod(i, print_every) == 0
			println("progress = $(i/length(data)), loss= $(loss(d..., w_ps))")
		end
	end
	return w_ps
end

# Trains the network for a mixed (hand-coded and neural network) feature set
function train_mixed_nn(prefs, WW; epoch=50, zero_out_inds=[])
	# Defines the neural net
	opt = NADAM()
	w_nn = rand(num_features_nn)

	# get the data
	X = zeros(length(prefs), 2*ns*num_steps*step_time)
	Y = zeros(length(prefs))
	for i = 1:length(prefs)
		x₁ = prefs[i].x₁
		x₂ = prefs[i].x₂
		x₁ = reshape(x₁, (1, length(x₁)))
		x₂ = reshape(x₂, (1, length(x₂)))
		X[i,:] = hcat(x₁, x₂)
		Y[i] = prefs[i].pref == -1 ? 0 : prefs[i].pref
	end
	w_hc = [mean(WW[i,:]) for i in 1:num_features_hc]

	# concatenate hc and nn weights
	w_tot = vcat(w_hc, w_nn)
	println(round.(w_tot; digits=3))
	# train network
	dataset = Iterators.repeated((X, Y), epoch)
	w_tot = my_train!(loss, Flux.params(nn_ϕ), w_tot, dataset, opt, α_w = 0.005,
	                                                     weight_update_every=20, zero_out_inds=zero_out_inds)

	# compute accuracy based on nn + hc_based calculated weight
	acc = accuracy(X, Y, w_tot[:], zero_out_inds=zero_out_inds)
	return w_tot, acc
end
