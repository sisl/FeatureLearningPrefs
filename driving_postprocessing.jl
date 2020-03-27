# This file contains various helper functions for running training
# and analyzing data

function plot_training_eval_test(acc_train, acc_eval, acc_test)
	iter = collect(1:length(acc_train))
	p = Plots.plot(iter, acc_train)
	Plots.plot!(p, iter, acc_eval)
	Plots.plot!(p, iter, acc_test)
	ra_train = [mean(acc_train[1:i]) for i in 1:length(acc_train)]
	ra_eval = [mean(acc_eval[1:i]) for i in 1:length(acc_eval)]
	ra_test = [mean(acc_test[1:i]) for i in 1:length(acc_test)]
	Plots.plot!(p, iter, ra_train)
	Plots.plot!(p, iter, ra_eval)
	Plots.plot!(p, iter, ra_test)
	return p
end

# Get training and test data from preferences
function get_datasets(prefs_train, prefs_test, final_weights)
	X_train = zeros(length(prefs_train), 2*ns*num_steps*step_time)
	Y_train = zeros(length(prefs_train))

	for i = 1:length(prefs_train)
	    x₁ = prefs_train[i].x₁
	    x₂ = prefs_train[i].x₂
	    x₁ = reshape(x₁, (1, length(x₁)))
	    x₂ = reshape(x₂, (1, length(x₂)))
	    X_train[i,:] = hcat(x₁, x₂)
	    Y_train[i] = prefs_train[i].pref == -1 ? 0 : prefs_train[i].pref
	end

	X_test = zeros(length(prefs_test), 2*ns*num_steps*step_time)
	Y_test = zeros(length(prefs_test))
	for i = 1:length(prefs_test)
	    x₁ = prefs_test[i].x₁
	    x₂ = prefs_test[i].x₂
	    x₁ = reshape(x₁, (1, length(x₁)))
	    x₂ = reshape(x₂, (1, length(x₂)))
	    X_test[i,:] = hcat(x₁, x₂)
	    Y_test[i] = prefs_test[i].pref == -1 ? 0 : prefs_test[i].pref
	end

	W = [mean(final_weights[i,:]) for i in 1:num_features_hc]

	return prefs_train, X_train, Y_train, X_test, Y_test, W
end

# Run training for multiple trials and get accuracies on training, test, and eval set
function run_mixed_very_fancy_save(prefs_train, X_test, Y_test, W, savefile; epoch=40, num_trials=40)
	num_train = convert(Int64, round(0.7*length(prefs_train)))
	num_eval = length(prefs_train) - num_train

	acc_train_vec = zeros(num_trials)
	acc_eval_vec = zeros(num_trials)
	acc_test_vec = zeros(num_trials)
	best_eval = 0
	for i = 1:num_trials
		println("Trial: $i")
		w_tot, acc = train_mixed_nn(prefs_train[1:num_train], W; epoch=epoch)
		acc_train_vec[i] = mean(acc)
		# Get eval set
		X_eval = zeros(num_eval, 2*ns*num_steps*step_time)
		Y_eval = zeros(num_eval)
		for j = 1:num_eval
			x₁ = prefs_train[j+num_train].x₁
			x₂ = prefs_train[j+num_train].x₂
			x₁ = reshape(x₁, (1, length(x₁)))
			x₂ = reshape(x₂, (1, length(x₂)))
			X_eval[j,:] = hcat(x₁, x₂)
			Y_eval[j] = prefs_train[j+num_train].pref == -1 ? 0 : prefs_train[j+num_train].pref
		end
		acc_eval_vec[i] = mean(accuracy(X_eval, Y_eval, w_tot[:]; hc_only=false))
		acc_test_vec[i] = mean(accuracy(X_test, Y_test, w_tot[:]; hc_only=false))
		if mean(accuracy(X_eval, Y_eval, w_tot[:]; hc_only=false)) > best_eval && i > 5
			@save savefile nn_ϕ w_tot
			best_eval = mean(accuracy(X_eval, Y_eval, w_tot[:]; hc_only=false))
		end
	end
	return acc_train_vec, acc_eval_vec, acc_test_vec
end
