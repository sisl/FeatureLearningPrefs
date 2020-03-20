using PyCall

function sim_drive_julia()
	# This function is a test for calling python in julia
	# we call a function from a local module.

	py"""
    import sys
    import numpy as np
    from utils_driving import *

    def sim_drive(sim_env, input_u, watch=True):
        sim_env.feed(input_u)
        phi = sim_env.get_features()
        if watch: sim_env.watch(1)
        return phi

    def get_phi(input_u):
        sim_env = Driver()
        sim_env.feed(input_u)
        phi = sim_env.get_features()
        return phi

    def sim_final(input_u, watch=True):
        sim_env = Driver()
        sim_env.feed(input_u)
        phi = sim_env.get_features()
        if watch: sim_env.watch(1)
        return phi

    def sim_trajectory(input_A, input_B, watch=True):
        sim_env = Driver()
        s = 0
        while s == 0:
            selection = input('A/B to watch, 1/2 to vote: ').lower()
            if selection == 'a':
                phi_A = sim_drive(sim_env, input_A, watch)
            elif selection == 'b':
                phi_B = sim_drive(sim_env, input_B, watch)
            elif selection == '1':
                s = 1
            elif selection == '2':
                s = -1
        return phi_A, phi_B, s
      """
end

# The line below allows to call local python modules.
current_folder = pwd()
drive_library  = string(current_folder, "/python_model/")
pushfirst!(PyVector(pyimport("sys")."path"), drive_library)

sim_drive_julia()
