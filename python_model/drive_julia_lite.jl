
include("load_python_lib.jl")

iA = [0.25 for _ in 1:10]
iB = [0.75 for _ in 1:10]
py"sim_trajectory"(iA, iB)
