using Interact

resetPGFPlotsPreamble()
pushPGFPlotsPreamble("\\usepackage{carshapes}")
include("support_code.jl")

function get_car_string(heading, x, y, body, window; width=1.5)
	return "\\node[sedan top,body color="*body*",window color="*window*", minimum width="*string(width)*"cm,rotate="*string(heading)*",scale = 0.27] at (axis cs:"*string(x)*", "*string(y)*") {};"
end

function viz_features(nn)
	currSavePlot = 0

	@manipulate for fileName in textbox(value="myFile.pdf",label="File Name") |> onchange,
		savePlot in button("Save Plot"),
		nbins = 100,
		xmin = -0.5,
		xmax = 0.5,
		ymin = -3,
		ymax = 3,
		feature = 1,
		v in slider(-1:0.2:1),
		θ in slider(0:π/4:2*π),
		t in slider(1:k)


		function get_heat(x, y)
			input_nn = vcat([x, y, θ, v], x_mat_h[:,t])
			if P.pre_process_nn
				input_nn = pre_process_nn(input_nn)
			end
			return -nn(input_nn).data[feature]
		end

		ax = Axis(PGFPlots.Plots.Image(get_heat, (xmin, xmax), (ymin, ymax), colormap = pasteljet, colorbar = false))
		#push!(ax, PGFPlots.Plots.Scatter([x_mat_h[1,t]], [x_mat_h[2,t]], markSize=5))
		push!(ax, PGFPlots.Plots.Command(get_car_string(rad2deg(x_mat_h[3,t]), x_mat_h[1,t], x_mat_h[2,t], "red", "black")))
		#push!(ax, PGFPlots.Plots.Command(get_car_string(rad2deg(θ), 0.7, 2.7, "blue", "black")))
		push!(ax, PGFPlots.Plots.Linear([-0.255, -0.255], [ymin, ymax], mark="none", style="white,solid,thick"))
		push!(ax, PGFPlots.Plots.Linear([0.255, 0.255], [ymin, ymax], mark="none", style="white,solid,thick"))
		push!(ax, PGFPlots.Plots.Linear([-0.085, -0.085], [ymin, ymax], mark="none", style="white,dashed,thick"))
		push!(ax, PGFPlots.Plots.Linear([0.085, 0.085], [ymin, ymax], mark="none", style="white,dashed,thick"))
		ax.axisEqualImage = true
		ax.xmin = xmin
		ax.xmax = xmax
		ax.ymin = ymin
		ax.ymax = ymax
		ax.width = "4cm"
		ax.height = "15cm"

		if savePlot > currSavePlot
			currSavePlot = savePlot
			ax2 = Axis(PGFPlots.Plots.Image(get_heat, (xmin, xmax), (ymin, ymax), colormap = pasteljet, colorbar = false))
			#push!(ax, PGFPlots.Plots.Scatter([x_mat_h[1,t]], [x_mat_h[2,t]], markSize=5))
			push!(ax2, PGFPlots.Plots.Command(get_car_string(rad2deg(x_mat_h[3,t]), x_mat_h[1,t], x_mat_h[2,t], "red", "black")))
			#push!(ax, PGFPlots.Plots.Command(get_car_string(rad2deg(θ), 0.7, 2.7, "blue", "black")))
			push!(ax2, PGFPlots.Plots.Linear([-0.255, -0.255], [ymin, ymax], mark="none", style="white,solid,thick"))
			push!(ax2, PGFPlots.Plots.Linear([0.255, 0.255], [ymin, ymax], mark="none", style="white,solid,thick"))
			push!(ax2, PGFPlots.Plots.Linear([-0.085, -0.085], [ymin, ymax], mark="none", style="white,dashed,thick"))
			push!(ax2, PGFPlots.Plots.Linear([0.085, 0.085], [ymin, ymax], mark="none", style="white,dashed,thick"))
			ax2.axisEqualImage = true
			ax2.xmin = xmin
			ax2.xmax = xmax
			ax2.ymin = ymin
			ax2.ymax = ymax
			ax2.width = "4cm"
			ax2.height = "15cm"
			PGFPlots.save(fileName, ax2, include_preamble=false)
		end

		return ax
	end
end

function plot_feature_PGF(nn, v, θ, t; xmin=-0.5, xmax=0.5, ymin=-3, ymax=3)
	function get_heat(x, y)
		input_nn = vcat([x, y, θ, v], x_mat_h[:,t])
		if P.pre_process_nn
			input_nn = pre_process_nn(input_nn)
		end
		return -nn(input_nn).data[1]
	end

	ax = Axis(PGFPlots.Plots.Image(get_heat, (xmin, xmax), (ymin, ymax), colormap = pasteljet, colorbar = false)) #zmin=-1, zmax=1,
	push!(ax, PGFPlots.Plots.Command(get_car_string(rad2deg(x_mat_h[3,t]), x_mat_h[1,t], x_mat_h[2,t], "red", "black")))
	push!(ax, PGFPlots.Plots.Linear([-0.255, -0.255], [ymin, ymax], mark="none", style="white,solid,thick"))
	push!(ax, PGFPlots.Plots.Linear([0.255, 0.255], [ymin, ymax], mark="none", style="white,solid,thick"))
	push!(ax, PGFPlots.Plots.Linear([-0.085, -0.085], [ymin, ymax], mark="none", style="white,dashed,thick"))
	push!(ax, PGFPlots.Plots.Linear([0.085, 0.085], [ymin, ymax], mark="none", style="white,dashed,thick"))
	ax.axisEqualImage = true
	ax.xmin = xmin
	ax.xmax = xmax
	ax.ymin = ymin
	ax.ymax = ymax
	ax.width = "4cm"
	ax.height = "15cm"

	return ax
end

function plot_trajectory_PGF(u; xmin=-0.5, xmax=0.5, ymin=-3, ymax=3)
	x = get_x_mat(u, init_x = [0, -0.3, π/2, 0.4])
	x_r = x[1,:]
	y_r = x[2,:]
	x_h = x_mat_h[1,:]
	y_h = x_mat_h[2,:]
	ax = Axis(PGFPlots.Plots.Linear([-0.255, -0.255], [ymin, ymax], mark="none", style="black,solid,thick, name path=A"))
	push!(ax, PGFPlots.Plots.Linear([0.255, 0.255], [ymin, ymax], mark="none", style="black,solid,thick, name path=B"))
	push!(ax, PGFPlots.Plots.Linear([-0.085, -0.085], [ymin, ymax], mark="none", style="white,dashed,thick"))
	push!(ax, PGFPlots.Plots.Linear([0.085, 0.085], [ymin, ymax], mark="none", style="white,dashed,thick"))
	push!(ax, PGFPlots.Plots.Command("\\path[name path=axis] (axis cs:$xmin,$ymin) -- (axis cs:$xmin,$ymax);"))
	push!(ax, PGFPlots.Plots.Command("\\path[name path=axisright] (axis cs:$xmax,$ymin) -- (axis cs:$xmax,$ymax);"))
	push!(ax, PGFPlots.Plots.Command("\\addplot[pastelGreen!40] fill between[of=A and axis];"))
	push!(ax, PGFPlots.Plots.Command("\\addplot[pastelGreen!40] fill between[of=B and axisright];"))
	push!(ax, PGFPlots.Plots.Command("\\addplot[black!40] fill between[of=A and B];"))
	push!(ax, PGFPlots.Plots.Linear(x_r, y_r, mark="none", style="red,solid, very thick"))
	push!(ax, PGFPlots.Plots.Linear(x_h, y_h, mark="none", style="blue,solid, very thick"))
	ax.xmin = xmin
	ax.xmax = xmax
	ax.ymin = ymin
	ax.ymax = ymax
	ax.width = "4cm"
	ax.height = "15cm"
	ax.axisEqualImage = true
	return ax
end
