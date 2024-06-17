import os
import pickle

from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import core.algorithms as algorithms
import core.operations as operations
import core.plot as plot

from core.types import *

name = "dead-units"

flags = {
	"chains": True,
	"versus-theoretical": True,
	"degrees-of-freedom-estimate": True
}

os.makedirs(f'.cache/{name}', exist_ok=True)

def save(stage, data):
	with open(f'.cache/{name}/{stage}.pkl', "wb") as f:
		pickle.dump(data, f)

def load(stage):
	with open(f'.cache/{name}/{stage}.pkl', "rb") as f:
		data = pickle.load(f)
	return data

"""Main"""

if __name__ == "__main__":
	rng = np.random.default_rng(12345)

	# Tasks

	width = 64

	def batcher(batch_size, input_size, weight):
		while True:
			inputs = rng.normal(size=[batch_size, input_size])
			outputs = operations.mlp(weight, inputs)
			yield (inputs, outputs)

	def kill_units_(weight: MLPWeight, kill_rate: float) -> Tuple[MLPWeight, int, int]:
		dead_units_per_layer = []
		for (_, b) in weight[0:-1]:
			mask = rng.uniform(0, 1, size=width) < kill_rate
			dead_units_per_layer.append(np.sum(mask.astype("int")))
			b[mask] = -np.inf

		n_dead_units = np.sum(dead_units_per_layer)
		n_dead_dimensions = 2 * n_dead_units * width + n_dead_units - np.sum((np.roll(dead_units_per_layer, 1) * np.array(dead_units_per_layer))[1:])

		return weight, n_dead_units, n_dead_dimensions


	def count_dead_weights(weight: MLPWeight) -> int:
		grad = jax.grad(lambda w, inputs : operations.mlp(w, inputs).mean())
		batch_size = 512
		batch = next(batcher(batch_size, width, weight))
		sensitivities = np.absolute(operations.pack(grad(weight, batch[0])))

		return np.sum(sensitivities == 0)

	tasks = pd.merge(
		pd.DataFrame({ "n_layers": [3, 4, 5, 6] }),
		pd.DataFrame({ "kill_rate": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] }),
		how="cross"
	)
	tasks["n_hidden_layers"] = tasks["n_layers"] - 2
	tasks["depth"] = tasks["n_layers"] - 1
	tasks["n_hidden_units"] = width * (tasks["n_layers"] - 2)
	tasks[["true_weight", "n_dead_units", "n_dead_dimensions"]] = tasks.astype("O").apply(lambda row : kill_units_(operations.initialise_mlp(dimensions=[width]*row["n_layers"]), row["kill_rate"]), axis=1, result_type="expand")
	tasks["n_dimensions"] = (tasks["n_layers"] - 1) * (width**2 + width)
	#tasks["n_dead_dimensions"] = 2 * tasks["n_dead_units"] * width + tasks["n_dead_units"]
	tasks["n_dead_dimensions_estimate"] = tasks.apply(lambda row : count_dead_weights(row["true_weight"]), axis=1)
	tasks["upper_bound"] =  0.5 * (tasks["n_dimensions"] - tasks["n_dead_dimensions_estimate"])
	tasks = tasks.reset_index(names="task")

	# Chains

	hyperparameters = pd.DataFrame({
		"n_layers": [3, 4, 5, 6],
		"n_steps": 30000,
		"epsilon": [2e-9, 6e-10, 6e-10, 6e-10],
		"beta": 10**7,
		"batch_size": 128,
		"checkpoint_interval": 15000
	})

	# Trials

	chains = pd.merge(
		pd.DataFrame({ "trial": np.arange(3) }),
		pd.merge(tasks, hyperparameters, how="left", on="n_layers"),
		how="cross"
	)
	chains = chains.reset_index(names="chain")

	# Run chains

	def run_trial(chain: pd.Series):
		print(f'{chain["chain"]}:{chain["trial"]}:depth={chain["depth"]}')

		steps = algorithms.sgld(
			loss_fn=lambda w, x, y : chain["beta"] * operations.regression_loss(operations.mlp, w, x, y),
			batcher=batcher(64, width, chain["true_weight"]),
			initial=chain["true_weight"],
			n_steps=chain["n_steps"],
			step_size=chain["epsilon"],
			seed=int(rng.integers(1016)),
			checkpoint_interval=chain["checkpoint_interval"]
		)
		steps["chain"] = chain["chain"]

		return steps

	if flags["chains"]:
		steps = pd.concat(list(chains.apply(run_trial, axis=1))).reset_index(drop=True)
		save("steps", steps)
	else:
		steps = load("steps")

	# Merged data

	data = pd.merge(steps, chains, how="left", left_on="chain", right_index=True)

	# Compute observables

	data["lambda_hat"] = data.groupby("chain")["loss"].rolling(1000).mean().reset_index(0, drop=True)
	data["relative_lambda_hat"] = data.groupby("task")["lambda_hat"].transform(lambda x : x / x.max())

	# Colorscales

	dead_units_colorscale = plot.linear_colorscale(cmap="viridis_r", min=tasks["n_dead_units"].min(), max=tasks["n_dead_units"].max())
	layers_colorscale = plot.linear_colorscale(cmap="viridis_r", min=tasks["n_hidden_layers"].min(), max=tasks["n_hidden_layers"].max())
	epsilon_colorscale = plot.log_colorscale(cmap="cool", min=data["epsilon"].min(), max=data["epsilon"].max())
	bound_colorscale = plot.linear_colorscale(cmap="viridis", min=data["upper_bound"].min(), max=data["upper_bound"].max())
	
	colorscales = {
		"epsilon": plot.log_colorscale(cmap="cool", min=data["epsilon"].min(), max=data["epsilon"].max()),
		"kill_rate": plot.linear_colorscale(cmap="viridis_r", min=data["kill_rate"].min(), max=data["kill_rate"].max()),
		"layers": plot.linear_colorscale(cmap="jet", min=data["depth"].min(), max=data["depth"].max())
	}

	# Ticks

	dead_units_ticks = np.arange(tasks["n_dead_units"].min(), tasks["n_dead_units"].max(), 16).tolist()

	# Plot theoretical versus estimate

	if flags["versus-theoretical"]:
		subdata = data.sort_values("step").groupby("chain").tail(1)

		for n_hidden_layers in subdata["n_hidden_layers"].unique():
			subsubdata = subdata[subdata["n_hidden_layers"] == n_hidden_layers]
			fig, ax = plot.square(data=subsubdata,
				index="task",
				x="n_dead_dimensions", x_label="Dimensions killed",
				y="lambda_hat", y_label=r'$\hat{\lambda}$',
				color="black",
				plotter=partial(plot.errorbar, fmt="o")
			)

		fig, ax = plot.square(data=subdata,
			index="task",
			x="lambda_hat", x_label=r'$\hat{\lambda}$',
			y="upper_bound", y_label="Upper bound",
			group="depth",
			color="depth",
			colorscale=colorscales["layers"],
			color_label='Depth',
			plotter=partial(plot.errorbar, fmt="o")
		)
		plt.plot([0, data["upper_bound"].max()], [0, data["upper_bound"].max()], linewidth=1, linestyle="dashed", color="black", alpha=0.2)

	# Plot degrees of freedom estimates

	if flags["degrees-of-freedom-estimate"]:
		fig, ax = plot.square(data=tasks,
			index="task",
			x="n_dead_dimensions", x_label="Dimensions killed",
			y="n_dead_dimensions_estimate", y_label="Estimated degrees of freedom",
			group="depth",
			color="depth",
			colorscale=colorscales["layers"],
			color_label="Depth",
			plotter=plot.scatter
		)
		plt.plot([0, tasks["n_dead_dimensions"].max()], [0, data["n_dead_dimensions"].max()], linewidth=1, linestyle="dashed", color="black", alpha=0.2)



