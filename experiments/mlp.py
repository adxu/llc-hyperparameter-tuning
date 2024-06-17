import os
import pickle

from functools import partial

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import core.algorithms as algorithms
import core.operations as operations
import core.plot as plot

from core.types import *

name = "mlp"

flags = {
	"chains": True,
	"observables": True,
	"plot-loss-traces": True,
	"plot-lambda-versus-epsilon": True,
	"plot-convergence-times": True,
	"moving-estimates": True,
	"plot-acceptance-probabilities": True,
	"plot-maximal-epsilons": True,
	"plot-optimal-convergence-times": True,
	"radial-traces": True
}

os.makedirs(f'.cache/{name}', exist_ok=True)

rng = np.random.default_rng(12345)

def save(stage, data):
	with open(f'.cache/{name}/{stage}.pkl', "wb") as f:
		pickle.dump(data, f)

def load(stage):
	with open(f'.cache/{name}/{stage}.pkl', "rb") as f:
		data = pickle.load(f)
	return data

"""Main"""

if __name__ == "__main__":
	# Models

	models = pd.DataFrame({
		"depth": np.array([2, 3, 4, 5]),
		"true_lambda": 1,
		"width": 64
	})
	models["n_dimensions"] = models.astype("O").apply(lambda model : (model["depth"] * model["width"]**2 + model["depth"] * model["width"]), axis=1)
	models["n_hidden_layers"] = models.astype("O").apply(lambda model : model["depth"] - 1, axis=1).astype("int")
	models["dimensions"] = models.astype("O").apply(lambda model : [model["width"]] * (model["depth"] + 1), axis=1)
	models["true_weight"] = models.astype("O").apply(lambda model : operations.initialise_mlp(model["dimensions"], method="kaiming"), axis=1)
	models["true_lambda"] = models.astype("O").apply(lambda model : model["true_lambda"] * model["n_dimensions"] / 2, axis=1)

	# Hyperparameters

	n_epsilons = 16
	hyperparameters = pd.concat([
		pd.DataFrame({
			"depth": 2,
			"beta": np.repeat([1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7], n_epsilons),
			"epsilon": [
				*np.logspace(-4, -2, num=n_epsilons),
				*np.logspace(-5, -3, num=n_epsilons),
				*np.logspace(-5, -3, num=n_epsilons),
				*np.logspace(-6, -4, num=n_epsilons),
				*np.logspace(-7, -5, num=n_epsilons),
				*np.logspace(-8, -6, num=n_epsilons),
				*np.logspace(-9, -7, num=n_epsilons)
			],
			"n_steps": 15000,
			"checkpoint_interval": np.repeat([500, 10000, 10000, 10000, 10000, 10000, 500], n_epsilons)
		}),
		pd.DataFrame({
			"depth": 3,
			"beta": np.repeat([1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7], n_epsilons),
			"epsilon": [
				*np.logspace(-4, -2, num=n_epsilons),
				*np.logspace(-5, -3, num=n_epsilons),
				*np.logspace(-6, -4, num=n_epsilons),
				*np.logspace(-7, -5, num=n_epsilons),
				*np.logspace(-8, -6, num=n_epsilons),
				*np.logspace(-9, -7, num=n_epsilons),
				*np.logspace(-10, -8, num=n_epsilons),
			],
			#"n_steps": 15000
			"n_steps": 15000,
			"checkpoint_interval": np.repeat([500, 10000, 10000, 10000, 10000, 10000, 500], n_epsilons)
		}),
		pd.DataFrame({
			"depth": 4,
			"beta": np.repeat([1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7], n_epsilons),
			"epsilon": [
				*np.logspace(-5, -3, num=n_epsilons),
				*np.logspace(-5, -3, num=n_epsilons),
				*np.logspace(-6, -4, num=n_epsilons),
				*np.logspace(-7, -5, num=n_epsilons),
				*np.logspace(-8, -6, num=n_epsilons),
				*np.logspace(-9, -7, num=n_epsilons),
				*np.logspace(-10, -8, num=n_epsilons),
			],
			#"n_steps": 50000
			"n_steps": 15000,
			"checkpoint_interval": np.repeat([500, 10000, 10000, 10000, 10000, 10000, 500], n_epsilons)
		}),
		pd.DataFrame({
			"depth": 5,
			"beta": np.repeat([1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7], n_epsilons),
			"epsilon": [
				*np.logspace(-5, -3, num=n_epsilons),
				*np.logspace(-5, -3, num=n_epsilons),
				*np.logspace(-6, -4, num=n_epsilons),
				*np.logspace(-7, -5, num=n_epsilons),
				*np.logspace(-8, -6, num=n_epsilons),
				*np.logspace(-9, -7, num=n_epsilons),
				*np.logspace(-10, -8, num=n_epsilons),
			],
			#"n_steps": 50000
			"n_steps": 15000,
			"checkpoint_interval": np.repeat([500, 10000, 10000, 10000, 10000, 10000, 500], n_epsilons)
		})
	], ignore_index=True)

	#hyperparameters["epsilon_0"] = hyperparameters["epsilon_0"] * np.power(hyperparameters["depth"] / 2, np.log(hyperparameters["beta"]))
	hyperparameters["epsilon_0"] = hyperparameters["epsilon"] / hyperparameters["beta"]
	hyperparameters["depth_max_beta"] = hyperparameters.groupby("depth")["beta"].transform(lambda group : group.max())
	hyperparameters["depth_min_beta"] = hyperparameters.groupby("depth")["beta"].transform(lambda group : group.min())

	# Chains

	def batcher(batch_size, input_size, weight):
		while True:
			inputs = rng.normal(size=[batch_size, input_size])
			outputs = operations.mlp(weight, inputs)
			yield (inputs, outputs)

	def run_chain(chain: pd.Series):
		print(f'{chain["chain"]}:{chain["trial"]} d={chain["n_dimensions"]}, width={chain["width"]}, depth={chain["depth"]}, epsilon={chain["epsilon"]}, beta={chain["beta"]}')

		steps = algorithms.sgld(
			loss_fn=lambda w, x, y : chain["beta"] * operations.regression_loss(operations.mlp, w, x, y),
			batcher=batcher(64, chain["width"], chain["true_weight"]),
			initial=chain["true_weight"],
			n_steps=chain["n_steps"],
			step_size=chain["epsilon"],
			seed=int(rng.integers(1016)),
			checkpoint_interval=chain["checkpoint_interval"]
		)
		steps["chain"] = chain["chain"]

		return steps

	chains = pd.merge(
		pd.DataFrame({ "trial": np.arange(3) }),
		pd.merge(models, hyperparameters, how="left", on="depth"),
		how="cross"
	)
	chains = chains.reset_index(names="chain")

	if flags["chains"]:
		steps = pd.concat(list(chains.astype("object").apply(run_chain, axis=1)), ignore_index=True)
		save("steps", steps)
	else:
		steps = load("steps")


	if flags["observables"]:
		data = pd.merge(steps, chains, how="left", on="chain")
		data = data.sort_values("step")

		data.loc[data["loss"] > 2 * data["true_lambda"], "loss"] = None

		data["lambda_hat_50"] = data.groupby("chain")["loss"].rolling(50).mean().reset_index(0, drop=True)
		data["relative_lambda_hat_50"] = (data["lambda_hat_50"] / data["true_lambda"])

		data["chain_relative_lambda_hat"] = data.groupby("chain")["lambda_hat_50"].transform(lambda chain : chain / chain.iloc[-1])

		data["lambda_hat"] = data.groupby("chain")["loss"].rolling(500).mean().reset_index(0, drop=True)
		data["relative_lambda_hat"] = (data["lambda_hat"] / data["true_lambda"])

		data["last_step"] = data.groupby("chain")["step"].transform(lambda steps : steps.max())
		data["blowed_up"] = data["last_step"] < data["n_steps"] - 1
		data["blowed_up"] = data.groupby("chain")["lambda_hat"].transform(lambda hats : pd.isna(hats.iloc[-1]))

		chains_data = data.loc[~data["blowed_up"]].groupby("chain")[["acceptance_probability", "epsilon", "beta", "width", "depth", "trial"]].mean()
		chains_data["convergence_time"] = data.loc[(data["relative_lambda_hat_50"] > 0.95) & ~data["blowed_up"]].groupby("chain")["step"].min()
		chains_data = chains_data.merge(
			data.groupby("chain").tail(1)[["chain", "lambda_hat", "relative_lambda_hat"]],
			how="left", on="chain"
		)

		#breakpoint()

		def radial_trace(checkpoint):
			points = algorithms.stochastic_trace(
				loss_fn=lambda w, x, y : operations.regression_loss(operations.mlp, w, x, y),
				batcher=batcher(128, 64, checkpoint["true_weight"]),
				w1=checkpoint["true_weight"],
				w2=checkpoint["weight"],
				n_points=16)
			points["tempered_loss"] = checkpoint["beta"] * points["loss"]
			points["relative_distance"] = points["distance"] / points["distance"].max()
			points["relative_lambda_hat"] = points["loss"] / checkpoint["true_lambda"]
			points["step"] = checkpoint["step"]
			points["beta"] = checkpoint["beta"]
			points["epsilon"] = checkpoint["epsilon"]
			points["depth"] = checkpoint["depth"]
			points["chain"] = checkpoint["chain"]

			"""
			try:
				assert points[points["point"] == points["point"].max()]["loss"].iloc[0] <= 2 * checkpoint["lambda_hat"]
			except:
				breakpoint()
			"""

			return points

		radial_trace_points = pd.concat(list(data[(~pd.isna(data["weight"])) & (~data["blowed_up"])].apply(radial_trace, axis=1))).reset_index(drop=True)


		save("data", data)
		save("chains_data", chains_data)
		save("radial_trace_points", radial_trace_points)

	else:
		data = load("data")
		chains_data = load("chains_data")
		radial_trace_points = load("radial_trace_points")

	healthy_chains_data = chains_data[(chains_data["relative_lambda_hat"] < 1.05) & (chains_data["beta"] >= 1e4)]
	
	#healthy_radial_trace_points = pd.merge(radial_trace_points, healthy_chains_data["chain"], on="chain")

	unhealthy_chains_data = chains_data[chains_data["beta"] < 1e4].groupby(["depth", "beta"]).apply(lambda group : group[group["epsilon"] <= group["epsilon"].median()]).reset_index(drop=True)
	
	#unhealthy_radial_trace_points = pd.merge(radial_trace_points, unhealthy_chains_data["chain"], on="chain")
	
	final_radial_trace_points = pd.concat([
		pd.merge(radial_trace_points, healthy_chains_data["chain"], on="chain"),
		pd.merge(radial_trace_points, unhealthy_chains_data["chain"], on="chain")
	], ignore_index=True)

	bad_chains_data = unhealthy_chains_data[unhealthy_chains_data["beta"] == hyperparameters["beta"].min()].groupby("depth").apply(lambda group : group[group["epsilon"] == group["epsilon"].max()]).reset_index(drop=True)
	bad_radial_trace_points = pd.merge(radial_trace_points, bad_chains_data["chain"], on="chain")

	good_chains_data = healthy_chains_data[healthy_chains_data["beta"] == hyperparameters["beta"].max()].groupby("depth").apply(lambda group : group[group["epsilon"] == group["epsilon"].max()]).reset_index(drop=True)
	good_radial_trace_points = pd.merge(radial_trace_points, good_chains_data["chain"], on="chain")

	optimal_chains_data = healthy_chains_data.sort_values("epsilon").groupby(["depth", "beta", "trial"]).tail(1)
	optimal_chains_data = optimal_chains_data.groupby(["depth", "beta"]).mean().reset_index()

	colorscales = {
		"beta": plot.log_colorscale(cmap="plasma_r", min=hyperparameters["beta"].min(), max=hyperparameters["beta"].max()),
		"epsilon_0": plot.log_colorscale(cmap="cool", min=hyperparameters["epsilon_0"].min(), max=hyperparameters["epsilon_0"].max()),
		"epsilon": plot.log_colorscale(cmap="cool", min=hyperparameters["epsilon"].min(), max=hyperparameters["epsilon"].max()),
		"width": plot.log_colorscale(cmap="winter_r", min=models["width"].min(), max=models["width"].max()),
		"n_dimensions": plot.log_colorscale(cmap="winter_r", min=models["n_dimensions"].min(), max=models["n_dimensions"].max()),
		"depth": plot.log_colorscale(cmap="winter_r", min=models["depth"].min(), max=models["depth"].max()),
		"step": plot.linear_colorscale(cmap="rainbow", min=data["step"].min(), max=data["step"].max())
	}

	labels = {
		"beta": r'$\tilde{\beta}$',
		"epsilon": r'$\epsilon$',
		"step": r'$\tau$',
		"lambda_hat": r'$\hat{\lambda}(\tau)$',
		"relative_lambda_hat": r'$\frac{\hat{\lambda}(\tau)}{\lambda}$',
		"final_lambda_hat": r'$\hat{\lambda}$',
		"final_relative_lambda_hat": r'$\frac{\hat{\lambda}}{\lambda}$',
		"distance": r'$\| w_i - w^* \|$',
		"relative_distance": r'$\frac{\| w_i - w^* \|}{\| w_\tau - w^* \|}$',
		"acceptance_probability": r'$p(\tau)$',
		"mean_acceptance_probability": r'$\bar{p}$',
		"convergence_time": r'$T_{\alpha}$',
		"optimal_convergence_time": r'$T^*_{\alpha}(\tilde{\beta})$',
		"maximal_epsilon": r'$\epsilon_{max}(\tilde{\beta})$',
		"tempered_loss": r'$\tilde{\beta} L(w_i, \mathcal{B}_i)$',
		"loss": r'$L(w_i, \mathcal{B}_i)$'
	}

	if flags["plot-loss-traces"]:
		fig, axs = plot.grid(
			data=data,
			x="step", x_label=labels["step"],
			y="relative_lambda_hat", y_label=labels["relative_lambda_hat"],
			column="depth",
			row="beta",
			group=["epsilon"],
			color="epsilon",
			colorscale=colorscales["epsilon"],
			color_label=labels["epsilon"],
			plotter=partial(plot.fill_between, alpha=0.1)
		)
		#axs[0][0].set_ylim(0, 1.25)
		"""
		for true_lambda in models["true_lambda"]:
			ax.axhline(true_lambda, color="gray", alpha=0.5)
		"""

	if flags["plot-lambda-versus-epsilon"]:

		subdata = chains_data
		#subdata = subdata.loc[~subdata["blowed_up"]]
		fig, axs = plot.column(data=subdata,
			x="epsilon", x_label=labels["epsilon"],
			y="relative_lambda_hat", y_label=labels["final_relative_lambda_hat"],
			row="depth",
			group="beta",
			color="beta",
			colorscale=colorscales["beta"],
			color_label=labels["beta"],
			plotter=partial(plot.fill_between, alpha=0.1)
		)
		axs[0].set_xscale("log")
		#axs[0].set_ylim(0.0, 1.5)

		for ax in axs:
			ax.axhline(1.05, color="gray", alpha=0.5, ls="--", lw=0.5)
			ax.axhline(0.95, color="gray", alpha=0.5, ls="--", lw=0.5)
			ax.axhline(1.0, color="black", alpha=0.5, ls="-", lw=0.5)

		for i, depth in enumerate(np.sort(data["depth"].unique())):
			ax2 = axs[i].twinx()
			ax2.set_ylabel(labels["mean_acceptance_probability"])
			ax2.set_ylim(0, 1)
			#cb2 = fig.colorbar(colorscales["n_dimensions"], ax=ax2, location="top", shrink=0.6)
			#cb2.set_label("Dimensions")

			subdata = chains_data[chains_data["depth"] == depth]
			plot.grouped(ax2, data=subdata,
				x="epsilon",
				y="acceptance_probability",
				group="beta",
				color="beta",
				colorscale=colorscales["beta"],
				plotter=partial(plot.errorbar, fmt="x-", linewidth=0.2, ms=3, alpha=0.25)
			)

	if flags["moving-estimates"]:

		subdata = data[(data["step"] + 1) % 1000 == 0]
		fig, axs = plot.column(data=subdata,
			x="epsilon", x_label=labels["epsilon"],
			y="relative_lambda_hat", y_label=labels["relative_lambda_hat"],
			row="depth",
			group=["beta", "step"],
			color="step",
			colorscale=colorscales["step"],
			color_label=labels["step"],
			plotter=partial(plot.errorbar, fmt="o-", linewidth=0.5, ms=1, alpha=0.5)
		)
		axs[0].set_xscale("log")
		for ax in axs:
			ax.axhline(1.05, color="gray", alpha=0.5, ls="--", lw=0.5)
			ax.axhline(0.95, color="gray", alpha=0.5, ls="--", lw=0.5)
			ax.axhline(1.0, color="black", alpha=0.5, ls="-", lw=0.5)

	if flags["plot-convergence-times"]:

		subdata = chains_data
		#subdata = subdata.loc[~subdata["blowed_up"]]
		fig, axs = plot.column(data=subdata,
			x="epsilon", x_label=labels["epsilon"],
			y="relative_lambda_hat", y_label=labels["final_relative_lambda_hat"],
			row="depth",
			group=["beta", "width"],
			color="beta",
			colorscale=colorscales["beta"],
			color_label=labels["beta"],
			plotter=partial(plot.fill_between, alpha=0.1)
		)
		axs[0].set_xscale("log")
		#axs[0].set_ylim(0.0, 1.5)
		for ax in axs:
			ax.axhline(1.05, color="gray", alpha=0.5, ls="--", lw=0.5)
			ax.axhline(0.95, color="gray", alpha=0.5, ls="--", lw=0.5)
			ax.axhline(1.0, color="black", alpha=0.5, ls="-", lw=0.5)

		for i, depth in enumerate(np.sort(data["depth"].unique())):
			ax2 = axs[i].twinx()
			ax2.set_ylabel(labels["convergence_time"])
			ax2.set_yscale("log")
			#cb2 = fig.colorbar(colorscales["n_dimensions"], ax=ax2, location="top", shrink=0.6)
			#cb2.set_label("Dimensions")

			subdata = healthy_chains_data[healthy_chains_data["depth"] == depth]
			plot.grouped(ax2, data=subdata,
				x="epsilon",
				y="convergence_time",
				group="beta",
				color="beta",
				colorscale=colorscales["beta"],
				plotter=partial(plot.errorbar, fmt="v-", linewidth=0.2, ms=2, alpha=0.25)
			)

	if flags["plot-acceptance-probabilities"]:

		subdata = optimal_chains_data
		fig, ax = plot.square(data=subdata,
			x="depth", x_label="Depth",
			y="acceptance_probability", y_label=labels["mean_acceptance_probability"],
			color="black",
			plotter=partial(plot.errorbar, fmt="x-", linewidth=0.5, ms=5, alpha=0.5)
		)
		ax.set_xscale("log")

		subdata = optimal_chains_data
		fig, ax = plot.square(data=subdata,
			x="beta", x_label=labels["beta"],
			y="acceptance_probability", y_label=labels["mean_acceptance_probability"],
			color="black",
			plotter=partial(plot.errorbar, fmt="x-", linewidth=0.5, ms=5, alpha=0.5)
		)
		ax.set_xscale("log")

	if flags["plot-maximal-epsilons"]:

		fig, ax = plot.square(data=optimal_chains_data,
			x="beta", x_label=labels["beta"],
			y="epsilon", y_label=labels["maximal_epsilon"],
			group="depth",
			color="depth",
			colorscale=colorscales["depth"],
			color_label="Depth",
			plotter=partial(plot.errorbar, fmt="D-", linewidth=0.5, ms=5, alpha=0.5)
		)
		ax.set_xscale("log")
		ax.set_yscale("log", base=2)

		fig, ax = plot.square(data=optimal_chains_data,
			x="depth", x_label="Depth",
			y="epsilon", y_label=labels["maximal_epsilon"],
			group="beta",
			color="beta",
			colorscale=colorscales["beta"],
			color_label=labels["beta"],
			plotter=partial(plot.errorbar, fmt="D-", linewidth=0.5, ms=5, alpha=0.5)
		)
		ax.set_xscale("log")
		ax.set_yscale("log", base=2)

	if flags["plot-optimal-convergence-times"]:

		fig, ax = plot.square(data=optimal_chains_data,
			x="beta", x_label=labels["beta"],
			y="convergence_time", y_label=labels["optimal_convergence_time"],
			group="depth",
			color="depth",
			colorscale=colorscales["depth"],
			color_label="Depth",
			plotter=partial(plot.errorbar, fmt="v-", linewidth=0.5, ms=5, alpha=0.5)
		)
		ax.set_xscale("log")
		ax.set_yscale("log", base=2)

	if flags["radial-traces"]:

		fig, axs = plot.grid(data=final_radial_trace_points[final_radial_trace_points["step"] == 10000 - 1],
			index="point",
			x="relative_distance", x_label=labels["relative_distance"],
			y="tempered_loss", y_label=labels["tempered_loss"],
			row="beta",
			column="depth",
			group=["epsilon"],
			color="epsilon",
			colorscale=colorscales["epsilon"],
			color_label=labels["epsilon"],
			plotter=plot.fill_between
		)

		fig, axs = plot.column(data=good_radial_trace_points,
			index="point",
			x="distance", x_label=labels["distance"],
			y="loss", y_label=labels["loss"],
			row="depth",
			group=["step", "epsilon"],
			color="step",
			colorscale=colorscales["step"],
			color_label=labels["step"],
			plotter=plot.fill_between
		)

		fig, axs = plot.column(data=bad_radial_trace_points,
			index="point",
			x="distance", x_label=labels["distance"],
			y="loss", y_label=labels["loss"],
			row="depth",
			group=["step", "epsilon"],
			color="step",
			colorscale=colorscales["step"],
			color_label=labels["step"],
			plotter=plot.fill_between
		)
