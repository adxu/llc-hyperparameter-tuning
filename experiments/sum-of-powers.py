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

name = "hyperparameters"

flags = {
	"chains": True,
	"observables": True,
	"plot-loss-traces": True,
	"plot-lambda-versus-epsilon": True,
	"plot-convergence-times": True,
	"plot-acceptance-probabilities": True,
	"plot-maximal-epsilons": True,
	"plot-optimal-convergence-times": True
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

	models = pd.merge(
		pd.DataFrame({ "n_dimensions": 1024 * np.array([8, 16, 32, 64, 128]) }),
		#pd.DataFrame({ "n_dimensions": 1024 * np.array([2, 4, 8, 16, 32]) }),
		pd.merge(
			pd.DataFrame({ "relative_rank": np.array([1, 2, 4]) / 4 }),
			pd.DataFrame({ "order": np.array([2, 4, 8]) }),
			how="cross"
		),
		how="cross"
	)
	models["rank"] = models.apply(lambda model : model["n_dimensions"] * model["relative_rank"], axis=1).astype("int")
	models["true_lambda"] = models.apply(lambda model : model["rank"] / model["order"], axis=1)

	# Hyperparameters

	hyperparameters = pd.DataFrame({
		"order": np.repeat([2, 4, 8], 24),
		"epsilon": [
			*np.logspace(-7, -1, num=24),
			*np.logspace(-5, -1, num=24),
			*np.logspace(-4, -1, num=24)
		]
	}).merge(
		pd.DataFrame({
			"order": np.repeat([2, 4, 8], 5),
			"beta": [
				*[10, 100, 1000, 10000, 100000],
				*[10, 100, 1000, 10000, 100000],
				*[10, 100, 1000, 10000, 100000]
			]
		}),
		how="left", on="order"
	).merge(
		pd.DataFrame({
			"order": [2, 4, 8],
			"n_steps": [2000, 2000, 3000]
		}),
		how="left", on="order"	
	)
	#hyperparameters["epsilon_0"] = hyperparameters["epsilon_0"] * np.power(hyperparameters["order"] / 2, np.log(hyperparameters["beta"]))
	hyperparameters["epsilon_0"] = hyperparameters["epsilon"] / hyperparameters["beta"]

	# Chains

	def run_chain(chain: pd.Series):
		print(f'{chain["chain"]}:{chain["trial"]} d={chain["n_dimensions"]}, rank={chain["rank"]}, order={chain["order"]}, epsilon={chain["epsilon"]}, beta={chain["beta"]}')

		steps = algorithms.ld(
			loss_fn=lambda w : chain["beta"] * jnp.sum(w[:chain["rank"]]**chain["order"]),
			initial=np.zeros(chain["n_dimensions"]),
			n_steps=chain["n_steps"],
			step_size=chain["epsilon"],
			seed=int(rng.integers(1024)),
		)
		steps["chain"] = chain["chain"]

		return steps

	chains = pd.merge(
		pd.DataFrame({ "trial": np.arange(3) }),
		pd.merge(models, hyperparameters, how="left", on="order"),
		how="cross"
	)
	chains = chains.reset_index(names="chain")

	if flags["chains"]:
		steps = pd.concat(list(chains.astype("object").apply(run_chain, axis=1)), ignore_index=True)
		save("steps", steps)
	else:
		steps = load("steps")

	if flags["observables"]:
		data = pd.merge(steps, chains, how="left", on="chain").sort_values("step")

		data["lambda_hat_50"] = data.groupby("chain")["loss"].rolling(50).mean().reset_index(0, drop=True)
		data["relative_lambda_hat_50"] = (data["lambda_hat_50"] / data["true_lambda"]).clip(0, 2)

		#data["chain_relative_lambda_hat"] = data.groupby("chain")["lambda_hat_50"].transform(lambda chain : chain / chain.iloc[-1])

		data["lambda_hat"] = data.groupby("chain")["loss"].rolling(500).mean().reset_index(0, drop=True)
		data["relative_lambda_hat"] = (data["lambda_hat"] / data["true_lambda"]).clip(0, 2)

		data["last_step"] = data.groupby("chain")["step"].transform(lambda steps : steps.max())
		data["blowed_up"] = data["last_step"] < data["n_steps"] - 1

		# Observables

		mean_acceptance_probabilities = data.loc[~data["blowed_up"]].groupby("chain")[["acceptance_probability", "epsilon", "beta", "order", "relative_rank", "n_dimensions"]].mean().reset_index()

		convergence_times = data.loc[(data["relative_lambda_hat_50"] > 0.95) & ~data["blowed_up"]].groupby("chain").head(1)

		final_estimates = data.groupby("chain").tail(1)

		"""
		reference_epsilons = final_estimates.sort_values("epsilon")
		reference_epsilons["epsilon_density"] = reference_epsilons.groupby(["n_dimensions", "relative_rank", "order", "beta"])["lambda_hat"].rolling(5).std()
		reference_epsilons = reference_epsilons.groupby(["n_dimensions", "relative_rank", "order", "beta"]).apply(lambda group : group.loc[group["epsilon_density"].idxmax()])
		"""

		maximal_epsilons = final_estimates[final_estimates["relative_lambda_hat"] < 1.05].sort_values("epsilon").groupby(["n_dimensions", "relative_rank", "order", "beta", "trial"]).tail(1)
		maximal_epsilons = pd.merge(
			maximal_epsilons[["n_dimensions", "rank", "relative_rank", "order", "beta", "epsilon", "chain"]],
			convergence_times[["chain", "step"]],
			how="left", on="chain"
		).merge(mean_acceptance_probabilities[["chain", "acceptance_probability"]], how="left", on="chain")
		maximal_epsilons = maximal_epsilons.groupby(["n_dimensions", "relative_rank", "order", "beta"]).mean().reset_index()

		save("data", data)
		save("mean_acceptance_probabilities", mean_acceptance_probabilities)
		save("convergence_times", convergence_times)
		save("final_estimates", final_estimates)
		save("maximal_epsilons", maximal_epsilons)
	else:
		data = load("data")
		mean_acceptance_probabilities = load("mean_acceptance_probabilities")
		convergence_times = load("convergence_times")
		final_estimates = load("final_estimates")
		maximal_epsilons = load("maximal_epsilons")

	colorscales = {
		"beta": plot.log_colorscale(cmap="plasma_r", min=hyperparameters["beta"].min(), max=hyperparameters["beta"].max()),
		"epsilon_0": plot.log_colorscale(cmap="cool", min=hyperparameters["epsilon_0"].min(), max=hyperparameters["epsilon_0"].max()),
		"epsilon": plot.log_colorscale(cmap="cool", min=hyperparameters["epsilon"].min(), max=hyperparameters["epsilon"].max()),
		"rank": plot.log_colorscale(cmap="winter_r", min=models["rank"].min(), max=models["rank"].max()),
		"n_dimensions": plot.log_colorscale(cmap="winter_r", min=models["n_dimensions"].min(), max=models["n_dimensions"].max()),
		"relative_rank": plot.linear_colorscale(cmap="winter_r", min=models["relative_rank"].min(), max=models["relative_rank"].max()),
		"order": plot.log_colorscale(cmap="winter_r", min=models["order"].min(), max=models["order"].max())
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
		subdata = data
		fig, axs = plot.grid(
			data=subdata,
			x="step", x_label=labels["step"],
			y="relative_lambda_hat", y_label=labels["relative_lambda_hat"],
			column="order",
			row="beta",
			group="epsilon",
			color="epsilon",
			colorscale=colorscales["epsilon"],
			color_label=labels["epsilon"],
			plotter=partial(plot.fill_between, alpha=0.1)
		)
		"""
		for true_lambda in models["true_lambda"]:
			ax.axhline(true_lambda, color="gray", alpha=0.5)
		"""

	if flags["plot-lambda-versus-epsilon"]:
		fig, axs = plot.column(data=data,
			x="epsilon", x_label=labels["epsilon"],
			y="relative_lambda_hat", y_label=labels["final_relative_lambda_hat"],
			row="order",
			group=["beta"],
			color="beta",
			colorscale=colorscales["beta"],
			color_label=labels["beta"],
			plotter=partial(plot.fill_between, alpha=0.1)
		)
		axs[0].set_xscale("log")
		axs[0].set_ylim(0.8, 1.2)
		for ax in axs:
			ax.axhline(1.05, color="gray", alpha=0.5, ls="--", lw=0.5)
			ax.axhline(0.95, color="gray", alpha=0.5, ls="--", lw=0.5)
			ax.axhline(1.0, color="black", alpha=0.5, ls="-", lw=0.5)

		for i, order in enumerate(data["order"].unique()):
			subdata = final_estimates[final_estimates["order"] == order]
			subdata = subdata.loc[~subdata["blowed_up"]]

			ax2 = axs[i].twinx()
			ax2.set_ylabel(labels["mean_acceptance_probability"])

			subdata = mean_acceptance_probabilities[mean_acceptance_probabilities["order"] == order]
			plot.grouped(ax2, data=subdata,
				x="epsilon",
				y="acceptance_probability",
				group="beta",
				color="beta",
				colorscale=colorscales["beta"],
				plotter=partial(plot.errorbar, fmt="x-", linewidth=0.2, ms=3, alpha=0.25)
			)

	if flags["plot-convergence-times"]:
		fig, axs = plot.column(data=data,
			x="epsilon", x_label=labels["epsilon"],
			y="relative_lambda_hat", y_label=labels["final_relative_lambda_hat"],
			row="order",
			group=["beta"],
			color="beta",
			colorscale=colorscales["beta"],
			color_label=labels["beta"],
			plotter=partial(plot.fill_between, alpha=0.1)
		)
		axs[0].set_xscale("log")
		axs[0].set_ylim(0.8, 1.2)
		for ax in axs:
			ax.axhline(1.05, color="gray", alpha=0.5, ls="--", lw=0.5)
			ax.axhline(0.95, color="gray", alpha=0.5, ls="--", lw=0.5)
			ax.axhline(1.0, color="black", alpha=0.5, ls="-", lw=0.5)


		for i, order in enumerate(data["order"].unique()):
			subdata = final_estimates[final_estimates["order"] == order]
			subdata = subdata.loc[~subdata["blowed_up"]]

			ax2 = axs[i].twinx()
			ax2.set_ylabel(labels["convergence_time"])
			ax2.set_yscale("log")
			#cb2 = fig.colorbar(colorscales["n_dimensions"], ax=ax2, location="top", shrink=0.6)
			#cb2.set_label("Dimensions")

			subdata = convergence_times[convergence_times["order"] == order]
			plot.grouped(ax2, data=subdata,
				x="epsilon",
				y="step",
				group=["beta"],
				color="beta",
				colorscale=colorscales["beta"],
				plotter=partial(plot.errorbar, fmt="v-", linewidth=0.2, ms=2, alpha=0.25)
			)

	if flags["plot-acceptance-probabilities"]:
		#breakpoint()
		#subdata = data.loc[~data["blowed_up"]].groupby("chain")[["epsilon", "acceptance_probability", "beta", "order", "n_dimensions", "rank"]].mean()
		fig, axs = plot.column(data=maximal_epsilons,
			x="n_dimensions", x_label="Dimensions",
			y="acceptance_probability", y_label=labels["mean_acceptance_probability"],
			row="order",
			group=["beta", "relative_rank"],
			color="relative_rank",
			colorscale=colorscales["relative_rank"],
			color_label="Relative rank",
			plotter=partial(plot.errorbar, fmt="x-", linewidth=0.5, ms=5, alpha=0.5)
			#colorscale=colorscales["rank"],
			#color_label="Rank"
		)
		axs[0].set_xscale("log")

	if flags["plot-maximal-epsilons"]:
		fig, ax = plot.square(data=maximal_epsilons,
			x="beta", x_label=labels["beta"],
			y="epsilon", y_label=labels["maximal_epsilon"],
			group=["order"],
			color="order",
			colorscale=colorscales["order"],
			color_label="Order",
			plotter=partial(plot.errorbar, fmt="D-", linewidth=0.5, ms=5, alpha=0.5)
		)
		ax.set_xscale("log")
		ax.set_yscale("log", base=2)

		fig, ax = plot.square(data=maximal_epsilons,
			x="n_dimensions", x_label="Dimensions",
			y="epsilon", y_label=labels["maximal_epsilon"],
			group=["order", "beta"],
			color="order",
			colorscale=colorscales["order"],
			color_label="Order",
			plotter=partial(plot.errorbar, fmt="D-", linewidth=0.5, ms=5, alpha=0.5)
		)
		ax.set_xscale("log", base=2)
		ax.set_yscale("log", base=2)

		fig, ax = plot.square(data=maximal_epsilons,
			x="relative_rank", x_label="Relative rank",
			y="epsilon", y_label=labels["maximal_epsilon"],
			group=["order", "beta"],
			color="order",
			colorscale=colorscales["order"],
			color_label="Order",
			plotter=partial(plot.errorbar, fmt="D-", linewidth=0.5, ms=5, alpha=0.5)
		)
		ax.set_xscale("log", base=2)
		ax.set_yscale("log", base=2)

	if flags["plot-optimal-convergence-times"]:
		fig, ax = plot.square(data=maximal_epsilons,
			x="beta", x_label=labels["beta"],
			y="step", y_label=labels["optimal_convergence_time"],
			group=["order", "n_dimensions", "relative_rank"],
			color="order",
			colorscale=colorscales["order"],
			color_label="Order",
			plotter=partial(plot.errorbar, fmt="v-", linewidth=0.5, ms=5, alpha=0.5)
		)
		ax.set_xscale("log")
		ax.set_yscale("log", base=2)
		ax.set_ylim(2**5, 2**7)

		fig, ax = plot.square(data=maximal_epsilons,
			x="n_dimensions", x_label="Dimensions",
			y="step", y_label=labels["optimal_convergence_time"],
			group=["order", "beta", "relative_rank"],
			color="order",
			colorscale=colorscales["order"],
			color_label="Order",
			plotter=partial(plot.errorbar, fmt="v-", linewidth=0.5, ms=5, alpha=0.5)
		)
		ax.set_xscale("log")
		ax.set_yscale("log", base=2)
		ax.set_ylim(2**5, 2**7)
