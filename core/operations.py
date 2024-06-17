from typing import Callable, Tuple

import os
import time as ti

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from jax.scipy.special import logsumexp

from .types import *

rng = np.random.default_rng(12345)

# Pytree utilities

def pack(w):
	if w is jax.tree.leaves(w)[0]:
		packed = w
	else:
		flattened, _ = jax.tree_util.tree_flatten(w)
		packed = jnp.concatenate([jnp.reshape(a, -1) for a in flattened])

	return packed

def reversible_pack(w):
	if w is jax.tree.leaves(w)[0]:
		packed = w
		descriptor = None
	else:
		flattened, tree_def = jax.tree_util.tree_flatten(w)
		packed = jnp.concatenate([jnp.reshape(a, -1) for a in flattened])
		shapes = [np.array(a.shape) for a in flattened]
		descriptor = (shapes, tree_def)

	return packed, descriptor

def unpack(array, descriptor):
	if descriptor is None:
		return array
	shapes, tree_def = descriptor

	indices = np.cumsum([shape.prod() for shape in shapes])[:-1]

	leaves = [flat.reshape(shapes[i]) for i, flat in enumerate(np.split(array, indices))]

	tree = jax.tree_util.tree_unflatten(tree_def, leaves)

	return tree

# Models

def initialise_xavier(
	n_input_dimensions: int,
	n_output_dimensions: int
) -> np.ndarray:
	bound = 1 / np.sqrt(n_input_dimensions)
	return rng.uniform(-bound, bound, size=[n_input_dimensions,  n_output_dimensions])
	
def initialise_kaiming(
	n_input_dimensions: int,
	n_output_dimensions: int
) -> np.ndarray:
	return rng.normal(0, np.sqrt(2 / n_input_dimensions), size=(n_input_dimensions, n_output_dimensions))	

def dln(
	weight: DLNWeight,
	inputs: Array
) -> Array:
	"""Deep linear network.

	Weight should be in the format [w_1, ..., w_n]
	"""

	outputs = inputs

	for i_layer in range(len(weight) - 1):
		outputs = jnp.matmul(outputs, weight[i_layer])
	return jnp.matmul(outputs, weight[len(weight) - 1])

def initialise_dln(
	dimensions: List[int],
	method: str = "xavier"
) -> DLNWeight:
	assert method in ["zeros", "xavier", "kaiming"]

	initialise_fn = {
		"zeros": lambda n1, n2 : np.zeros([n1, n2]),
		"xavier": initialise_xavier,
		"kaiming": initialise_kaiming
	}[method]

	return [
		initialise_fn(dimensions[i], dimensions[i + 1])
		for i in range(len(dimensions) - 1)
	]

def mlp(
	weight: MLPWeight,
	inputs: Array
) -> Array:
	"""Multilayer perception.

	Weight should be in the format [(w_1, b_1), ..., (w_n, b_n)]
	"""

	outputs = inputs

	for i_layer in range(len(weight) - 1):
		outputs = jnp.matmul(outputs, weight[i_layer][0]) + weight[i_layer][1]
		outputs = jnp.maximum(outputs, 0)

	return jnp.matmul(outputs, weight[len(weight) - 1][0]) + weight[len(weight) - 1][1]

def initialise_mlp(
	dimensions: List[int],
	method: str = "kaiming",
	bias: float = 0
) -> MLPWeight:
	assert method in ["zeros", "xavier", "kaiming"]

	initialise_fn = {
		"zeros": lambda n1, n2 : np.zeros([n1, n2]),
		"xavier": initialise_xavier,
		"kaiming": initialise_kaiming
	}[method]

	return [
		(initialise_fn(dimensions[i], dimensions[i + 1]), bias * np.ones(dimensions[i + 1]))
		for i in range(len(dimensions) - 1)
	]

# Metrics

def regression_loss(
	model: Model[W],
	weight: W,
	inputs: Array,
	targets: Array
) -> Array:
	outputs = model(weight, inputs)

	ls = 0.5 * jnp.sum((outputs - targets) ** 2, axis=-1)

	return jnp.mean(ls)

def step_mala_deterministic(current, gradient, epsilon):
	return jax.tree_map(lambda w, g : w - 0.5 * epsilon * g, current, gradient)

def step_local_mala_deterministic(current, origin, gradient, epsilon, gamma, beta=1):
	return jax.tree_map(lambda w, w0, g : w - 0.5 * epsilon * (beta * g + gamma * (w - w0)), current, origin, gradient)

def step_mala(current, gradient, epsilon, beta=1):
	return step_noise(step_mala_deterministic(current, gradient, epsilon=epsilon, beta=beta), epsilon)

def step_local_mala(current, origin, gradient, epsilon, gamma, beta=1):
	return step_noise(step_local_mala_deterministic(current, origin, gradient, epsilon=epsilon, gamma=gamma, beta=beta), epsilon)

def step_noise(weight, variance):
	return jax.tree_map(
		lambda w : w + rng.normal(0, np.sqrt(variance), size=w.shape),
		weight
	)

def acceptance_probability(current_log_density, proposal_log_density, forward_log_density, backward_log_density):
	return np.minimum(
		np.exp((proposal_log_density - current_log_density) + (backward_log_density - forward_log_density)),
		1
	)

def mala_acceptance_probability(current, proposal, loss_and_grad_fn, epsilon):
	l_current, g_forward = loss_and_grad_fn(current)
	l_proposal, g_backward = loss_and_grad_fn(proposal)

	forward = step_mala_deterministic(current, g_forward, epsilon)
	backward = step_mala_deterministic(proposal, g_backward, epsilon)

	current_log_density = -1.0 * l_current
	proposal_log_density = -1.0 * l_proposal
	forward_log_density = -1.0 * np.sum((pack(proposal) - pack(forward)) ** 2) / (2 * epsilon)
	backward_log_density = -1.0 * np.sum((pack(current) - pack(backward))** 2) / (2 * epsilon)

	p = acceptance_probability(
		current_log_density=current_log_density,
		proposal_log_density=proposal_log_density,
		forward_log_density=forward_log_density,
		backward_log_density=backward_log_density
	)

	return p

# Primitives

def add(x1, x2):
	return jax.tree_map(lambda x1i, x2i : x1i + x2i, x1, x2)	

def subtract(x1, x2):
	return jax.tree_map(lambda x1i, x2i : x1i - x2i, x1, x2)

def scalar_multiply(x, a):
	return jax.tree_map(lambda xi : xi * a, x)

def scalar_divide(x, a):
	return jax.tree_map(lambda xi : xi / a, x)

def norm(x):
	return jnp.linalg.norm(pack(x))

def distance(x1, x2):
	return jnp.linalg.norm(pack(x1) - pack(x2))

def radial_projection(current, origin, radius):
	displacement = subtract(current, origin)
	return add(origin, scalar_multiply(displacement, radius / norm(displacement)))

def add_gaussian(x, key, standard_deviation=1):
	return jax.tree_map(lambda xi : xi + standard_deviation * jax.random.normal(key, shape=xi.shape), x)

# True learning coefficients

def dln_rank(weight: DLNWeight) -> int:
	matrix = weight[0]
	for layer in weight[1:]:
		matrix = np.matmul(matrix, layer)
	return np.linalg.matrix_rank(matrix)

def dln_learning_coefficient(
	weight: DLNWeight,
	verbose: bool = False
) -> float:
	"""The following implementation is recycled from https://github.com/edmundlth/validating_lambdahat/.
	"""

	true_rank = dln_rank(weight)
	input_dim = weight[0].shape[0]
	layer_widths = [layer.shape[1] for layer in weight]

	def _condition(indices, intlist, verbose=False):
	    intlist = np.array(intlist)
	    ell = len(indices) - 1
	    subset = intlist[indices]
	    complement = intlist[[i for i in range(len(intlist)) if i not in indices]]
	    has_complement = len(complement) > 0
	    # print(indices, subset, complement)
	    if has_complement and not (np.max(subset) < np.min(complement)):
	        if verbose: print(f"max(subset) = {np.max(subset)}, min(complement) = {np.min(complement)}")
	        return False
	    if not (np.sum(subset) >= ell * np.max(subset)):
	        if verbose: print(f"sum(subset) = {sum(subset)}, ell * max(subset) = {ell * np.max(subset)}")
	        return False
	    if has_complement and not (np.sum(subset) < ell * np.min(complement)):
	        if verbose: print(f"sum(subset) = {sum(subset)}, ell * min(complement) = {ell * np.min(complement)}")
	        return False
	    return True

	def _search_subset(intlist):
		def generate_candidate_indices(intlist):
			argsort_indices = np.argsort(intlist)
			for i in range(1, len(intlist) + 1):
				yield argsort_indices[:i]
		for indices in generate_candidate_indices(intlist):
			if _condition(indices, intlist):
				return indices
		raise RuntimeError("No subset found")

	M_list = np.array([input_dim] + list(layer_widths)) - true_rank
	indices = _search_subset(M_list)
	M_subset = M_list[indices]
	if verbose:
		print(f"M_list: {M_list}, indices: {indices}, M_subset: {M_subset}")
	M_subset_sum = np.sum(M_subset)
	ell = len(M_subset) - 1
	M = np.ceil(M_subset_sum / ell)
	a = M_subset_sum - (M - 1) * ell
	output_dim = layer_widths[-1]

	term1 = (-true_rank**2 + true_rank * (output_dim + input_dim)) / 2
	term2 = a * (ell - a) / (4 * ell)
	term3 = -ell * (ell - 1) / 4 * (M_subset_sum / ell)**2
	term4 = 1 / 2 * np.sum([M_subset[i] * M_subset[j] for i in range(ell + 1) for j in range(i + 1, ell + 1)])
	learning_coefficient = term1 + term2 + term3 + term4
	
	#multiplicity = a * (ell - a) + 1

	return learning_coefficient
