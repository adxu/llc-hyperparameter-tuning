import warnings

from functools import partial, reduce
from itertools import product
from typing import Callable, cast, Generator

from collections.abc import Iterable

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from . import operations

from .types import *

rng = np.random.default_rng(12345)

def stochastic_trace(
    loss_fn: StochasticLossFunction[W],
    batcher: Batcher,
    w1: W, w2: W,
    n_points: int = 128,
    crop: Optional[Tuple[int, int]] = None
) -> pd.DataFrame:
    if crop is None:
        crop = (0, n_points)

    points = np.arange(crop[0], crop[1])

    w1_packed, descriptor = operations.reversible_pack(w1)
    w2_packed = operations.pack(w2)
    ws = [ operations.unpack(packed, descriptor) for packed in np.linspace(w1_packed, w2_packed, n_points) ]

    ws = ws[crop[0]:crop[1]]

    return pd.DataFrame({
        "point": points,
        "loss": [ loss_fn(w, *next(batcher)).item() for i, w in enumerate(ws) ],
        "distance": np.linspace(
            0,
            np.linalg.norm(w2_packed - w1_packed),
            n_points
        )[crop[0]:crop[1]]
    })

def ld(
    loss_fn: LossFunction[W],
    initial: W,
    step_size: float,
    n_steps: int = 1000,
    checkpoint_interval: int = 100,
    metropolis_interval: int = 10,
    seed: int = 0
) -> pd.DataFrame:
    loss_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    random_keys = jax.random.split(jax.random.key(seed), num=n_steps)
    @jax.jit
    def update(current, i):
        loss, grad = loss_and_grad_fn(current)
        current = operations.subtract(current, operations.scalar_multiply(grad, step_size / 2))
        current = operations.add_gaussian(current, standard_deviation=np.sqrt(step_size), key=random_keys[i])
        return loss, current
    steps = []
    current = initial
    for i in range(n_steps):
        loss, proposed = update(current, i)

        if i % metropolis_interval == 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                p = operations.mala_acceptance_probability(
                    current=current, proposal=proposed,
                    loss_and_grad_fn=lambda w : loss_and_grad_fn(w),
                    epsilon=step_size
                )
        else:
            p = None

        if (i + 1) % checkpoint_interval == 0:
            weight = current
        else:
            weight = None

        current = proposed

        steps.append({
            "step": i,
            "loss": loss.item(),
            "acceptance_probability": p,
            "weight": weight
        })

        if np.isnan(loss):
            break

    return pd.DataFrame(steps)

def sgld(
    loss_fn: StochasticLossFunction[W],
    batcher: Batcher,
    initial: W,
    step_size: float,
    n_steps: int = 1000,
    checkpoint_interval: int = 1000,
    metropolis_interval: int = 100,
    seed: int = 0
) -> pd.DataFrame:
    loss_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    random_keys = jax.random.split(jax.random.key(seed), num=n_steps)
    @jax.jit
    def update(current, batch, i):
        loss, grad = loss_and_grad_fn(current, *batch)
        current = operations.subtract(current, operations.scalar_multiply(grad, step_size / 2))
        current = operations.add_gaussian(current, standard_deviation=np.sqrt(step_size), key=random_keys[i])
        return loss, current
    steps = []
    current = initial
    for i in range(n_steps):
        batch = next(batcher)
        loss, proposed = update(current, batch, i)
        if i % metropolis_interval == 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ps = []
                for j in range(8):
                    batch = next(batcher)
                    ps.append(operations.mala_acceptance_probability(
                        current=current, proposal=proposed,
                        loss_and_grad_fn=lambda w : loss_and_grad_fn(w, *batch),
                        epsilon=step_size
                    ))
                p = np.mean(ps)
        else:
            p = None

        if (i + 1) % checkpoint_interval == 0:
            weight = current
        else:
            weight = None

        current = proposed

        steps.append({
            "step": i,
            "loss": loss.item(),
            "acceptance_probability": p,
            "weight": weight
        })

        if np.isnan(loss):
            break

    return pd.DataFrame(steps)
