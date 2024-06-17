from typing import (
	Any,
	cast,
	Callable,
	Dict,
	Generator,
	List,
	Literal,
	Optional,
	Self,
	Sequence,
	TypeAlias,
	Tuple,
	TypeVar,
	Protocol
)

import jax
import numpy as np
import pandas as pd

Array = np.ndarray | jax.Array

W = TypeVar("W", contravariant=True)

Batcher = Generator[Tuple[Array, Array], None, None]

class Model(Protocol[W]):
	def __call__(self, weight: W, inputs: Array) -> jax.Array: ...

class StochasticLossFunction(Protocol[W]):
	def __call__(self, weight: W, inputs: Array, targets: Array) -> jax.Array: ...

class LossFunction(Protocol[W]):
	def __call__(self, weight: W) -> jax.Array: ...

DLNWeight: TypeAlias = Sequence[Array]
MLPWeight: TypeAlias = Sequence[Tuple[Array, Array]]
