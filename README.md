# Hyperparameter tuning for SGLD-based LLC estimation

This codebase is unpolished; contact the author if there are any issues.

The `core` module can be directly transplanted into another codebase or else used as a reference, and contains utilities for the following.

1. `algorithms.py`: LD, SGLD and radial traces.
2. `operations.py`: DLN and MLP implementations, MH acceptance probabilities and some other primitives.
3. `plot.py`: A visualisation layer used for all plots in the thesis.

See the `experiments` folder for example usages.

The implementation uses [JAX](https://jax.readthedocs.io/en/latest/). Model weights need to be converted into [pytrees](https://jax.readthedocs.io/en/latest/working-with-pytrees.html). Even if you are not using JAX, any Python class can be [registered as a pytree](https://jax.readthedocs.io/en/latest/working-with-pytrees.html#custom-pytree-nodes).

## Run

```
pip install pipenv
pipenv install
pipenv shell
python experiments/<experiment>.py
```

If you are on a Mac and want to use the GPU with JAX, refer to the [official instructions](https://developer.apple.com/metal/jax/).

## Citation

```
@masterthesis{xu2024mastersthesis,
  title        = {Hyperparameter tuning for SGLD-based LLC estimation},
  author       = {Adrian K. Xu},
  year         = 2024,
  month        = {June},
  school       = {University of Melbourne},
  type         = {Master's thesis}
}
```
