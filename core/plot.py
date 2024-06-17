import warnings

from typing import Protocol

import numpy as np
import matplotlib as mpl
import matplotlib.typing as mplt
import matplotlib.pyplot as plt
import pandas as pd

from .types import *

"""Types"""

class Plotter(Protocol):
	def __call__(
		self,
		axes: mpl.axes.Axes,
		data: pd.DataFrame,
		x: str,
		y: str,
		color: mplt.ColorType
	): ...

"""Color map"""

def log_colorscale(min, max, cmap):
	return mpl.cm.ScalarMappable(
		norm=mpl.colors.LogNorm(vmin=min, vmax=max),
		cmap=cmap
	)

def linear_colorscale(min, max, cmap):
	return mpl.cm.ScalarMappable(
		norm=mpl.colors.Normalize(vmin=min, vmax=max),
		cmap=cmap
	)

"""Plotters"""

def fill_between(
	axes: mpl.axes.Axes,
	data: pd.DataFrame,
	x: str,
	y: str,
	color: mplt.ColorType,
	alpha: float = 0.2
):
	axes.fill_between(
		data[x]["mean"],
		data[y]["min"],
		data[y]["max"],
		alpha=alpha, ec="none", fc=color
	)
	axes.plot(
		data[x]["mean"],
		data[y]["mean"],
		c=color,
		linewidth=0.5
	)

def scatter(
	axes: mpl.axes.Axes,
	data: pd.DataFrame,
	x: str,
	y: str,
	color: mplt.ColorType,
	s: int = 5,
	alpha: float = 0.5
):
	axes.scatter(
		data[x]["mean"],
		data[y]["mean"],
		color=color,
		s=s,
		alpha=alpha
	)

def errorbar(
	axes: mpl.axes.Axes,
	data: pd.DataFrame,
	x: str,
	y: str,
	color: mplt.ColorType,
	fmt: str = "o",
	ms: float = 5,
	alpha: float = 1,
	linewidth: float = 1
):
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")

		axes.errorbar(
			data[x]["mean"],
			data[y]["mean"],
			color=color,
			xerr=[(data[x]["mean"] - data[x]["min"]).clip(lower=0), (data[x]["max"] - data[x]["mean"]).clip(lower=0)],
			yerr=[(data[y]["mean"] - data[y]["min"]).clip(lower=0), (data[y]["max"] - data[y]["mean"]).clip(lower=0)],
			fmt=fmt,
			ms=ms,
			alpha=alpha,
			linewidth=linewidth
		)

"""Axes level"""

def single(
	axes: mpl.axes.Axes,
	data: pd.DataFrame,
	x: str,
	y: str,
	color: mplt.ColorType,
	plotter: Plotter,
	index: Optional[str] = None,
):
	if index is None:
		index = x

	for c in [x, y, index]:
		assert pd.api.types.is_numeric_dtype(data[c]), f'column {c} has invalid type {data[c].dtype}'

	data = data[~pd.isna(data[y])]

	aggregated = data.groupby(index)[[x, y]].agg(["mean", "min", "max"])

	plotter(axes, aggregated, x, y, color)

def grouped(axes: mpl.axes.Axes,
	data: pd.DataFrame,
	x: str,
	y: str,
	group: str | List[str],
	color: str,
	plotter: Plotter,
	colorscale: Optional[mpl.cm.ScalarMappable] = None,
	index: Optional[str] = None
):
	"""
	if color is None:
		color = group
	if colorscale is None:
		colorscale = linear_colorscale(data[color].min(), data[color].max(), "cool")
	"""

	if isinstance(group, str):
		group = [group]

	if not colorscale is None:
		assert isinstance(color, str), f'Color must be column name when no colorscale provided'
		assert color in group, f'Color column ({color}) not amongst group columns ({group})'

	for c in group:
		assert pd.api.types.is_numeric_dtype(data[c]), f'column {c} has invalid type {data[c].dtype}'

	grouped = data.groupby(group, as_index=False)

	for name, subdata in grouped:
		single(
			axes=axes,
			data=subdata,
			x=x,
			y=y,
			color=color if colorscale is None else cast(mplt.ColorType, colorscale.to_rgba(subdata[color].iloc[0])),
			plotter=plotter,
			index=index
		)

"""Figure level"""

def square(data: pd.DataFrame,
	x: str, x_label: str,
	y: str, y_label: str,
	plotter: Plotter = fill_between,
	index: Optional[str] = None,
	group: Optional[str | List[str]] = None, color_label: Optional[str] = None,
	color: str = "black", color_ticks: Optional[Sequence[int | float]] = None,
	colorscale: Optional[mpl.cm.ScalarMappable] = None,
) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
	fig, ax = plt.subplots()

	if group is None:
		single(ax, index=index, x=x, y=y, color=color, data=data, plotter=plotter)
	else:
		grouped(ax, index=index, x=x, y=y, group=group, color=color, colorscale=colorscale, data=data, plotter=plotter)
	
	ax.set_ylabel(y_label)
	ax.set_xlabel(x_label)

	if not group is None and not colorscale is None:
		colorbar = fig.colorbar(colorscale, ax=ax, shrink=0.6, location="bottom")
		if color_label is not None:
			colorbar.set_label(color_label)
		color_values = data[color].unique()
		#colorbar.ax.ticklabel_format(style="sci")
		#colorbar.ax.set_xticks([], minor=True)
		if color_ticks is not None:
			colorbar.ax.set_xticks(color_ticks)
		#else:
		#	colorbar.ax.set_xticks(color_values)


	fig.show()

	return fig, ax

def column(data: pd.DataFrame,
	x: str, x_label: str,
	y: str, y_label: str,
	row: str,
	plotter: Plotter = fill_between,
	index: Optional[str] = None,
	group: Optional[str | List[str]] = None, 
	color: str = "black",
	color_label: Optional[str] = None,
	color_ticks: Optional[Sequence[int | float]] = None,
	colorscale: Optional[mpl.cm.ScalarMappable] = None
):
	row_values = np.sort(data[row].unique())

	fig, axs = plt.subplots(len(row_values), 1, layout="constrained", sharex=True, sharey=True)

	for i_row in range(len(row_values)):
		ax = axs[i_row]

		row_value = row_values[i_row]
		subdata = data.loc[data[row] == row_value]

		if group is None:
			single(ax, index=index, x=x, y=y, color=color, data=subdata, plotter=plotter)
		else:
			grouped(ax, index=index, x=x, y=y, group=group, color=color, colorscale=colorscale, data=subdata, plotter=plotter)

		ax.set_ylabel(y_label)

		if i_row == len(row_values) - 1:
			ax.set_xlabel(x_label)

	if not group is None and not colorscale is None:
		colorbar = fig.colorbar(colorscale, ax=axs, shrink=0.8, location="bottom")
		if color_label is not None:
			colorbar.set_label(color_label)
		color_values = data[color].unique()
		#colorbar.ax.ticklabel_format(style="sci")
		#colorbar.ax.set_xticks([], minor=True)
		if color_ticks is not None:
			colorbar.ax.set_xticks(color_ticks)
		#else:
		#	colorbar.ax.set_xticks(color_values)


	fig.show()

	return fig, axs

def grid(data: pd.DataFrame,
	x: str, x_label: str,
	y: str, y_label: str,
	column: str,
	row: str,
	plotter: Plotter = fill_between,
	index: Optional[str] = None,
	group: Optional[str | List[str]] = None, 
	color: str = "black",
	color_label: Optional[str] = None,
	color_ticks: Optional[Sequence[int | float]] = None,
	colorscale: Optional[mpl.cm.ScalarMappable] = None,
	share_x: bool | Literal["row"] | Literal["col"] = "col",
	share_y: bool | Literal["row"] | Literal["col"] = "row"
):
	column_values = np.sort(data[column].unique())
	row_values = np.sort(data[row].unique())

	fig, axs = plt.subplots(len(row_values), len(column_values), layout="constrained", sharex=share_x, sharey=share_y)

	for i_row in range(len(row_values)):
		for i_column in range(len(column_values)):
			if len(row_values) == 1:
				ax = axs[i_column]
			elif len(column_values) == 1:
				ax = axs[i_row]
			else:
				ax = axs[i_row, i_column]
			column_value = column_values[i_column]
			row_value = row_values[i_row]
			subdata = data.loc[(data[column] == column_value) & (data[row] == row_value)]

			if group is None:
				single(ax, index=index, x=x, y=y, color=color, data=subdata, plotter=plotter)
			else:
				grouped(ax, index=index, x=x, y=y, group=group, color=color, colorscale=colorscale, data=subdata, plotter=plotter)
			
			#ax.set_title(f'{column_label} = {column_value}, {row_label} = {row_value}')

			if i_column == 0:
				ax.set_ylabel(y_label)
			if i_row == len(row_values) - 1:
				ax.set_xlabel(x_label)

	if not group is None and not colorscale is None:
		colorbar = fig.colorbar(colorscale, ax=axs, shrink=0.8, location="bottom")
		if color_label is not None:
			colorbar.set_label(color_label)
		color_values = data[color].unique()
		#colorbar.ax.ticklabel_format(style="sci")
		#colorbar.ax.set_xticks([], minor=True)
		if color_ticks is not None:
			colorbar.ax.set_xticks(color_ticks)
		#else:
		#	colorbar.ax.set_xticks(color_values)


	fig.show()

	return fig, axs
