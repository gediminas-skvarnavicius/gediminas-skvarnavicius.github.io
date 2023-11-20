# pylint: disable=import-error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Sequence, Union
import numpy as np
import numpy.typing as npt
from datetime import datetime
import pandas as pd


def two_yaxis_plotly(
    x_values: Sequence[float],
    y1_values: Sequence[float],
    y2_values: Sequence[float],
    y1_title: str,
    y2_title: str,
    x_title: str,
    colors: Sequence[str],
    size: Sequence[str],
    yrange: Optional[Sequence[float]],
    **kwargs,
) -> go.Figure:
    """Creates a graph with two separate lines sharing the x axis but having different y axes."""
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=x_values, y=y1_values, name=y1_title, line=dict(color=colors[0])),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=x_values, y=y2_values, name=y2_title, line=dict(color=colors[1])),
        secondary_y=True,
    )

    # Set x-axis title
    fig.update_xaxes(title_text=x_title)

    # Set y-axes titles
    fig.update_yaxes(
        title_text=f"<b>{y1_title}</b>", secondary_y=False, patch=dict(color=colors[0])
    )
    fig.update_yaxes(
        title_text=f"<b>{y2_title}</b>", secondary_y=True, patch=dict(color=colors[1])
    )

    fig.update_layout(
        width=size[0], height=size[1], legend=dict(orientation="h", y=1.15), **kwargs
    )

    if yrange:
        fig.update_layout(yaxis=dict(range=yrange))

    return fig


def add_alert_background(
    fig: go.Figure,
    orange: bool = True,
    red: bool = True,
    blue: bool = False,
    yellow: bool = False,
) -> None:
    """Adds shapes to a plotly figure that correspond to infectious disease alert levels"""
    if red:
        fig.add_shape(
            type="rect",
            x0="2020-02-23",  # start x-value of rectangle
            x1="2020-07-05",  # end x-value of rectangle
            y0=fig.layout.yaxis.range[0],  # start y-value of rectangle
            y1=fig.layout.yaxis.range[1],  # end y-value of rectangle
            fillcolor="rgba(255, 0, 0, 0.2)",  # fill color with opacity
            line=dict(width=0),  # set the border width to 0 to remove the border
        )

    if orange:
        fig.add_shape(
            type="rect",
            x0="2020-01-28",  # start x-value of rectangle
            x1="2020-02-23",  # end x-value of rectangle
            y0=fig.layout.yaxis.range[0],  # start y-value of rectangle
            y1=fig.layout.yaxis.range[1],  # end y-value of rectangle
            fillcolor="rgba(255, 155, 0, 0.2)",  # fill color with opacity
            line=dict(width=0),  # set the border width to 0 to remove the border
        )
    if blue:
        fig.add_shape(
            type="rect",
            x0="2020-01-03",  # start x-value of rectangle
            x1="2020-01-20",  # end x-value of rectangle
            y0=fig.layout.yaxis.range[0],  # start y-value of rectangle
            y1=fig.layout.yaxis.range[1],  # end y-value of rectangle
            fillcolor="rgba(0, 0, 255, 0.2)",  # fill color with opacity
            line=dict(width=0),  # set the border width to 0 to remove the border
        )
    if yellow:
        fig.add_shape(
            type="rect",
            x0="2020-01-20",  # start x-value of rectangle
            x1="2020-01-28",  # end x-value of rectangle
            y0=fig.layout.yaxis.range[0],  # start y-value of rectangle
            y1=fig.layout.yaxis.range[1],  # end y-value of rectangle
            fillcolor="rgba(255, 255, 0, 0.2)",  # fill color with opacity
            line=dict(width=0),  # set the border width to 0 to remove the border
        )


def closest_point_plotly(
    fig: go.Figure,
    value: Union[float, datetime],
    val_to_get: str = "y",
    data_no: int = 0,
) -> float:
    """Gets the closest x or y value to x or y argument"""
    assert val_to_get in ["x", "y"], "val to get must be x or y"
    if val_to_get == "y":
        x_vals: npt.NDArray = fig.data[data_no].x
        closest_xi: Union[float, datetime] = np.abs(x_vals - value).argmin()
        closest_point = fig.data[0].y[closest_xi]
    if val_to_get == "x":
        y_vals: npt.NDArray = fig.data[data_no].y
        closest_yi: Union[float, datetime] = np.abs(y_vals - value).argmin()
        closest_point = fig.data[0].y[closest_yi]
    return closest_point


def annotate_plotly_by_val(
    fig: go.Figure,
    value: Union[float, datetime],
    text: str,
    val_ax: str = "x",
    data_no: int = 0,
    **kwargs,
) -> None:
    """Annotates the given figure by single x or y value"""
    assert val_ax in ["x", "y"], "val to get must be x or y"
    if val_ax == "x":
        y_val = closest_point_plotly(fig, value, val_to_get="y", data_no=data_no)
        fig.add_annotation(
            x=value,  # x-coordinate of text position
            y=y_val,  # y-coordinate of text position
            xref="x",  # x-coordinate reference
            yref="y",  # y-coordinate reference
            text=text,  # text to display
            **kwargs,
        )
    if val_ax == "y":
        x_val = closest_point_plotly(fig, value, val_to_get="x", data_no=data_no)
        fig.add_annotation(
            x=x_val,  # x-coordinate of text position
            y=value,  # y-coordinate of text position
            xref="x",  # x-coordinate reference
            yref="y",  # y-coordinate reference
            text=text,  # text to display
            **kwargs,
        )


def get_correlation_pairs(
    data: pd.DataFrame,
    positive_cut_off: Optional[float] = None,
    negative_cut_off: Optional[float] = None,
    leave_center: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Produces a data frame that contains pairs of features
    and their r-values based on selected cut-offs
    """
    if positive_cut_off is not None and not 0 <= positive_cut_off <= 1:
        raise ValueError("Positive cut-offs must be between 0 and 1")
    if negative_cut_off is not None and not -1 <= negative_cut_off <= 0:
        raise ValueError("Negative cut-offs must be between -1 and 0")
    corr_matrix = data.corr(numeric_only=True, **kwargs)
    if not leave_center:
        assert positive_cut_off is None or negative_cut_off is None, (
            "one sided cut-off requires only one positive or negative cut-off value, "
            "use leave_center=True to cut off both ends"
        )
        np.fill_diagonal(corr_matrix.values, None)
        cut_correlation_matrix = corr_matrix.mask(corr_matrix <= positive_cut_off)
        cut_correlation_matrix = cut_correlation_matrix.mask(
            corr_matrix >= negative_cut_off
        )
        # filtering out values with r of less than neg, more than pos or 1
    else:
        cut_correlation_matrix = corr_matrix.mask(corr_matrix <= negative_cut_off)
        # filtering out values with r lower than -0.1
        cut_correlation_matrix = cut_correlation_matrix.mask(
            corr_matrix >= positive_cut_off
        )
        # filtering out values with r higher than 0.1
    cut_correlation_matrix = cut_correlation_matrix.stack().reset_index()
    # stacking the remaining values
    cut_correlation_matrix["feature_pair"] = cut_correlation_matrix[
        ["level_0", "level_1"]
    ].apply(frozenset, axis=1)
    # combining levels after stacking into a single pair frozenset
    cut_correlation_matrix = cut_correlation_matrix.drop(columns=["level_0", "level_1"])
    # dropping previous level columns
    cut_correlation_matrix = cut_correlation_matrix.drop_duplicates(
        subset="feature_pair"
    )
    # removing duplicate pairs
    cut_correlation_matrix = cut_correlation_matrix.rename(columns={0: "r-value"})
    return cut_correlation_matrix


def create_bins(
    data: Union[pd.Series, np.ndarray], number: int, log: bool = False
) -> np.ndarray:
    """Creates bins from min to max of specified data"""
    if not log:
        bins: np.ndarray = np.linspace(data.min(), data.max(), number)
    else:
        bins = np.logspace(np.log10(data.min()), np.log10(data.max()), number)
    return bins


def create_scatter_traces_sliding(dataframe, x_col, y_col, **kwargs):
    """Creates a list of plotly graph objects to be used as traces in a trend line plot with a slider"""
    scatter_objects = []
    for index, row in dataframe.iterrows():
        scatter_objects.append(
            go.Scatter(x=dataframe[x_col][:index], y=dataframe[y_col][:index], **kwargs)
        )
    return scatter_objects


def create_slider_steps(fig, num_lines: int, last_visible: bool = False):
    """Creates a list of sliders from a figure containing multiple traces to be scrolled over"""
    steps = []
    for i in range(int(len(fig.data) / num_lines)):
        visible = [False] * len(fig.data)
        data_points = int(len(fig.data) / num_lines)
        for j in range(0, num_lines):
            visible[i + j * data_points] = True
        if last_visible == True:
            visible[-1] = True  # last trace always visible
        step = dict(
            method="update",
            args=[{"visible": visible}],
            label=fig.data[i].customdata[i],
        )
        steps.append(step)
    return steps


def rename_describe_table(
    table: Union[pd.DataFrame, pd.Series],
    index_name: str = "Metric",
    value_name: str = "Value",
):
    """Renames the pandas df.describe() output table columns"""
    if isinstance(table, pd.Series):
        table = table.to_frame()
    table.reset_index(inplace=True)
    table = table.rename(
        columns={table.columns[0]: index_name, table.columns[1]: value_name}
    )
    return table


def drop_outliers_by_std(data: pd.DataFrame, columns: list[str], multiplier: float = 2):
    """
    Drops rows from a pd.DataFrame that have outlier values,
    based on difference from mean by std
    """
    raw_data = data
    rows_to_drop = np.array([], dtype=int)
    for column in columns:
        outliers = raw_data[
            ~raw_data[column].between(
                raw_data[column].mean() - raw_data[column].std() * multiplier,
                raw_data[column].mean() + raw_data[column].std() * multiplier,
            )
        ]
        rows_to_drop = np.append(rows_to_drop, outliers.index.values)
    rows_to_drop = np.unique(rows_to_drop)
    df_without_outliers = raw_data.drop(index=rows_to_drop)
    return df_without_outliers
