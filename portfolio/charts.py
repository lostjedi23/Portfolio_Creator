"""Plotly figures."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    return px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        labels=dict(color="Correlation"),
    )


def growth_chart(series: dict[str, pd.Series]) -> go.Figure:
    """Cumulative return lines. `series` maps a legend name to cumulative returns."""
    fig = go.Figure()
    for name, values in series.items():
        fig.add_trace(go.Scatter(x=values.index, y=values, mode="lines", name=name))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Cumulative return",
        yaxis_tickformat=".0%",
        hovermode="x unified",
    )
    return fig


def drawdown_chart(series: dict[str, pd.Series]) -> go.Figure:
    fig = go.Figure()
    for name, values in series.items():
        fig.add_trace(go.Scatter(x=values.index, y=values, mode="lines", name=name))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        hovermode="x unified",
    )
    return fig


def frontier_chart(
    frontier: pd.DataFrame,
    assets: pd.DataFrame,
    markers: dict[str, tuple[float, float]],
) -> go.Figure:
    """Efficient frontier line, individual assets, and labeled portfolios.

    `assets` has Return and Volatility columns indexed by ticker.
    `markers` maps a label to (volatility, return).
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frontier["Volatility"], y=frontier["Return"], mode="lines", name="Efficient frontier"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=assets["Volatility"],
            y=assets["Return"],
            mode="markers+text",
            text=list(assets.index),
            textposition="top center",
            name="Assets",
        )
    )
    for label, (vol, ret) in markers.items():
        fig.add_trace(
            go.Scatter(
                x=[vol],
                y=[ret],
                mode="markers",
                marker=dict(size=12, symbol="diamond"),
                name=label,
            )
        )
    fig.update_layout(
        xaxis_title="Annualized volatility",
        yaxis_title="Annualized return (CAGR)",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
    )
    return fig
