import pandas as pd
import matplotlib.pyplot as plt
import psutil
from typing import Optional
from scipy.stats import probplot, chi2_contingency


def transform_category_str(category: str):
    """ "Transforms a category string"""
    category = category.split("-")[0]
    religions = [
        "christianity",
        "islam",
        "hinduism",
        "buddhism",
        "sikhism",
        "judaism",
        "taoism",
        "confucianism",
        "shinto",
        "zoroastrianism",
        "spirituality",
        "religion",
    ]
    if category in religions:
        category = "spirituality-religion"
    if category == "true":
        category = "true-crime"

    return category


def transform_category_list(cat_list: list, leave_set=True):
    """Transforms a list of categories into a aggregated list"""
    cat_list = [transform_category_str(cat) for cat in cat_list]

    # remove duplicates
    if leave_set:
        cat_list = set(cat_list)
    else:
        cat_list = list(set(cat_list))
    return cat_list


def axis_titles(
    ax: plt.Axes,
    xtitle: Optional[str] = None,
    ytitle: Optional[str] = None,
    title: Optional[str] = None,
):
    """Sets the labels of axes and the graph name"""
    ax.set_ylabel(ytitle)
    ax.set_xlabel(xtitle)
    ax.set_title(title)


def col_frequency_table(df, column_name, index_name=None):
    """Returns a clean frequency table"""

    # Calculate the value counts of the specified column
    counts = df[column_name].value_counts().to_frame()

    # Set the index name
    if index_name is None:
        index_name = column_name
    counts.index.name = index_name

    # Rename the column
    counts.rename(columns={column_name: "Frequency"}, inplace=True)

    # Transpose the table
    transposed_table = counts.T

    return transposed_table


def memory_used():
    """Returns memory used in MB"""
    mem = psutil.Process().memory_full_info().uss / (1024**2)
    print(f"Memory used: {mem:.0f} MB")


def chi2_test(col1, col2, return_results=False, print_out=True):
    """Returns the chi-squared statistic values"""
    contingency_table = pd.crosstab(col1, col2)
    chi2_stat, p_val, dof, expected_freq = chi2_contingency(contingency_table)

    if print_out:
        print("Chi-square test statistic:", round(chi2_stat, 2))
        print("p-value:", "{:.2e}".format(p_val))

    if return_results:
        return {
            "chi": chi2_stat,
            "p": p_val,
            "dof": dof,
            "freq": expected_freq,
            "con_table": contingency_table,
        }


def pol_points(ratings, freq):
    """Calculates rating polarization points"""
    points = 0
    for rating, fre in zip(ratings, freq):
        if rating in [1, 5]:
            points += 2 * fre
        if rating in [2, 4]:
            points += 1 * fre
    points = points / freq.sum()
    return points


def default_plotly_margins(fig, l=65, r=10, t=25, b=30):
    fig.update_layout(
        margin=dict(l=l, r=r, t=t, b=b),
    )
