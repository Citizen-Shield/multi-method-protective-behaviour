# -*- coding: utf-8 -*-
"""Submodule correlation.py includes the following functions:
   - correlation_analysis(): Run correlations for numerical features and return output in different formats
   - plot_correlogram(): Plot correlogram of numerical features. Rows with missing values are excluded
   - correlation_as_sample_increases(): Run correlations for subparts of the data to check robustness
"""
import logging
import warnings
from itertools import combinations
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols

from neuropy_frequentist_statistics import normal_check
from neuropy_frequentist_statistics import permutation_plot
from neuropy_frequentist_statistics import permute_test


def correlation_analysis(  # noqa: C901
    data: pd.DataFrame,
    col_list=None,
    row_list=None,
    check_norm=False,
    method: str = "pearson",
    dropna: str = "pairwise",
    permutation_test: bool = False,
    n_permutations: int = 1000,
    random_state=None,
    plot_permutation: bool = False,
    figsize: tuple = (11.7, 8.27),
):
    r"""Run correlations for numerical features and return output in different formats

    Different methods to compute correlations and to handle missing values are implemented.
    Inspired by `researchpy.corr_case` and `researchpy.corr_pair`.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe with variables in columns, cases in rows
    row_list: list or None (default: None)
        List with names of columns in `data` that should be in the rows of the correlogram.
        If None, all columns are used but only every unique combination.
    col_list: list or None (default: None)
        List with names of columns in `data` that should be in the columns of the correlogram.
        If None, all columns are used and only every unique combination.
    check_norm: bool (default: False)
        If True, normality will be checked for columns in `data` using `normal_check`. This influences the used method
        for correlations, i.e. Pearson or Spearman. Note: normality check ignores missing values.
    method: {'pearson', 'kendall', 'spearman'}, default 'pearson'
        Type of correlation, either Pearson's r, Spearman's rho, or Kendall's tau, implemented via respectively
        `scipy.stats.pearsonr`, `scipy.stats.spearmanr`, and `scipy.stats.kendalltau`
        Will be ignored if check_norm=True. Instead, Person's r is used for every combination of normally distributed
        columns and Spearman's rho is used for all other combinations.
    dropna : {'listwise', 'pairwise'}, default 'pairwise'
        Should rows with missing values be dropped over the complete `data` ('listwise') or for every correlation
        separately ('pairwise')
    permutation_test: bool (default: False)
        If true, a permutation test will added
    n_permutations: int (default: 1000)
        Number of permutations in the permutation test
    random_state: None or int (default: None)
        Random state for permutation_test. If not None, random_state will be updated for every permutation
    plot_permutation: bool (default: False)
        Whether to plot the results of the permutation test
    figsize: tuple (default: (11.7, 8.27))
        Width and height of the figure in inches

    Returns
    ----------
    result_dict: dict
        Dictionary containing with the following keys:
        info : pandas.DataFrame
            Description of correlation method, missing values handling and number of observations
        r-values : pandas.DataFrame
            Dataframe with correlation coefficients. Indices and columns are column names from `data`. Only lower
            triangle is filled.
        p-values : pandas.DataFrame
            Dataframe with p-values. Indices and columns are column names from `data`. Only lower triangle is filled.
        N        : pandas.DataFrame
            Dataframe with numbers of observations. Indices and columns are column names from `data`. Only lower
            triangle is filled. If dropna ='listwise', every correlation will have the same number of observations.
        summary : pandas.DataFrame
            Dataframe with columns ['analysis', 'feature1', 'feature2', 'r-value', 'p-value', 'N', 'stat-sign']
            which indicate the type of test used for the correlation, the pair of columns, the correlation coefficient,
            the p-value, the number of observations for each combination of columns in `data` and whether the r-value is
            statistically significant.
    plotted_permuations: Figure

    Examples
    ----------
    >>> from neuropy.correlation import correlation_analysis
    >>> import seaborn as sns
    >>> iris = sns.load_dataset('iris')
    >>> dict_results, fig_permutations = correlation_analysis(iris, method='pearson', dropna='listwise', permutation_test=True,
    >>>                                        n_permutations=100, check_norm=True)
    >>> dict_results['summary']

    References
    ----------
    Bryant, C (2018). researchpy's documentation [Revision 9ae5ed63]. Retrieved from
    https://researchpy.readthedocs.io/en/latest/

    """

    # Settings test
    if method == "pearson":
        test, test_name = stats.pearsonr, "Pearson"
    elif method == "spearman":
        test, test_name = stats.spearmanr, "Spearman Rank"
    elif method == "kendall":
        test, test_name = stats.kendalltau, "Kendall's Tau-b"
    else:
        raise ValueError("method not in {'pearson', 'kendall', 'spearman'}")

    # Copy numerical data from the original data
    data = data.copy().select_dtypes("number")
    plotted_permutations = None

    # Get correct lists
    if col_list and not row_list:
        row_list = data.select_dtypes("number").drop(col_list, axis=1).columns.tolist()
    elif row_list and not col_list:
        col_list = data.select_dtypes("number").drop(row_list, axis=1).columns.tolist()

    # Initializing dataframes to store results
    info = pd.DataFrame()
    summary = pd.DataFrame()
    if not col_list and not row_list:
        r_vals = pd.DataFrame(columns=data.columns, index=data.columns)
        p_vals = pd.DataFrame(columns=data.columns, index=data.columns)
        n_vals = pd.DataFrame(columns=data.columns, index=data.columns)
        iterator = combinations(data.columns, 2)
    else:
        r_vals = pd.DataFrame(columns=col_list, index=row_list)
        p_vals = pd.DataFrame(columns=col_list, index=row_list)
        n_vals = pd.DataFrame(columns=col_list, index=row_list)
        iterator = product(col_list, row_list)

    if dropna == "listwise":
        # Remove rows with missing values
        data = data.dropna(how="any", axis="index")
        info = info.append(
            {
                f"{test_name} correlation test using {dropna} deletion": f"Total observations used = {len(data)}"
            },
            ignore_index=True,
        )
    elif dropna == "pairwise":
        info = info.append(
            {
                f"{test_name} correlation test using {dropna} deletion": f"Observations in the data = {len(data)}"
            },
            ignore_index=True,
        )
    else:
        raise ValueError("dropna not in {'listwise', 'pairwise'}")

    if check_norm:
        # Check normality of all columns in the data
        df_normality = normal_check(data)
        norm_names = df_normality.loc[df_normality["normality"], "feature"].tolist()

    if (not permutation_test) and plot_permutation:
        warnings.warn(
            "Asked for plots but permutation=False so no permutation plot will be added"
        )

    # Iterating through the Pandas series and performing the correlation
    for col1, col2 in iterator:
        if dropna == "pairwise":
            # Remove rows with missing values in the pair of columns
            test_data = data[[col1, col2]].dropna()
        else:
            test_data = data

        if check_norm:
            # Select Pearson's r only if both columns are normally distributed
            if (col1 in norm_names) and (col2 in norm_names):
                test, test_name = stats.pearsonr, "Pearson"
            else:
                test, test_name = stats.spearmanr, "Spearman Rank"

        # Run correlations
        r_value, p_value = test(test_data.loc[:, col1], test_data.loc[:, col2])
        n_value = len(test_data)

        # Store output in matrix format
        try:
            r_vals.loc[col2, col1] = r_value
            p_vals.loc[col2, col1] = p_value
            n_vals.loc[col2, col1] = n_value
        except KeyError:
            r_vals.loc[col1, col2] = r_value
            p_vals.loc[col1, col2] = p_value
            n_vals.loc[col1, col2] = n_value

        # Store output in dataframe format
        dict_summary = {
            "analysis": test_name,
            "feature1": col1,
            "feature2": col2,
            "r-value": r_value,
            "p-value": p_value,
            "stat-sign": (p_value < 0.05),
            "N": n_value,
        }

        if permutation_test:
            # Copy the complete data
            col2_shuffle = np.array(test_data.loc[:, col2])
            col2_shuffle = np.repeat(
                col2_shuffle[:, np.newaxis], n_permutations, axis=1
            )
            # Shuffle within the columns
            np.random.seed(random_state)
            ix_i = np.random.sample(col2_shuffle.shape).argsort(axis=0)
            ix_j = np.tile(np.arange(col2_shuffle.shape[1]), (col2_shuffle.shape[0], 1))
            col2_shuffle = col2_shuffle[ix_i, ix_j]
            permutations = np.apply_along_axis(
                permute_test,
                axis=0,
                arr=col2_shuffle,
                test_type="correlation",
                test=test,
                a2=np.array(test_data.loc[:, col1]),
            )

            extreme_permutation = np.where(permutations < p_value, 1, 0)
            p_permutation = extreme_permutation.sum() / len(permutations)
            dict_summary["permutation-p-value"] = p_permutation

            if plot_permutation:
                plotted_permutations = permutation_plot(
                    permutations, p_value, feature1=col1, feature2=col2, figsize=figsize
                )

            # Reset random seed numpy
            np.random.seed(None)

        summary = pd.concat(
            [summary, pd.DataFrame(data=dict_summary, index=[0])],
            axis=0,
            ignore_index=True,
            sort=False,
        )

    # Embed results within a dictionary
    result_dict = {
        "r-value": r_vals,
        "p-value": p_vals,
        "N": n_vals,
        "info": info,
        "summary": summary,
    }

    return result_dict, plotted_permutations


def plot_correlogram(
    data: pd.DataFrame,
    row_list=None,
    col_list=None,
    check_norm=False,
    method="pearson",
    dropna="pairwise",
    margins=None,
    font_scale=1.2,
    show_p=True,
    figsize=(15, 15),
):
    r"""Plot correlogram of numerical features. Rows with missing values are excluded.

    Different methods to compute correlations are implemented.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe with variables in columns, cases in rows
    row_list: list or None (default: None)
        List with names of columns in `data` that should be in the rows of the correlogram.
        If None, all columns are used and only the lower half of the correlogram will be filled.
    col_list: list or None (default: None)
        List with names of columns in `data` that should be in the columns of the correlogram.
        If None, all columns are used and only the lower half of the correlogram will be filled.
    check_norm: bool (default: False)
        If True, normality will be checked for columns in `data` using `normal_check`. This influences the used method
        for correlations, `method` will be ignored. Note: normality check ignores missing values.
    method: {'pearson', 'kendall', 'spearman'}, default 'pearson'
        Type of correlation, either Pearson's r, Spearman's rho, or Kendall's tau, implemented via respectively
        `scipy.stats.pearsonr`, `scipy.stats.spearmanr`, and `scipy.stats.kendalltau`. Ignored if check_norm is True.
    dropna : {'listwise', 'pairwise'}, default 'pairwise'
        Should rows with missing values be dropped over the complete `data` ('listwise') or for every correlation
        separately ('pairwise')
    margins: dict or 'jupyter' or None (default: None)
        Margins for the correlogram. Any of them that are None are referred from `matplotlib.pyplot.subplots_adjust`
        If 'jupyter', default values are {'left': None, 'bottom': 1, 'right': None, 'top': 2}.
    font_scale: float
        Size of the labels in the correlogram.
    show_p: bool (default: True)
        Place crosses when correlation is not significant (i.e. p-value higher than 0.05).
    figsize: tuple (default: (15, 15))
        Width and height of the figure in inches.

    Returns
    ----------
    corplot: Figure
        Graph with `seaborn.heatmap` of the correlations (lower triangle only)

    Examples
    ----------
    >>> iris = sns.load_dataset('iris')
    >>> _ = plot_correlogram(iris, method='pearson')

    """

    # Compute correlation matrix
    dict_results, _ = correlation_analysis(
        data,
        col_list=col_list,
        row_list=row_list,
        check_norm=check_norm,
        method=method,
        dropna=dropna,
    )
    corr = dict_results["r-value"]
    corr = corr.astype("float64")

    if col_list or row_list:
        mask = None
    else:
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

    # Add the mask to the heatmap
    sns.set(font_scale=font_scale)
    corplot, ax = plt.subplots(figsize=figsize)
    cmap = sns.diverging_palette(h_neg=10, h_pos=240, as_cmap=True)
    _ = sns.heatmap(
        corr,
        ax=ax,
        mask=mask,
        cmap=cmap,
        center=0,
        linewidths=1,
        annot=True,
        fmt=".3f",
        vmin=-1,
        vmax=1,
    )

    if show_p:
        pvalues = dict_results["p-value"].values
        # Set X where pvalues is bigger than 0.05
        pvalues_str = np.where(pvalues < 0.05, "", "X")

        if col_list or row_list:
            # Run over all elements of the array
            iterator = np.ndindex(pvalues.shape)
        else:
            # Only take one half of the pvalues
            iterator = combinations(range(pvalues.shape[0]), 2)

        for y, x in iterator:
            if not col_list and not row_list:
                # Reverse x and y to make sure the crosses are plotted at the right places
                x, y = y, x
            _ = plt.text(
                x + 0.5,
                y + 0.5,
                pvalues_str[y, x],
                horizontalalignment="center",
                verticalalignment="center",
                color="gray",
                fontsize=font_scale * 30,
            )

    if check_norm:
        _ = ax.set_title("Correlation using Pearson and Spearman")
    else:
        _ = ax.set_title(f"{method.capitalize()} correlation")

    # Move axis to make sure they align
    ymax, ymin = plt.gca().get_ylim()
    _ = plt.gca().set_ylim(bottom=ymin - 0.5, top=ymax + 0.5)

    if not margins:
        margins = {"left": None, "bottom": None, "right": None, "top": None}
    elif margins == "jupyter":
        margins = {"left": None, "bottom": 1, "right": None, "top": 2}

    _ = plt.subplots_adjust(**margins)

    return corplot


def correlations_as_sample_increases(
    data: pd.DataFrame,
    feature1: str,
    feature2: str,
    starting_N: int = 10,
    step: int = 1,
    method="pearson",
    random_state=None,
    plot: bool = True,
    addition_to_title: str = "",
    alpha: float = 0.05,
):
    r"""Plot changes in r-value and p-value from correlation between two features when sample size increases.

    Different methods to compute correlations are implemented. Data is shuffled first, to prevent any order effects.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe with variables in columns, cases in rows
    feature1: str
        Name of column with first feature to be included in correlation
    feature2: str
        Name of column with second feature to be included in correlation
    starting_N: int (default: 10)
        Number of cases that should be used for first correlation
    step: int (default: 1)
        Step for increasing the number of cases for the correlations
    method: {'pearson', 'kendall', 'spearman'}, default 'pearson'
        Type of correlation, either Pearson's r, Spearman's rho, or Kendall's tau, implemented via respectively
        `scipy.stats.pearsonr`, `scipy.stats.spearmanr`, and `scipy.stats.kendalltau`.
    random_state: int or None (default: None)
        Random state for reordering the data
    plot: bool (default: True)
        Whether to plot the results
    addition_to_title: str (default: '')
        The title of the plot will be "The absolute r-value between {feature1} and {feature2} as N increases" and
        followed by the addition (e.g. to describe a dataset).
    alpha: float (default: 0.05)
        Threshold for p-value that should be shown in the plot

    Returns
    ----------
    cor_results: pd.DataFrame
        Dataframe with the results for all ran analyses
    fig: Figure
        Figure will be returned if plot=True, otherwise None. This allows you to change properties of the figure
        afterwards, e.g. fig.axes[0].set_title('This is my new title')

    Examples
    ----------
    >>> import seaborn as sns
    >>> from neuropy.correlation import correlations_as_sample_increases
    >>> iris = sns.load_dataset('iris')
    >>> summary,  fig = correlations_as_sample_increases(data = iris, feature1='petal_width',
    ... feature2='sepal_length', starting_N=20)


    """
    data = (
        data[[feature1, feature2]].copy()
        # Remove rows with np.nans
        .dropna()
        # Randomize order of the data
        .sample(frac=1, random_state=random_state)
    )
    if data.shape[0] < starting_N:
        raise ValueError("Number of valid cases is smaller than the starting_N")
    if data.shape[0] < starting_N + step:
        raise ValueError(
            "Number of valid cases is smaller than the starting_N + step (only one correlation possible)"
        )

    # Initiate data frame for results
    corr_results = pd.DataFrame()

    # Loop through all possible number of rows from starting N till number of rows
    for i in range(starting_N, data.shape[0] + 1, step):
        # Run correlation with all data from first row until row i
        current_corr = correlation_analysis(
            data.iloc[0:i], method=method, check_norm=False, permutation_test=False
        )[0]["summary"][["r-value", "p-value", "N"]]
        corr_results = pd.concat([corr_results, current_corr], ignore_index=True)

    fig = None
    if plot:
        fig, ax = plt.subplots()
        # Add r-value and p-value
        _ = sns.lineplot(
            corr_results["N"],
            abs(corr_results["r-value"]),
            label="absolute r-value",
            ax=ax,
        ).set_title(
            f"The absolute r-value between {feature1} and {feature2}\nas N increases {addition_to_title}"
        )
        _ = sns.lineplot(
            corr_results["N"], corr_results["p-value"], label="p-value", ax=ax
        )
        # Add alpha level (threshold for p-value)
        _ = ax.axhline(
            y=alpha, color="black", alpha=0.5, linestyle="--", label=f"\u03B1 = {alpha}"
        )

        _ = ax.set_ylabel("")
        _ = ax.set_ylim(0, 1)
        _ = plt.legend()
    return corr_results, fig


def rm_correlation_analysis(
    df: pd.DataFrame,
    feature_list: list = None,
    target_list: list = None,
    subject_column: str = "user_id",
    dropna: str = "pairwise",
):
    r"""Run a repeated measure correlation for numerical features and return output in different formats

    Repeated measures correlation (rmcorr) is a statistical technique for determining the common
    within-individual association for paired measures assessed on two or more occasions for multiple individuals.
    Inspired by `pinqoun.rm_corr`.

    Rmcorr can be viewed as a “light” version of multilevel modeling because it is comparable to a simple,
    null multilevel model with random/varying effects of intercept for each individual and a fixed effect
    (i.e., common/overall) slope. However, rmcorr only analyzes intra-individual variance. Multilevel modeling
     can simultaneously analyze both intra- and inter-individual variance using partial pooling, which permits
     varying slopes and other parameters that cannot be estimated with simpler techniques.

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with variables in columns, cases in rows
    feature_list: list or None (default: None)
        List with names of columns in `df` that represent keystroke dynamic features.
        If None, all numerical columns are used but only every unique combination.
    target_list: list or None (default: None)
        List with names of columns in `df` that represent endpoints or clinical measures.
        If None, all columns are used and only every unique combination.
    subject_column: str (default: 'user_id')
        Name of column in `df` containing the subject indicator.
    dropna : {'listwise', 'pairwise'}, default 'pairwise'
        Should rows with missing values be dropped over the complete `df` ('listwise') or for every correlation
        separately ('pairwise')

    Returns
    -------
    result_dict: dict
        Dictionary containing with the following keys:
        info : pandas.DataFrame
            Description of correlation method, missing values handling and number of observations
        r-values : pandas.DataFrame
            Dataframe with rm correlation coefficients. Indices and columns are column names from `data`. Only lower
            triangle is filled.
        p-values : pandas.DataFrame
            Dataframe with p-values. Indices and columns are column names from `data`. Only lower triangle is filled.
        N        : pandas.DataFrame
            Dataframe with numbers of observations. Indices and columns are column names from `data`. Only lower
            triangle is filled. If dropna ='listwise', every correlation will have the same number of observations.
        summary : pandas.DataFrame
            Dataframe with columns ['analysis', 'feature1', 'feature2', 'r-value', 'p-value', 'N', 'stat-sign']
            which indicate the type of test used for the correlation, the pair of columns, the correlation coefficient,
            the p-value, the number of observations for each combination of columns in `data` and whether the r-value is
            statistically significant.

    Examples
    ----------
    >>> from neuropy.correlation import rm_correlation_analysis
    >>> import pingouin as pg
    >>> df = pg.read_dataset('rm_corr')
    >>> dict_results = rm_correlation_analysis(df=df, feature_list=['pH'], target_list=['PacO2'], subject_column='Subject')
    >>> dict_results['summary']

    References
    ----------
    Bakdash, J.Z., Marusich, L.R., 2017. Repeated Measures Correlation. Front. Psychol. 8, 456.
    https://doi.org/10.3389/fpsyg.2017.00456

    """
    # Initial checks
    if (df.groupby([subject_column])[subject_column].transform("count").le(2)).any():
        df = df[df.groupby([subject_column])[subject_column].transform("count").ge(3)]
        logging.warning(
            f"Subjects with less then 3 samples were detected and removed. "
            f"Number of unique Subjects = {df[subject_column].nunique()}"
        )

    if (feature_list is not None) and (df[feature_list].isna().sum().any()):
        logging.warning(
            f"Nan values were found in the following columns:"
            f"{df[feature_list].columns[df[feature_list].isna().any()].tolist()}. "
            f"These will be removed using {dropna} dropping"
        )
    elif (feature_list and target_list is None) and (df.isna().sum().any()):
        logging.warning(
            f"Nan values were found in the following columns:"
            f"{df.columns[df.isna().any()].tolist()}. "
            f"These will be removed using {dropna} dropping"
        )
    elif (target_list is not None) and (df[target_list].isna().sum().any()):
        logging.warning(
            f"Nan values were found in the following columns:"
            f"{df[target_list].columns[df[target_list].isna().any()].tolist()}. "
            f"These will be removed using {dropna} dropping"
        )

    # Copy numerical data from the original data
    data = df.select_dtypes("number")

    # Get correct lists
    if target_list and not feature_list:
        feature_list = (
            data.select_dtypes("number")
            .drop(target_list + [subject_column], axis=1)
            .columns.tolist()
        )
    elif feature_list and not target_list:
        target_list = (
            data.select_dtypes("number")
            .drop(feature_list + [subject_column], axis=1)
            .columns.tolist()
        )

    # Initializing dataframes to store results
    info = pd.DataFrame()
    summary = pd.DataFrame()
    if not target_list and not feature_list:
        r_vals = pd.DataFrame(
            columns=data.drop(subject_column, axis=1).columns,
            index=data.drop(subject_column, axis=1).columns,
        )
        p_vals = pd.DataFrame(
            columns=data.drop(subject_column, axis=1).columns,
            index=data.drop(subject_column, axis=1).columns,
        )
        n_vals = pd.DataFrame(
            columns=data.drop(subject_column, axis=1).columns,
            index=data.drop(subject_column, axis=1).columns,
        )
        iterator = combinations(data.drop(subject_column, axis=1).columns, 2)
    else:
        r_vals = pd.DataFrame(columns=target_list, index=feature_list)
        p_vals = pd.DataFrame(columns=target_list, index=feature_list)
        n_vals = pd.DataFrame(columns=target_list, index=feature_list)
        iterator = product(target_list, feature_list)
        data = data[target_list + feature_list + [subject_column]]

    if dropna == "listwise":
        # Remove rows with missing values
        data = data.dropna(how="any", axis="index")
        info = info.append(
            {
                f"RM correlation test using {dropna} deletion": f"Total observations used = {len(data)}"
            },
            ignore_index=True,
        )
    elif dropna == "pairwise":
        info = info.append(
            {
                f"RM correlation test using {dropna} deletion": f"Observations in the data = {len(data)}"
            },
            ignore_index=True,
        )
    else:
        raise ValueError("dropna not in {'listwise', 'pairwise'}")

    for col1, col2 in iterator:
        if dropna == "pairwise":
            # Remove rows with missing values in the pair of columns
            test_data = data[[col1, col2, subject_column]].dropna()
        else:
            test_data = data

        if (
            test_data.groupby([subject_column])[subject_column].transform("count").le(2)
        ).any():
            test_data = test_data[
                test_data.groupby([subject_column])[subject_column]
                .transform("count")
                .ge(3)
            ]
            logging.warning(
                f"Subjects with less then 3 samples were detected and removed. "
                f"Number of unique Subjects = {df[subject_column].nunique()}"
            )

        if test_data[subject_column].nunique() < 3:
            r_value, p_value, n_value = np.nan, np.nan, np.nan
            logging.warning(
                f"At least 3 unique subjects are required. This condition is not met after dropping NaN for:"
                f"{col1} and {col2}. "
            )
        else:
            # Using PINGOUIN
            n_value = len(test_data)
            aov = pg.ancova(dv=col1, covar=col2, between=subject_column, data=test_data)
            bw = aov.bw_  # Beta within parameter
            sign = np.sign(bw)
            ssfactor = aov.at[1, "SS"]
            sserror = aov.at[2, "SS"]
            r_value = sign * np.sqrt(ssfactor / (ssfactor + sserror))
            p_value = aov.at[1, "p-unc"]

            # Store output in matrix format
            try:
                r_vals.loc[col2, col1] = r_value
                p_vals.loc[col2, col1] = p_value
                n_vals.loc[col2, col1] = n_value
            except KeyError:
                r_vals.loc[col1, col2] = r_value
                p_vals.loc[col1, col2] = p_value
                n_vals.loc[col1, col2] = n_value

        # Store output in dataframe format
        dict_summary = {
            "analysis": "rm_corr",
            "feature1": col1,
            "feature2": col2,
            "r-value": r_value,
            "p-value": p_value,
            "stat-sign": (p_value < 0.05),
            "N": n_value,
        }

        summary = pd.concat(
            [summary, pd.DataFrame(data=dict_summary, index=[0])],
            axis=0,
            ignore_index=True,
            sort=False,
        )

        # Embed results within a dictionary
        result_dict = {
            "r-value": r_vals,
            "p-value": p_vals,
            "N": n_vals,
            "info": info,
            "summary": summary,
        }

    return result_dict


def plot_rm_correlation(
    df: pd.DataFrame,
    feature: str,
    target: str,
    subject_column: str = "user_id",
    legend: bool = False,
    kwargs_facetgrid=dict(height=6, aspect=1, palette="Paired"),
):
    r"""

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with variables in columns, cases in rows
    feature, target : str
        Name of columns in `df` containing the two dependent variables.
    subject_column: str
        Name of column in `df` containing the subject indicator.
    legend : boolean (default=True)
        If True, add legend to plot. Legend will show all the unique values in
        ``subject``.
    kwargs_facetgrid : dict
        Optional keyword argument passed to :py:class:`seaborn.FacetGrid`

    Returns
    -------
    g : :py:class:`seaborn.FacetGrid`
        Seaborn FacetGrid.

    See also
    --------
    rm_correlation_analysis

    Notes
    -----
    Repeated measures correlation (rmcorr) is a statistical technique
    for determining the common within-individual association for paired
    measures assessed on two or more occasions for multiple individuals.

    Results have been tested against the `rmcorr` R package. Note that this
    function requires the statsmodels Python package.

    Missing values are automatically removed from the ``data``
    (listwise deletion).

    References
    ----------
    .. [1] Bakdash, J.Z., Marusich, L.R., 2017. Repeated Measures Correlation.
           Front. Psychol. 8, 456. https://doi.org/10.3389/fpsyg.2017.00456

    .. [2] Bland, J. M., & Altman, D. G. (1995). Statistics notes: Calculating
           correlation coefficients with repeated observations:
           Part 1—correlation within subjects. Bmj, 310(6977), 446.

    .. [3] https://github.com/cran/rmcorr

    Examples
    --------
    >>> from neuropy.correlation import plot_rm_correlation
    >>> import pingouin as pg
    >>> df = pg.read_dataset('rm_corr')
    >>> _ = plot_rm_correlation(df=df, feature='pH', target='PacO2', subject_column='Subject')

    """
    # Initial checks
    if (df.groupby([subject_column])[subject_column].transform("count").le(2)).any():
        df = df[df.groupby([subject_column])[subject_column].transform("count").ge(3)]
        logging.warning(
            f"Subjects with less then 3 samples were detected and removed. "
            f"Number of unique Subjects = {df[subject_column].nunique()}"
        )

    if df[subject_column].nunique() < 3:
        raise ValueError("At least 3 unique subjects are required.")

    # Copy numerical data from the original data
    test_data = df.select_dtypes("number")

    if test_data.loc[:, [feature, target]].isna().sum().any():
        raise ValueError("Input contains NaN.")

    # Fit ANCOVA model
    # https://patsy.readthedocs.io/en/latest/builtins-reference.html
    # C marks the data as categorical
    # Q allows to quote variable that do not meet Python variable name rule
    # e.g. if variable is "weight.in.kg" or "2A"
    formula = "Q('%s') ~ C(Q('%s')) + Q('%s')" % (target, subject_column, feature)
    model = ols(formula, data=test_data).fit()

    # Fitted values
    test_data["pred"] = model.fittedvalues

    # Define color palette
    if "palette" not in kwargs_facetgrid:
        kwargs_facetgrid["palette"] = sns.hls_palette(
            test_data[subject_column].nunique()
        )

    # set limits for plotting
    x_min = test_data[feature].min() * 0.95
    x_max = test_data[feature].max() * 1.05
    y_min = test_data[target].min() * 0.95
    y_max = test_data[target].max() * 1.05

    # Start plot
    g = sns.FacetGrid(test_data, hue=subject_column, **kwargs_facetgrid)
    _ = g.map(sns.regplot, feature, "pred", scatter=False, ci=None, truncate=True)
    _ = g.map(
        sns.scatterplot,
        feature,
        target,
        hue=subject_column,
        data=test_data,
        palette=kwargs_facetgrid["palette"],
        size=None,
    )
    _ = g.set(xlim=(x_min, x_max))
    _ = g.set(ylim=(y_min, y_max))
    plt.title(
        f"RM Correlation - {feature} - {target}",
        fontdict={"fontsize": 14, "fontweight": 14},
    )

    if legend:
        g.add_legend()

    return g
