# -*- coding: utf-8 -*-
"""Submodule frequentist_statistics.py includes the following functions:
   - chisquare(): Run chi-square test using `stats.chi2_contingency`
   - permutation_plot(): Show the result of the permutations
   - permute_test(): Helper function to run tests for permutations
   - dependent_t_test(): Run paired sample t-tests including assumptions using `scipy.stats.ttest_rel`
   - analysis_dependent_t_test(): Run `dependent_t_test` for all combinations of features
   - independent_t_test(): Run independent sample t-tests including assumptions using `scipyt.stats.levene` and
     `statsmodels.stats.weightstats.ttest_ind`
   - analysis_independent_t_test(): Run `independent_t_test` for all combinations of features
   - dependent_wilcoxon_test(): Run paired sample Wilcoxon test using `scipy.stats.wilcoxon`
   - analysis_dependent_wilcoxon_test(): Run `dependent_wilcoxon_test` for all combinations of features
   - independent_mannwhitneyu_test(): Run independent Mann-Whitney U test using `scipy.stats.mannwhitneyu`
   - analysis_independent_mannwhitneyu_test(): Run `independent_mannwhitneyu_test` for all features
   - one_way_ANOVA(): Run one-way ANOVAs using `scipy.stats.f_oneway` and check homogeneity of variances with Levenes
     test using `scipy.stats.levene`
   - mapping_dist_mean_median(): Adds line and text relative to moments of distribution to densityplots using
     `seaborn.distplots`
   - plot_distribution(): Plot distributions relative to numerical (ungrouped) features
   - ecdf(): Computes emperical cumulative distribution function for a one-dimensional array of measurements
   - mapping_ecdf(): Adds emperical and theoretical distribution of the data
   - compare_theoretical_ecdf(): compare emperical distribution to the theoretical distribution based on mean and
     standard deviation
   - normal_check(): compare the distribution of numeric variables to a normal distribution using the Kolmogrov-Smirnov
     test
   - apply_power_transformations(): transform data with log transformations and Box-cox transformations
   - find_optimal_transformation(): choose the best transformation based on the results of `normal_check` after
     `apply_power_transformations`
   - ols(): run a linear regression model with ordinary least squares (OLS) estimation
   - diagnostic_plots(): function to reproduce the 4 diagnostic plots of an OLS model in R.
   - correct_pvalues(): function to correct for mulitple testing
   - bland_altman_plot(): function to plot a mean-difference plot also know as a Bland-Altman Plot.
"""
import math
import warnings
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats.weightstats as smsw
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.eval_measures import rmse

from sklearn.model_selection import KFold
from sklearn.base import is_classifier
from sklearn.base import is_regressor
from sklearn.feature_selection import RFECV

def chisquare(data, groups, categories, show=True, correction=True):
    r"""Run chi-square test using `stats.chi2_contingency`

    Parameters
    ---------
    data : pandas.DataFrame
        Dataframe with variables in columns, cases in rows
    groups: str
        column name of groups in `data`
    categories: str
        column name of categories in `data`
    show: bool (default: True)
        Whether to print the results of the analysis
    correction: bool (default: True)
        Whether to apply the Yates’ correction for continuity

    Returns
    ---------
    pd.DataFrame
        Data frame with columns ['test-type', 'grouping-var', 'categories', 'chi-square', 'df', 'p-value', 'stat-sign']

    Examples
    ---------
    >>> tips = sns.load_dataset('tips')
    >>> chisquare(tips, 'smoker', 'sex')

    """

    ct = pd.crosstab(data[groups], data[categories])
    chi2, p_value, degrees_freedom, expected = stats.chi2_contingency(
        ct, correction=correction
    )

    if show:
        print(
            f"=== Chi-square test: groups = *{groups}* | categories = *{categories}* ===\n"
        )
        print(f"chi2 = {chi2:.2f}"), print(f"p-value = {p_value:.3f}"), print(
            f"df = {degrees_freedom:.2f}"
        )
        print("\nExpected frequencies")
        print(pd.DataFrame(data=expected, columns=ct.columns, index=ct.index).round(3))
        print("\nOriginal data")
        print(ct)

    # return result on a row data frame
    dict_result = {
        "test-type": "chisquare",
        "grouping-var": groups,
        "categories": categories,
        "chi-square": chi2,
        "df": degrees_freedom,
        "p-value": p_value,
        "stat-sign": (p_value < 0.05),
    }

    df_result = pd.DataFrame(data=dict_result, index=[0])

    if 1 in ct.shape:
        warnings.warn(
            f"One of {groups} and {categories} consisted of a single value",
            stacklevel=2,
        )
    if (ct < 5).any(axis=None):
        warnings.warn(
            "At least one observed frequency is < 5 so test should not be used",
            stacklevel=2,
        )
    if (expected < 5).any(axis=None):
        warnings.warn(
            "At least one expected frequency is < 5 so test should not be used",
            stacklevel=2,
        )

    return df_result


def permutation_plot(
    permutations, p_value: float, feature1=None, feature2=None, figsize=(11.7, 8.27)
):
    r"""Show the result of the permutations using a histogram

    Parameters
    ---------
    permutations: array-like
        Array with single p-value per permutation
    p_value: float
        P-value obtained from real data
    feature1: None or str (default: None)
        Name of the first feature involved
    feature2: None or str (default: None)
        Name of the second feature involved
    figsize: tuple (default: (11.7, 8.27))
        Width and height of the figure in inches

    Returns
    ---------
    permutation_plot: Figure

    Examples
    ---------

    Demonstration of the function using random data

    >>> permutations = np.random.rand(100)
    >>> permutation_plot = permutation_plot(permutations, p_value = 0.02)
    >>> # plt.show()

    """

    permutation_plot = plt.figure(figsize=figsize)
    _ = sns.distplot(
        permutations, bins=100, label="p-values from permutations of the data"
    )
    _ = plt.axvline(p_value, c="red", label="p-value from real data")
    _ = plt.ylim(0, 1.5)
    _ = plt.legend()

    # Add custom title
    if feature1 and feature2:
        _ = plt.title(f"Permutations {feature1} and {feature2}")
    return permutation_plot


def dependent_t_test(
    data: pd.DataFrame,
    feature1: str,
    feature2: str,
    show=False,
    plot=False,
    figsize=(11.7, 8.27),
):
    r"""Run paired sample t-tests including assumptions using `scipy.stats.ttest_rel`

    Rows with missing values in any of the features will be dropped.

    Parameters
    ----------
    data: pandas.DataFrame)
        Dataframe with `feature1` and `feature2` in columns
    feature1: str
        Name of the first feature
    feature2: str
        Name of the second feature
    show: bool
        whether to print the results
    plot: bool
        whether to plot the distribution and the data
    figsize: tuple (default: (11.7, 8.27))
        Width and height of the figure in inches

    Returns
    ----------
    df_t_test: pandas.DataFrame
        Dataframe with most important results (name of the test, features, means, stds,
                                mean difference, t-value, degrees of freedom, p-value, significance)
    residplot: Figure
        Figure if plot == True, else None
    boxplot: Figure
        Figure if plot == True, else None

    Examples
    ----------
    >>> tips = sns.load_dataset("tips")
    >>> bill_and_tip_difference, _, _ = dependent_t_test(tips, 'total_bill', 'tip', show = True, plot = False)

    """

    # select feature1 featur2 columns and remove row if any nan present
    data = data[[feature1, feature2]].copy()
    _ = data.dropna(axis=0, how="any", inplace=True)
    df_descriptive = data.describe()

    # Raise error if features are not numeric
    if not all(
        [
            feature in data.select_dtypes("number").columns
            for feature in [feature1, feature2]
        ]
    ):
        raise TypeError(f"Features ({feature1} and {feature2}) should be numeric")

    # Perform t test
    (t_value, p_value) = stats.ttest_rel(data[feature1], data[feature2])
    degrees_freedom = len(data) - 1
    diff = data[feature1] - data[feature2]
    # Confidence interval
    critical_t = stats.t.ppf(1 - 0.025, degrees_freedom)
    critical_z = 1.96
    moe = critical_t * diff.std() / math.sqrt(len(data))
    moe_z = critical_z * diff.std() / math.sqrt(len(data))

    if show:
        print(f"========= Paired sample t-tests between [ {feature1} | {feature2} ] \n")
        print("Pairs with missing values are dropped\n")
        # Summary of the data
        print(df_descriptive[:3])
        print("\n")

        # Print results t-test
        print(f"- t-value = {t_value:.2f}")
        print(f"- p-value = {p_value:.3f}")
        print(f"- degrees of freedom = {degrees_freedom}")
        print(
            f"- Difference between the two groups is {diff.mean():.2f} "
            f"[{diff.mean() - moe:.2f} to {diff.mean() + moe:.2f}] (mean [95% CI])"
        )
        print(
            f"- Confidence interval based on z [{diff.mean() - moe_z:.2f} to {diff.mean() + moe_z:.2f}] ([95% CI])"
        )
        if p_value < 0.05:
            print("- Statistical significance detected")
        else:
            print("- Statistical significance NOT detected")

    residplot = None
    boxplot = None

    if plot:
        # Assumption check: normality of residuals
        residplot, ax = plt.subplots(figsize=figsize)
        _ = sns.distplot(ax=ax, a=diff, rug=True)
        _ = ax.set_title(
            f"Check distribution of residuals \n res = {feature1} - {feature2}"
        )

        # Plot the data
        boxplot, ax = plt.subplots(figsize=figsize)
        _ = sns.boxplot(ax=ax, data=data[[feature1, feature2]], width=0.4)
        _ = sns.swarmplot(
            ax=ax, data=data[[feature1, feature2]], color=".25", alpha=0.5
        )
        _ = ax.set_title(f"Boxplot {feature1} and {feature2}")

    # format for row data frame
    feature1_mean_std = f"{df_descriptive.loc['mean', feature1]:.2f} ({df_descriptive.loc['std', feature1]:.2f})"
    feature2_mean_std = f"{df_descriptive.loc['mean', feature2]:.2f} ({df_descriptive.loc['std', feature2]:.2f})"

    # return result on a row data frame
    dict_result = {
        "test-type": "dependent T-test",
        "feature1": feature1,
        "n-sample1": len(data[feature1]),
        "mean1 (std1)": feature1_mean_std,
        "feature2": feature2,
        "n-sample2": len(data[feature2]),
        "mean2 (std2)": feature2_mean_std,
        "mean-difference": diff.mean(),
        "t-value": t_value,
        "df": degrees_freedom,
        "p-value": p_value,
        "stat-sign": (p_value < 0.05),
    }

    df_result = pd.DataFrame(data=dict_result, index=[0])

    return df_result, residplot, boxplot


def analysis_dependent_t_test(
    data: pd.DataFrame, features: list, show=False, plot=False, figsize=(11.7, 8.27)
):
    r"""Run `dependent_t_test` for all combinations of features

    Parameters
    ----------
    data: pandas.DataFrame)
        Dataframe with `features` in columns
    features: list
        list with feature names (str)
    show: bool
        whether to print the results
    plot: bool
        whether to plot the distribution and the data
    figsize: tuple (default: (11.7, 8.27))
        Width and height of the figure in inches

    Returns
    ----------
    df_analysis: pandas.DataFrame
        Dataframe with most important results (name of the test, features, means, stds,
                                mean difference, t-value, degrees of freedom, p-value, significance)
    list_residplots: list
        List with figures if plot == True, else with None
    list_boxplots: list
        List with figures if plot == True, else with None

    Examples
    ----------
    >>> tips = sns.load_dataset("tips")
    >>> df_analysis, _, _ = analysis_dependent_t_test(tips, ['total_bill', 'tip', 'size'], show=True, plot=False)

    """

    # initialize empty data frame
    df_analysis = pd.DataFrame()

    # initialize empty lists for figures
    list_residplots = []
    list_boxplots = []

    for combination in combinations(features, 2):
        df_t_test, residplot, boxplot = dependent_t_test(
            data,
            feature1=combination[0],
            feature2=combination[1],
            show=show,
            plot=plot,
            figsize=figsize,
        )
        # Add the results of the t-test to the general dataframe
        df_analysis = pd.concat(
            [df_analysis, df_t_test], axis=0, ignore_index=True, sort=False
        )

        # Add the plots to the lists
        list_residplots.append(residplot)
        list_boxplots.append(boxplot)

    # Sort the plots by the p-values of the dataframe
    idx = df_analysis["p-value"].argsort()
    list_residplots = [list_residplots[i] for i in idx.values.tolist()]
    list_boxplots = [list_boxplots[i] for i in idx.values.tolist()]

    # Sort the results dataframe by p-values
    _ = df_analysis.sort_values(by=["p-value"], ascending=True, inplace=True)

    return df_analysis, list_residplots, list_boxplots


def permute_test(a, test_type, test, **kwargs):
    r"""Helper function to run tests for permutations

    Parameters
    ----------
    a : np.array
    test_type: str {'correlation', 'independent_t_test'}
        Type of the test to be used
    test:
        e.g. `scipy.stats.pearsonr` or `statsmodels.stats.weightstats.ttest_ind`

    **kwargs:
        Additional keywords to be added to `test`
        - `n` for the number of observations in the first group if test_type = 'independent_t_test'
        - `usevar` for the type of variances if test_type = 'independent_t_test'
        - `a2` for the second feature if test_type = 'correlation'

    Returns
    ----------
    float:
        p value for permutation

    """
    if test_type == "correlation":
        a2 = kwargs["a2"]
        _, p = test(a, a2)

    elif test_type == "independent_t_test":
        n = kwargs["n"]
        usevar = kwargs.get("usevar", "pooled")
        _, p, _ = test(a[:n], a[n:], usevar=usevar)
    elif test_type == "mannwhitneyu":
        n = kwargs["n"]
        _, p = test(a[:n], a[n:])
    else:
        raise ValueError("Unknown test_type provided")

    return p


def independent_t_test(
    data: pd.DataFrame,
    feature: str,
    grouping_var: str,
    grouping_grp1: str,
    grouping_grp2: str,
    permutation_test=False,
    n_permutations=1000,
    random_state=None,
    show=False,
    plot=False,
    figsize=(11.7, 8.27),
):
    r"""Run independent sample t-tests including assumptions using `scipy.stats.levene` and `statsmodels.stats.
    weightstats.ttest_ind`

    Rows with missing values in `features` or `grouping_var` will be dropped.

    Parameters
    ----------
    data: pandas.DataFrame)
        Dataframe with `feature` and `grouping_var` in columns
    feature: str
        Name of the feature
    grouping_var: str
        Name of the  column with grouping labels in `data`
    grouping_grp1: str
        Name of the first label in `data[grouping_var]`
    grouping_grp2: str
        Name of the second label in `data[grouping_var]`
    permutation_test: bool (default: False)
        If true, a permutation test will added
    n_permutations: int (default: 1000)
        Number of permutations in the permutation test
    random_state: None or int (default: None)
        Random state for permutation_test. If not None, random_state will be updated for every permutation
    show: bool (default: False)
        whether to print the results
    plot: bool (default: False)
        whether to plot the distribution and the data
    figsize: tuple (default: (11.7, 8.27))
        Width and height of the figure in inches

    Returns
    ----------
    df_t_test: pandas.DataFrame
        Dataframe with most important results (name of the test, feature, groups, means, stds,
                                Levenes test, t-value, degrees of freedom, p-value, significance)
    distplot: matplotlib.figure.Figure with distributions (if plot is True else None)
    boxplot: matplotlib.figure.Figure with data (if plot is True else None)
    p_plot: matplotlib.figure.Figure with permutation results (if plot and permutation_test else None)

    Examples
    ----------
    >>> tips = sns.load_dataset("tips")
    >>> sex_tip_difference, _, _, _ = independent_t_test(tips, 'tip','sex', 'Female', 'Male', show = True, plot = False)

    """

    # select the 'feature' and 'grouping_var' columns and remove row if any nan present
    data = data.copy()
    data = data[[feature, grouping_var]].dropna(axis=0, how="any")

    # Raise error if feature is not numeric
    if feature not in data.select_dtypes("number").columns:
        raise TypeError(f"Feature {feature} should be numeric")

    # New dataframes
    group1 = data.loc[data[grouping_var] == grouping_grp1]
    _ = group1.reset_index(inplace=True)
    group2 = data.loc[data[grouping_var] == grouping_grp2]
    _ = group2.reset_index(inplace=True)

    # get descriptive values
    df_descriptive = data.groupby(grouping_var)[feature].describe()

    if (grouping_grp1 not in df_descriptive.index) or (
        grouping_grp2 not in df_descriptive.index
    ):
        warnings.warn(
            f"One of the groups did not have any observations for {feature}",
            stacklevel=2,
        )
        dict_result = {
            "test-type": "independent T-test",
            "feature": feature,
            "group-var": grouping_var,
        }
        return pd.DataFrame(data=dict_result, index=[0]), None, None, None

    # Store descriptives
    mean_group1 = df_descriptive.loc[grouping_grp1, "mean"]
    mean_group2 = df_descriptive.loc[grouping_grp2, "mean"]
    std_group1 = df_descriptive.loc[grouping_grp1, "std"]
    std_group2 = df_descriptive.loc[grouping_grp2, "std"]

    # Check assumption: homogeneity of variances
    (levene, p_value) = stats.levene(group1[feature], group2[feature])

    if show:
        print(
            f"=== Independent t-tests: variable = *{feature}* | group = *{grouping_grp1}* & *{grouping_grp2}*"
            f" defined in *{grouping_var}* ===\n"
        )
        print("Missing values are dropped\n")

        # Describe the samples
        print(df_descriptive)
        print("\n")

        # Print results Levenes test
        print("- Levenes test for homogeneity of variances (H0 = homogeneity):")
        print(f"- W = {levene:.2f}")
        print(f"- p-value = {p_value: .3f}")

    distplot = None

    if plot:
        # Check assumpution: normally distributed within groups
        distplot, axs = plt.subplots(1, 2, figsize=figsize)
        _ = distplot.suptitle("Check distribution within the groups")
        _ = sns.distplot(group1[feature], rug=True, ax=axs[0]).set_title(grouping_grp1)
        _ = sns.distplot(group2[feature], rug=True, ax=axs[1]).set_title(grouping_grp2)

    if p_value < 0.05:
        # YES: Run t-test using statsmodels - unequal variance
        if show:
            print("- Unequal variances detected \n")

        group_variance = "Unequal"
        use_var = "unequal"
    else:
        # NO: Run t-test using statsmodels - equal variances
        if show:
            print("- Equal variances detected \n")

        group_variance = "Equal"
        use_var = "pooled"

    # Run the t-test
    (t_value, p_value, degrees_freedom) = smsw.ttest_ind(
        group1[feature], group2[feature], usevar=use_var
    )

    # Store results in dictionary
    dict_result = {  # type: ignore
        "test-type": "independent T-test",
        "feature": feature,
        "group-var": grouping_var,
        "group1": grouping_grp1,
        "n-sample1": len(group1[feature]),
        "mean1 (std1)": f"{round(mean_group1, 2)} ({round(std_group1, 2)})",
        "group2": grouping_grp2,
        "n-sample2": len(group2[feature]),
        "mean2 (std2)": f"{round(mean_group2, 2)} ({round(std_group2, 2)})",
        "variances": group_variance,
        "t-value": t_value,
        "df": degrees_freedom,
        "p-value": p_value,
        "stat-sign": (p_value < 0.05),
    }

    # Run the permutation test
    if permutation_test:
        total_group = np.array(group1[feature].append(group2[feature]))
        # Copy the complete data
        total_group = np.repeat(total_group[:, np.newaxis], n_permutations, axis=1)
        # Shuffle within the columns
        np.random.seed(random_state)
        ix_i = np.random.sample(total_group.shape).argsort(axis=0)
        ix_j = np.tile(np.arange(total_group.shape[1]), (total_group.shape[0], 1))
        total_group = total_group[ix_i, ix_j]
        permutations = np.apply_along_axis(
            permute_test,
            axis=0,
            arr=total_group,
            test_type="independent_t_test",
            test=smsw.ttest_ind,
            n=len(group1),
            usevar=use_var,
        )

        extreme_permutation = np.where(permutations < p_value, 1, 0)
        p_permutation = extreme_permutation.sum() / len(permutations)
        dict_result["permutation-p-value"] = p_permutation

        # Reset random seed numpy
        np.random.seed(None)

    if show:
        # Print results t-test
        print("- Outcome t-test: ")
        print(f"- t-value = {t_value:.2f}")
        print(f"- p-value = {p_value:.3f}")
        print(f"- degrees of freedom = {degrees_freedom:.2f}")
        if p_value < 0.05:
            print("- Statistical significance detected")
        else:
            print("- Statistical significance NOT detected")

        print("\n")

    boxplot = None
    p_plot = None

    if plot:
        # Plot the data
        boxplot, ax = plt.subplots(figsize=figsize)
        _ = sns.boxplot(
            ax=ax,
            x=grouping_var,
            y=feature,
            data=data.loc[data[grouping_var].isin([grouping_grp2, grouping_grp1])],
            width=0.4,
        )
        _ = sns.swarmplot(
            ax=ax,
            x=grouping_var,
            y=feature,
            data=data.loc[data[grouping_var].isin([grouping_grp2, grouping_grp1])],
            color=".25",
            alpha=0.50,
        )

        _ = ax.set_title(f"Boxplot {feature} across {grouping_var}")

        # Add plot for permutations
        if permutation_test:
            p_plot = permutation_plot(
                permutations,
                p_value,
                feature1=feature,
                feature2=grouping_var,
                figsize=figsize,
            )

    # Return results using a data frame
    df_result = pd.DataFrame(data=dict_result, index=[0])
    return df_result, distplot, boxplot, p_plot


def analysis_independent_t_test(
    data: pd.DataFrame,
    features: list,
    grouping_var: str,
    grouping_grp1: str,
    grouping_grp2: str,
    permutation_test=False,
    n_permutations=1000,
    random_state=None,
    show=False,
    plot=False,
    figsize=(11.7, 8.27),
):
    r"""Run `independent_t_test` for all combinations of features

    Parameters
    ----------
    data: pandas.DataFrame)
        Dataframe with `features` and `grouping_var` in columns
    features: list
        list with feature names (str)
    grouping_var: str
        Name of the  column with grouping labels in `data`
    grouping_grp1: str
        Name of the first label in `data[grouping_var]`
    grouping_grp2: str
        Name of the second label in `data[grouping_var]`
    permutation_test: bool
        If true, a permutation test will added
    n_permutations: int (default: 1000)
        Number of permutations in the permutation test
    random_state: None or int (default: None)
        Random state for permutations
    show: bool
        whether to print the results
    plot: bool
        whether to plot the distribution and the data
    figsize: tuple (default: (11.7, 8.27))
        Width and height of the figure in inches

    Returns
    ----------
    df_analysis: pandas.DataFrame
        Dataframe with most important results (name of the test, features, groups, means, stds,
                                Levenes test, t-value, degrees of freedom, p-value, significance)
    list_distplots: list
        List with figures if plot == True, else with None
    list_boxplots: list
        List with figures if plot == True, else with None
    list_p_plots: list
        List with figures if plot == True, else with None

    Examples
    ----------
    >>> tips = sns.load_dataset("tips")
    >>> df_analysis, _, _, _ = analysis_independent_t_test(tips, ['total_bill', 'tip', 'size'], 'sex', 'Female', 'Male',
    >>>                                           show=True, plot=False)

    """

    # initialize empty data frame
    df_analysis = pd.DataFrame()

    # initialize empty lists for figures
    list_distplots = []
    list_boxplots = []
    list_p_plots = []

    # run t-test for every feature
    for feature in features:
        df_t_test, distplot, boxplot, p_plot = independent_t_test(
            data,
            feature,
            grouping_var,
            grouping_grp1,
            grouping_grp2,
            permutation_test=permutation_test,
            n_permutations=n_permutations,
            random_state=random_state,
            show=show,
            plot=plot,
            figsize=figsize,
        )

        df_analysis = pd.concat([df_analysis, df_t_test], axis=0, ignore_index=True)
        list_distplots.append(distplot)
        list_boxplots.append(boxplot)
        list_p_plots.append(p_plot)

    # Sort the plots by the p-values of the dataframe
    idx = df_analysis["p-value"].argsort()
    list_distplots = [list_distplots[i] for i in idx.values.tolist()]
    list_boxplots = [list_boxplots[i] for i in idx.values.tolist()]
    list_p_plots = [list_p_plots[i] for i in idx.values.tolist()]

    # sort results by p-value
    _ = df_analysis.sort_values(by=["p-value"], ascending=True, inplace=True)

    return df_analysis, list_distplots, list_boxplots, list_p_plots


def dependent_wilcoxon_test(
    data: pd.DataFrame,
    feature1: str,
    feature2: str,
    show=False,
    plot=False,
    figsize=(11.7, 8.27),
):
    r"""Run paired sample Wilcoxon test using `scipy.stats.wilcoxon`

    Rows with missing values in any of the features will be dropped.

    Parameters
    ----------
    data: pandas.DataFrame)
        Dataframe with `feature1` and `feature2` in columns
    feature1: str
        Name of the first feature
    feature2: str
        Name of the second feature
    show: bool
        whether to print the results
    plot: bool
        whether to plot the distribution and the data
    figsize: tuple (default: (11.7, 8.27))
        Width and height of the figure in inches

    Returns
    ----------
    df_t_test: pandas.DataFrame
        Dataframe with most important results (name of the test, features, means, stds,
                                mean difference, t-value, p-value, significance)
    boxplot: Figure
        Figure if plot == True, else None

    Examples
    ----------
    >>> tips = sns.load_dataset("tips")
    >>> bill_and_tip_difference, _ = dependent_wilcoxon_test(tips, 'total_bill', 'tip', show = True, plot = True)

    """

    # select the 'feature1' and 'feature2' columns and remove row if any nan present
    data = data.copy()
    data = data[[feature1, feature2]].dropna(axis=0, how="any")

    # Descriptives
    df_descriptive = data[[feature1, feature2]].describe()[:3]

    # Raise error if features are not numeric
    if not all(
        [
            feature in data.select_dtypes("number").columns
            for feature in [feature1, feature2]
        ]
    ):
        raise TypeError(f"Features ({feature1} and {feature2}) should be numeric")

    # Non parametrical test
    t_value, p_value = stats.wilcoxon(data[feature1], data[feature2])

    # format for row data frame
    feature1_mean_std = f"{df_descriptive.loc['mean', feature1]:.2f} ({df_descriptive.loc['std', feature1]:.2f})"
    feature2_mean_std = f"{df_descriptive.loc['mean', feature2]:.2f} ({df_descriptive.loc['std', feature2,]:.2f})"

    # return result on a row data frame
    dict_result = {
        "test-type": "dependent Wilcoxon test",
        "feature1": feature1,
        "n-sample1": len(data[feature1]),
        "mean1 (std1)": feature1_mean_std,
        "feature2": feature2,
        "n-sample2": len(data[feature2]),
        "mean2 (std2)": feature2_mean_std,
        "t-value": t_value,
        "p-value": p_value,
        "stat-sign": (p_value < 0.05),
    }

    df_result = pd.DataFrame(data=dict_result, index=[0])

    if show:
        print(
            f"========= Paired sample Wilcoxon tests between [ {feature1} | {feature2} ] \n"
        )
        # Summary of the data
        print(df_descriptive)
        print("\n")
        # Results Wilcoxon
        print(f"- t-value Wilcoxon signed rank test = {t_value:.2f}")
        print(f"- p-value wilcoxon signed rank test = {p_value:.3f}")
        if p_value < 0.05:
            print("- Statistical significance detected")
        else:
            print("- Statistical significance NOT detected")

    boxplot = None

    if plot:
        # Plot the data
        boxplot, ax = plt.subplots(figsize=figsize)
        _ = sns.boxplot(ax=ax, data=data[[feature1, feature2]], width=0.5)
        _ = sns.swarmplot(
            ax=ax, data=data[[feature1, feature2]], color=".25", alpha=0.5
        )
        _ = ax.set_title(f"Boxplot {feature1} and {feature2}")
        print("\n")

    return df_result, boxplot


def analysis_dependent_wilcoxon_test(
    data: pd.DataFrame, features: list, show=False, plot=False, figsize=(11.7, 8.27)
):
    r"""Run `dependent_wilcoxon_test` for all combinations of features

    Parameters
    ----------
    data: pandas.DataFrame)
        Dataframe with `features` in columns
    features: list
        list with feature names (str)
    show: bool
        whether to print the results
    plot: bool
        whether to plot the distribution and the data
    figsize: tuple (default: (11.7, 8.27))
        Width and height of the figure in inches

    Returns
    ----------
    df_analysis: pandas.DataFrame
        Dataframe with most important results (name of the test, features, means, stds,
                                mean difference, t-value, p-value, significance)
    list_boxplots: list
        List with figures if plot == True, else with None

    Examples
    ----------
    >>> tips = sns.load_dataset("tips")
    >>> df_analysis, _ = analysis_dependent_wilcoxon_test(tips, ['total_bill', 'tip', 'size'], show=False, plot=False)

    """

    # initialize empty data frame
    df_analysis = pd.DataFrame()
    # initialize empty list for plots
    list_boxplots = []

    # run t-test for all combinations of features
    for combination in combinations(features, 2):
        df_wilcoxon_test, boxplot = dependent_wilcoxon_test(
            data,
            feature1=combination[0],
            feature2=combination[1],
            show=show,
            plot=plot,
            figsize=figsize,
        )
        df_analysis = pd.concat(
            [df_analysis, df_wilcoxon_test], axis=0, ignore_index=True
        )
        list_boxplots.append(boxplot)

    # Sort the plots by the p-values of the dataframe
    idx = df_analysis["p-value"].argsort()
    list_boxplots = [list_boxplots[i] for i in idx.values.tolist()]

    # Sort data based on p-value
    _ = df_analysis.sort_values(by=["p-value"], ascending=True, inplace=True)

    return df_analysis, list_boxplots


def independent_mannwhitneyu_test(
    data: pd.DataFrame,
    feature: str,
    grouping_var: str,
    grouping_grp1: str,
    grouping_grp2: str,
    permutation_test=False,
    n_permutations=1000,
    random_state=None,
    show=False,
    plot=False,
    figsize=(11.7, 8.27),
):
    r"""Run independent Mann-Whitney U test using `scipy.stats.mannwhitneyu`

    Rows with missing values in `feature` or `grouping_var` will be dropped.

    Parameters
    ----------
    data: pandas.DataFrame)
        Dataframe with `feature` and `grouping_var` in columns
    feature: str
        Name of the feature
    grouping_var: str
        Name of the  column with grouping labels in `data`
    grouping_grp1: str
        Name of the first label in `data[grouping_var]`
    grouping_grp2:
        Name of the second label in `data[grouping_var]`
    permutation_test: bool (default: False)
        If true, a permutation test will added
    n_permutations: int (default: 1000)
        Number of permutations in the permutation test
    random_state: None or int (default: None)
        Random state for permutation_test. If not None, random_state will be updated for every permutation
    show: bool
        whether to print the results
    plot: bool
        whether to plot the distribution and the data
    figsize: tuple (default: (11.7, 8.27))
        Width and height of the figure in inches

    Returns
    ----------
    df_t_test: pandas.DataFrame
        Dataframe with most important results (name of the test, feature, groups, means, stds,
                                                         u-value, p-value, significance)
    boxplot: Figure
        Figure if plot == True, else None
    p_plot: Figure
        Figure if plot == True, else None

    Examples
    ----------
    >>> tips = sns.load_dataset("tips")
    >>> sex_tip_difference, _, _ = independent_mannwhitneyu_test(tips, 'tip','sex', 'Female', 'Male', show = True,
    >>>                                                       plot = True)

    """

    # select the 'feature' and 'grouping_var' columns and remove row if any nan present
    data = data.copy()
    data = data[[feature, grouping_var]].dropna(axis=0, how="any")

    # Raise error if feature is not numeric
    if feature not in data.select_dtypes("number").columns:
        raise TypeError(f"Feature {feature} should be numeric")

    # New dataframes
    group1 = data.loc[data[grouping_var] == grouping_grp1]
    _ = group1.reset_index(inplace=True)
    group2 = data.loc[data[grouping_var] == grouping_grp2]
    _ = group2.reset_index(inplace=True)

    # get descriptive values
    df_descriptive = data.groupby(grouping_var)[feature].describe()

    if (grouping_grp1 not in df_descriptive.index) or (
        grouping_grp2 not in df_descriptive.index
    ):
        warnings.warn(
            f"One of the groups did not have any observations for {feature}",
            stacklevel=2,
        )
        dict_result = {
            "test-type": "independent MannwhitneyU test",
            "feature": feature,
            "group-var": grouping_var,
        }
        return pd.DataFrame(data=dict_result, index=[0]), None, None

    mean_group1 = df_descriptive.loc[grouping_grp1, "mean"]
    mean_group2 = df_descriptive.loc[grouping_grp2, "mean"]
    std_group1 = df_descriptive.loc[grouping_grp1, "std"]
    std_group2 = df_descriptive.loc[grouping_grp2, "std"]

    # Run nonparametric test
    (u_value, p_value) = stats.mannwhitneyu(group1[feature], group2[feature])

    if show:
        print(
            f"=== Independent Mann-Whitney U: variable = *{feature}* | group = *{grouping_grp1}* & *{grouping_grp2}*"
            f" defined in *{grouping_var}* ===\n"
        )
        # Describe the samples
        print(df_descriptive)
        print("\n")

        print(f"- Mann-Whitney U = {u_value:.2f}")
        print(f"- Mann-Whitney p-value = {p_value:.3f}")
        if p_value < 0.05:
            print("- Statistical significance detected")
        else:
            print("- Statistical significance NOT detected")

        print("\n")

    # return result on a row data frame
    group1_mean_std = f"{mean_group1:.2f} ({std_group1:.2f})"
    group2_mean_std = f"{mean_group2:.2f} ({std_group2:.2f})"

    dict_result = {  # type: ignore
        "test-type": "independent MannwhitneyU test",
        "feature": feature,
        "group-var": grouping_var,
        "group1": grouping_grp1,
        "n-sample1": len(group1[feature]),
        "mean1 (std1)": group1_mean_std,
        "group2": grouping_grp2,
        "n-sample2": len(group2[feature]),
        "mean2 (std2)": group2_mean_std,
        "u-value": u_value,
        "p-value": p_value,
        "stat-sign": (p_value < 0.05),
    }

    # Run the permutation test
    if permutation_test:
        total_group = np.array(group1[feature].append(group2[feature]))
        # Copy the complete data
        total_group = np.repeat(total_group[:, np.newaxis], n_permutations, axis=1)
        # Shuffle within the columns
        np.random.seed(random_state)
        ix_i = np.random.sample(total_group.shape).argsort(axis=0)
        ix_j = np.tile(np.arange(total_group.shape[1]), (total_group.shape[0], 1))
        total_group = total_group[ix_i, ix_j]
        permutations = np.apply_along_axis(
            permute_test,
            axis=0,
            arr=total_group,
            test_type="mannwhitneyu",
            test=stats.mannwhitneyu,
            n=len(group1),
        )

        extreme_permutation = np.where(permutations < p_value, 1, 0)
        p_permutation = extreme_permutation.sum() / len(permutations)
        dict_result["permutation-p-value"] = p_permutation

        # Reset random seed numpy
        np.random.seed(None)

    boxplot = None
    p_plot = None

    if plot:
        # Plot the data
        boxplot, ax = plt.subplots(figsize=figsize)
        _ = sns.boxplot(
            ax=ax,
            x=grouping_var,
            y=feature,
            data=data.loc[data[grouping_var].isin([grouping_grp2, grouping_grp1])],
            width=0.4,
        )
        _ = sns.swarmplot(
            ax=ax,
            x=grouping_var,
            y=feature,
            data=data.loc[data[grouping_var].isin([grouping_grp2, grouping_grp1])],
            color=".25",
            alpha=0.50,
        )

        _ = ax.set_title(f"Boxplot {feature} across {grouping_var}")

        # Add plot for permutations
        if permutation_test:
            p_plot = permutation_plot(
                permutations,
                p_value,
                feature1=feature,
                feature2=grouping_var,
                figsize=figsize,
            )

    df_result = pd.DataFrame(data=dict_result, index=[0])
    return df_result, boxplot, p_plot


def analysis_independent_mannwhitneyu_test(
    data: pd.DataFrame,
    features: list,
    grouping_var: str,
    grouping_grp1: str,
    grouping_grp2: str,
    permutation_test=False,
    n_permutations=1000,
    random_state=None,
    show=False,
    plot=False,
    figsize=(11.7, 8.27),
):
    """Run `independent_mannwhitneyu_test` for all features

    Parameters
    ----------
    data: pandas.DataFrame)
        Dataframe with `features` and `grouping_var` in columns
    features: list
        list with feature names (str)
    grouping_var: str
        Name of the  column with grouping labels in `data`
    grouping_grp1: str
        Name of the first label in `data[grouping_var]`
    grouping_grp2:
        Name of the second label in `data[grouping_var]`
    permutation_test: bool
        If true, a permutation test will added
    n_permutations: int (default: 1000)
        Number of permutations in the permutation test
    random_state: None or int (default: None)
        Random state for permutations
    show: bool
        whether to print the results
    plot: bool
        whether to plot the distribution and the data
    figsize: tuple (default: (11.7, 8.27))
        Width and height of the figure in inches

    Returns
    ----------
    df_analysis: pandas.DataFrame
        Dataframe with most important results (name of the test, features, groups, means, stds,
                                                        u-value, p-value, significance)
    list_boxplots: list
        List with figures if plot == True, else None
    list_p_plots: list
        List with figures if plot == True, else None

    Examples
    ----------
    >>> tips = sns.load_dataset("tips")
    >>> df_analysis, _, _ = analysis_independent_mannwhitneyu_test(tips, ['total_bill', 'tip', 'size'], 'sex', 'Female',
    >>>                                                      'Male', show = True, plot = False)

    """

    # initialize empty data frame
    df_analysis = pd.DataFrame()
    # initialize empty lists for figures
    list_boxplots: list = []
    list_p_plots: list = []

    for feature in features:
        df_mannwhitneyu_test, boxplot, p_plot = independent_mannwhitneyu_test(
            data,
            feature,
            grouping_var,
            grouping_grp1,
            grouping_grp2,
            permutation_test=permutation_test,
            n_permutations=n_permutations,
            random_state=random_state,
            show=show,
            plot=plot,
            figsize=figsize,
        )

        df_analysis = pd.concat(
            [df_analysis, df_mannwhitneyu_test], axis=0, ignore_index=True
        )
        list_boxplots.append(boxplot)
        list_p_plots.append(p_plot)

    # Sort the plots by the p-values of the dataframe
    idx = df_analysis["p-value"].argsort()
    list_boxplots = [list_boxplots[i] for i in idx.values.tolist()]
    list_p_plots = [list_p_plots[i] for i in idx.values.tolist()]

    # Sort the results based on p-value
    _ = df_analysis.sort_values(by=["p-value"], ascending=True, inplace=True)

    return df_analysis, list_boxplots, list_p_plots


def one_way_ANOVA(
    data: pd.DataFrame,
    feature: str,
    grouping_var: str,
    groups_of_interest: list,
    show=False,
    plot=False,
    figsize=(11.7, 8.27),
    col_wrap=None,
):
    r"""Run one-way ANOVAs using `scipy.stats.f_oneway` and check homogeneity of variances with Levenes test
    using `scipy.stats.levene`

    `one_way_ANOVA` assumes equal variances within the groups and will not give a warning if show=False

    Parameters
    ----------
    data: pandas.DataFrame)
        Dataframe with `feature` and `grouping_var` in columns
    feature: str
        Name of the feature
    grouping_var: str
        Name of the  column with grouping labels in `data`
    groups_of_interest:
        Names (str) of labels in `data[grouping_var]`
    show: bool
        whether to print the results
    plot: bool
        whether to plot the distribution and the data
    figsize: tuple (default: (11.7, 8.27))
        Width and height of the figure in inches
    col_wrap: int or None (default: None)
        If int, number of subplots that are allowed in a single row

    Returns
    ----------
    df_result: pd.DataFrame
    df_descriptive: pd.DataFrame
    distplot: Figure
        Figure if plot == True, else None
    boxplot: Figure
        Figure if plot == True, else None

    Examples
    ----------
    >>> import seaborn as sns
    >>> tips = sns.load_dataset("tips")
    >>> _, _, _, _ = one_way_ANOVA(tips, 'tip', 'day', ['Sat','Sun','Thur'], show = True, plot = False)

    """

    # select the 'feature' and 'grouping_var' columns and remove row if any nan present
    data = data.copy()
    data = data[[feature, grouping_var]].dropna(axis=0, how="any")

    # Raise error if feature is not numeric
    if feature not in data.select_dtypes("number").columns:
        raise TypeError(f"Feature {feature} should be numeric")

    # select the groups of interest and remove any not used category from the categorical index
    data = data.loc[data[grouping_var].isin(groups_of_interest), :]
    if data[grouping_var].dtype.name == "category":
        data[grouping_var] = data[grouping_var].cat.remove_unused_categories()

    # get descriptive values, keep only interested rows
    df_descriptive = data.groupby(grouping_var, observed=True)[feature].describe()
    _ = df_descriptive.reset_index(inplace=True)

    # Raise warning if groups of interest not in the dataframe
    if not all(
        grp in df_descriptive[grouping_var].values.tolist()
        for grp in groups_of_interest
    ):
        warnings.warn(
            f"One of the groups did not have any observations for {feature}",
            stacklevel=2,
        )

    values_per_group = {
        grp_label: values
        for grp_label, values in data.groupby(grouping_var, observed=True)[feature]
    }

    # Check assumption: homogeneity of variances
    (levene, levene_p_value) = stats.levene(*values_per_group.values())

    if levene_p_value > 0.05:
        # Equal variances:
        variance_outcome = "Equal"
        trust_results = "trustworthy"
    else:
        # Unequal variances: ANOVA cannot be trusted
        variance_outcome = "Unequal"
        trust_results = "untrustworthy"

    # Run one way ANOVA
    (f_value, p_value) = stats.f_oneway(*values_per_group.values())
    df1 = len(df_descriptive) - 1
    df2 = len(data) - len(df_descriptive)
    if show:
        print(
            f"=== One-way anova: variable = *{feature}* | groups = *{', '.join(groups_of_interest)}"
            f"* defined in *{grouping_var}* ===\n"
        )
        print("Missing values are dropped\n")

        # Describe the samples
        print(df_descriptive)
        print("\n")

        # Print results Levenes test
        print("Levenes test for homogeneity of variances (H0 = homogeneity):")
        print(f"- W = {levene:.2f}")
        print(f"- p-value = {levene_p_value:.3f}")

        if levene_p_value > 0.05:
            # Equal variances:
            print("- Equal variances detected \n")
        else:
            print(
                "- Unequal variances detected by Levenes test, so ANOVA results might be untrustworthy"
            )

        # Print results ANOVA
        print("Outcome ANOVA: ")
        print(f"- F-value = {f_value:.2f}")
        print(f"- df1 = {df1}")
        print(f"- df2 = {df2}")
        print(f"- p-value = {p_value:.3f}")

        if p_value < 0.05:
            print("- Statistical significance detected")
        else:
            print("- Statistical significance NOT detected")
        print("\n")

    distplot = None
    boxplot = None

    if plot:
        # Check assumpution: normally distributed within groups
        distplot = sns.FacetGrid(
            data, col=grouping_var, sharex=False, sharey=False, col_wrap=col_wrap
        )
        _ = plt.gcf().set_size_inches(figsize)
        _ = plt.gcf().suptitle("Check distribution within the groups")
        _ = distplot.map(sns.distplot, feature, rug=True)
        _ = plt.subplots_adjust(top=0.8)

        # Plot the data
        boxplot, ax = plt.subplots(figsize=figsize)
        _ = sns.boxplot(ax=ax, x=grouping_var, y=feature, data=data)
        _ = sns.swarmplot(
            ax=ax, x=grouping_var, y=feature, data=data, color=".25", alpha=0.50
        )
        _ = ax.set_title(f"Boxplot {feature} across {grouping_var}")

    dict_result = {
        "test-type": "one way ANOVA",
        "feature": feature,
        "group-var": grouping_var,
        "f-value": round(f_value, 3),
        "p-value": round(p_value, 3),
        "stat-sign": (p_value < 0.05),
        "variance": variance_outcome,
        "results": trust_results,
    }

    df_result = pd.DataFrame(data=dict_result, index=[0])

    return df_result, df_descriptive, distplot, boxplot


def mapping_dist_mean_median(x, **kwargs):
    r"""Adds line and text relative to moments of distribution to densityplots using `seaborn.distplots`

    Helper function (map) for plotting distributions, can be used with `plot_distribution`
    Adds dashed line at mean value and solid line at median value.
    Adds text with mean, median, skewness, kurtosis values at right top of the plot.

    Parameters
    ----------
    (**kwargs)
        arguments passed by `plot_distribution` or `seaborn.FacetGrid`: `data`, `bins`, `**kwargs`


    Returns
    ----------
    function
        Plotting functions to be used in `plot_distribution`

    See Also
    ----------
    frequentist_statistics_preparation.plot_distribution

    Examples
    ----------
    >>> tips = sns.load_dataset("tips").select_dtypes(include='number')
    >>> tips_melted = tips.melt(var_name='feature', value_name='values')
    >>> g = sns.FacetGrid(tips_melted, col="feature", sharex=False, sharey=True)
    >>> _ = (g.map(mapping_dist_mean_median, "values", bins=10)
    >>>    .set(ylim=(0, 1))
    >>>    .set_titles("{col_name}")
    >>>    .set_xlabels("")
    >>>    .fig.subplots_adjust(wspace=.2, hspace=.3))
    >>> # _ = plt.show()


    """
    # plot distribution
    ax = sns.distplot(x, **kwargs)

    # find min and max x in the plot to allign text
    min_x, max_x = plt.xlim()
    x_pos = max_x - (max_x - min_x) * 0.325

    # add mean (dashed line and text)
    _ = ax.twinx()
    _ = plt.axvline(x.mean(), color="k", linestyle="--")
    _ = plt.text(x_pos, 0.95, "Mean: {:.2f}".format(x.mean()))

    # add median (solid line and text)
    _ = ax.twinx()
    _ = plt.axvline(x.median(), color="k")
    _ = plt.text(x_pos, 0.90, "Median: {:.2f}".format(x.median()))

    # add skewness and kurtosis (text only)
    _ = plt.text(
        x_pos, 0.85, "Skewness: {:.2f}".format(stats.skew(x, axis=0, bias=True))
    )
    _ = plt.text(
        x_pos,
        0.80,
        "Kurtosis: {:.2f}".format(
            stats.kurtosis(x, axis=0, fisher=True, bias=False, nan_policy="propagate")
        ),
    )


def plot_distribution(
    data: pd.DataFrame, figsize=(11.7, 8.27), col_wrap=None, **kwargs
):
    r""" Plot distributions relative to numerical (ungrouped) features

    Wrapper of `seaborn.FacetGrid` that uses plotting function `dist_mean_median` to plot distributions and displays
    the moments of the distribution for numerical columns in `data`

    Parameters:
    ----------
    data : pandas.DataFrame
        Dataframe containing the features of interest
    figsize: tuple (default: (11.7, 8.27))
        Width and height of the figure in inches
    col_wrap: int or None (default: None)
        If int, number of subplots that are allowed in a single row

    Returns
    ----------
    distplot: Figure

    Examples
    ----------
    >>> tips = sns.load_dataset("tips")
    >>> _ = plot_distribution(tips)
    >>> # plt.show()

    """

    # consider only numerical features and build a new df with two columns: ['feature' | 'values']
    data_numerical = data.select_dtypes(include="number")
    data_numerical_m = data_numerical.melt(var_name="feature", value_name="values")

    # subplots for every numerical column
    distplot = sns.FacetGrid(
        data_numerical_m, col="feature", col_wrap=col_wrap, sharey=False, sharex=False
    )
    _ = plt.gcf().set_size_inches(figsize)

    for ax in distplot.axes.flatten():
        _ = ax.tick_params(labelbottom=True)

    # distplot with mean and median lines and values
    _ = (
        distplot.map(mapping_dist_mean_median, "values", bins=10, **kwargs)
        .set_titles("{col_name}")
        .set_xlabels("")
        .fig.subplots_adjust(wspace=0.2, hspace=0.3)
    )
    _ = plt.gcf().suptitle("")

    return distplot


def ecdf(data: pd.Series):
    r"""Computes emperical cumulative distribution function for a one-dimensional array of measurements.

    Parameters
    ----------
    data: pandas.Series
        series with measurements

    Returns
    ----------
    x: pandas.Series
        sorted data
    y: pandas.Series
        cumulative distribution

    Examples
    ----------
    >>> bills = sns.load_dataset("tips")['total_bill']
    >>> x, y = ecdf(bills)

    """
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, len(x) + 1) / n
    return x, y


def mapping_ecdf(x, **kwargs):
    r"""Adds emperical and theoretical distribution of the data

    Helper function (map) for plotting distributions, can be used with `compare_theoretical_ecdf`
    Adds solid line at theoretical distribution
    Adds points at emperical data

    Parameters
    ----------
    (**kwargs)
        arguments passed by `compare_theoretical_ecdf` or `seaborn.FacetGrid`: `data`, `**kwargs`


    Returns
    ----------
    function
        Plotting functions to be used in `compare_theoretical_ecdf`

    See Also
    ----------
    frequentist_statistics_preparation.compare_theoretical_ecdf

    Examples
    ----------
    >>> tips = sns.load_dataset("tips").select_dtypes(include='number')
    >>> tips_melted = tips.melt(var_name='feature', value_name='values')
    >>> g = sns.FacetGrid(tips_melted, col="feature", sharex=False, sharey=True)
    >>> _ = (g.map(mapping_ecdf, "values")
    >>>    .set(ylim=(0, 1))
    >>>    .set_titles("{col_name}")
    >>>    .set_xlabels("")
    >>>    .fig.subplots_adjust(wspace=.2, hspace=.3))
    >>> # _ = plt.show()

    """

    # Compute mean and standard deviation from data: mu, sigma
    mu = np.nanmean(x)
    sigma = np.nanstd(x)
    # Sample out of a normal distribution with this mu and sigma: samples
    samples = np.random.normal(mu, sigma, size=10000)
    # Get the CDF of the samples and of the data
    x_theor, y_theor = ecdf(pd.Series(samples))
    x_emp, y_emp = ecdf(x)

    # Plot the CDFs
    _ = plt.plot(x_theor, y_theor)
    _ = plt.plot(x_emp, y_emp, marker=".", linestyle="none")


def compare_theoretical_ecdf(data: pd.DataFrame, figsize=(11.7, 8.27), col_wrap=None):
    r"""Compare emperical distribution to the theoretical distribution based on mean and standard deviation
    Uses `mapping_ecdf` to plot the distributions and `ecdf` to calculate the distributions.

    Parameters
    ----------
    data: pandas.DataFrame
        Dataframe including numerical features for which distribution is checked
    figsize: tuple (default: (11.7, 8.27))
        Width and height of the figure in inches
    col_wrap: int or None (default: None)
        If int, number of subplots that are allowed in a single row

    Returns
    ----------
    ecdfplot: Figure
        Figure with emperical and theoretical distributions of all numerical variables in subplots

    Examples
    ----------
    >>> tips = sns.load_dataset("tips")
    >>> _ = compare_theoretical_ecdf(tips)

    """

    # consider only numerical features and build a new df with two columns: ['feature' | 'values']
    data_numerical = data.select_dtypes(include="number")
    data_numerical_m = data_numerical.melt(var_name="feature", value_name="values")

    # subplots for every numerical column
    ecdfplot = sns.FacetGrid(
        data_numerical_m, col="feature", col_wrap=col_wrap, sharex=False, sharey=False
    )
    _ = plt.gcf().set_size_inches(figsize)
    for ax in ecdfplot.axes.flatten():
        _ = ax.tick_params(labelbottom=True)

    # ecdf for every column
    _ = (
        ecdfplot.map(mapping_ecdf, "values")
        .set(ylim=(-0.01, 1.01))
        .set_titles("{col_name}")
        .set_xlabels("")
        .set_ylabels("CDF")
    )
    _ = plt.subplots_adjust(wspace=0.2, hspace=0.3, top=0.75)

    _ = plt.gcf().suptitle("Comparison between theoretical and empirical distributions")

    return ecdfplot


def normal_check(data: pd.DataFrame) -> pd.DataFrame:
    r"""Compare the distribution of numeric variables to a normal distribution using the Kolmogrov-Smirnov test

    Wrapper for `scipy.stats.kstest`: the empircal data is compared to a normally distributed variable with the
    same mean and standard deviation. A significant result (p < 0.05) in the goodness of fit test means that the
    data is not normally distributed.

    Parameters
    ----------
    data: pandas.DataFrame
        Dataframe including the columns of interest

    Returns
    ----------
    df_normality_check: pd.DataFrame
        Dataframe with column names, p-values and an indication of normality

    Examples
    ----------
    >>> tips = sns.load_dataset("tips")
    >>> df_normality_check = normal_check(tips)

    """
    # Select numeric columns only
    num_features = data.select_dtypes(include="number").columns.tolist()
    # Compare distribution of each feature to a normal distribution with given mean and std
    df_normality_check = data[num_features].apply(
        lambda x: stats.kstest(
            x.dropna(), stats.norm.cdf, args=(np.nanmean(x), np.nanstd(x)), N=len(x)
        )[1],
        axis=0,
    )

    # create a label that indicates whether a feature has a normal distribution or not
    df_normality_check = pd.DataFrame(df_normality_check).reset_index()
    df_normality_check.columns = ["feature", "p-value"]
    df_normality_check["normality"] = df_normality_check["p-value"] >= 0.05

    return df_normality_check


def apply_power_transformations(data: pd.DataFrame):
    r"""Transform data with log transformations and Box-cox transformations

    Columns that have values <=0 will be rescaled before transformations.
    Any `numpy.nans` will be ignored and retained. However, a warning 'invalid value encountered in less equal' is
    returned by `scipy.stats.boxcox` which can be ignored.
    Transformations
    - natural logarithm
    - base 10 logarithm
    - square root
    - inverse of square root
    - inverse
    - square

    Parameters
    ----------
    data: pandas.DataFrame
        Dataframe with the features to be transformed

    Returns
    ----------
    pandas.Dataframe
        Dataframe with original data and transformations. Abbrevations of the transformations are added to the feature
        names (i.e., ln, log10, box_sqrt, box_inv, box_square)

    Examples
    ----------
    >>> tips = sns.load_dataset("tips")
    >>> tips_transformed = apply_power_transformations(tips)

    """

    # Leave original data untouched
    data = data.copy()

    # Check whether features have negative values or zeros
    negative_values = data.apply(lambda x: any(x.dropna() <= 0))

    # Features with negative values
    features_any_negative = negative_values.loc[
        negative_values.isin([True])
    ].index.values
    # Features without any negative values
    features_only_positive = negative_values.loc[
        negative_values.isin([False])
    ].index.values

    if len(features_only_positive) > 0:
        # Apply natural log
        data["ln_" + features_only_positive] = data[features_only_positive].apply(
            np.log
        )
        # Apply log base 10
        data["log10_" + features_only_positive] = data[features_only_positive].apply(
            np.log10
        )
        # Apply Box-cox transformations
        data["box_sqrt_" + features_only_positive] = data[features_only_positive].apply(
            lambda x: stats.boxcox(x, lmbda=0.5, alpha=0.05)
        )
        data["box_sqrt_inv_" + features_only_positive] = data[
            features_only_positive
        ].apply(lambda x: stats.boxcox(x, lmbda=-0.5, alpha=0.05))
        data["box_inv_" + features_only_positive] = data[features_only_positive].apply(
            lambda x: stats.boxcox(x, lmbda=-1.0, alpha=0.05)
        )
        data["box_square_" + features_only_positive] = data[
            features_only_positive
        ].apply(lambda x: stats.boxcox(x, lmbda=2.0, alpha=0.05))

    if len(features_any_negative) > 0:
        # Apply natural log
        data["ln_" + features_any_negative] = data[features_any_negative].apply(
            lambda x: np.log(x + 1 - np.nanmin(x))
        )
        # Apply log base 10
        data["log10_" + features_any_negative] = data[features_any_negative].apply(
            lambda x: np.log10(x + 1 - np.nanmin(x))
        )
        # Apply Box-cox transformations
        data["box_sqrt_" + features_any_negative] = data[features_any_negative].apply(
            lambda x: stats.boxcox(x + 1 - np.nanmin(x), lmbda=0.5, alpha=0.05)
        )
        data["box_sqrt_inv_" + features_any_negative] = data[
            features_any_negative
        ].apply(lambda x: stats.boxcox(x + 1 - np.nanmin(x), lmbda=-0.5, alpha=0.05))
        data["box_inv_" + features_any_negative] = data[features_any_negative].apply(
            lambda x: stats.boxcox(x + 1 - np.nanmin(x), lmbda=-1.0, alpha=0.05)
        )
        data["box_square_" + features_any_negative] = data[features_any_negative].apply(
            lambda x: stats.boxcox(x + 1 - np.nanmin(x), lmbda=2.0, alpha=0.05)
        )

    return data


def find_optimal_transformation(
    normality_df: pd.DataFrame,
    feature_list: list,
    prefix_list=[
        "ln_",
        "log10_",
        "box_sqrt_",
        "box_sqrt_inv_",
        "box_inv_",
        "box_square_",
    ],
):
    r"""Choose the best transformation based on the results of `normal_check` after `apply_transformation`
    If no transformation returns a normal distribution, the original feature should be used with nonparametric methods

    Parameters:
    ----------
    df_normality_check: pandas.DataFrame
        Dataframe that includes feature (str), p-value (float) and normality (bool)
    feature_list: list
        List with the original feature names, corresponding to parts of the column names in df_normality_check (i.e.
        the original feature names)
    prefix_list: list
        List with the prefixes that were added to the original feature names in the feature_list

    Returns
    ----------
    use_transformation: list
        optimal transformation for each feature
    use_nonparametric: list
        features for which no appropriate transformation was available

    Examples
    ----------
    >>> tips = sns.load_dataset("tips")
    >>> #Assumption: all columns need transformations
    >>> tips_transformed = apply_power_transformations(tips.select_dtypes('number'))
    >>> df_normality_check = normal_check(tips_transformed)
    >>> use_transformation, use_nonparametric = find_optimal_transformation(df_normality_check,
    >>> tips.select_dtypes('number').columns)

    """

    # initialize lists
    use_transformation = []
    use_nonparametric = []

    for feature in feature_list:
        # Select feature and transformations
        search_strings = ["^" + feature + "$"] + [
            "^" + i + feature + "$" for i in prefix_list
        ]
        df_selection_feature = normality_df.loc[
            normality_df["feature"].str.match("|".join(search_strings))
        ]

        # If any of the transformations yielded a normal distribution, find the optimal transformation
        if df_selection_feature["normality"].isin([True]).any():
            # Select highest p-value (distribution )
            optimal_feature = df_selection_feature["feature"][
                df_selection_feature["p-value"].idxmax()
            ]
            use_transformation.append(optimal_feature)

        # If none of the transformations yielded a normal distribution, return the original feature
        else:
            use_nonparametric.append(feature)
    return use_transformation, use_nonparametric


def ols(
    X: pd.DataFrame,
    y: pd.Series,
    fit_intercept=True,
    normalize=False,
    impute=False,
    imputation_strategy="median",
    show=False,
    plot=False,
    figsize=(11.7, 8.27),
    cov_type="nonrobust",
    cov_kwds=None,
):
    r"""Run a linear regression model with ordinary least squares (OLS) estimation

    Parameters
    ----------
    X: pandas.DataFrame
        Dataframe with predictors
    y: pandas.Series
        Series with outcome
    fit_intercept:  bool
        whether to add an intercept to the model
    normalize: bool
        whether to mean center and scale the numerical predictors (z-scores)
    impute: bool
        whether to impute missing values (`np.nan`) in numerical predictors in X
    imputation_strategy: {'mean', 'median', 'most_frequent'} (default='median')
        imputation strategy when impute=True
    show: bool
        whether to print the main results
    plot: bool
        whether to show plots with model diagnostics
        if missing values in the data are not imputed, it is not possible to show model diagnostics
    figsize: tuple (default: (11.7, 8.27))
        Width and height of the figure in inches
    cov_type: str
        covariance estimator, see statsmodels.regression.linear_model.RegressionResults.get_robustcov_results for
        available covariance estimators
    cov_kwds: list or None, optional
        See statsmodels.regression.linear_model.RegressionResults.get_robustcov_results for a description of the
        required keyswords for alternative covariance estimators

    Returns
    ----------
    df_coef: pandas.DataFrame
        Dataframe with the results regarding coefficients: predictors, coefficients, standard errors, confidence
                                                        intervals, p-values
    df_fitting: pandas.DataFrame
        Dataframe with the results regarding model fit: outcome, number of observations, RMSE, F-value, df model,
                                                        df residuals, p-value, R-squared, adjusted R-squared
    diagplot: Figure
        Figure with model diagnostics if plot == True, else None

    Examples
    ----------
    >>> data = sns.load_dataset(name="mpg")
    >>> y = data['mpg']
    >>> X = data[['cylinders', 'displacement', 'weight', 'acceleration']]
    >>> coefs, fits, _ = ols(X, y, show=True, plot=True)

    """
    X = X.copy()
    y = y.copy()

    if len(X) != len(y):
        raise RuntimeError("Lengths of X and y differ")

    if not pd.api.types.is_numeric_dtype(y):
        raise RuntimeError("y is not numeric")

    # Missing values handling in y
    if y.isna().any():
        # Remove all observations that have missing y values
        warnings.warn(
            "WARNING: y included missing values. These observations are dropped in both y and X",
            stacklevel=2,
        )
        correct_y_indices = pd.notna(y)
        y = y[correct_y_indices]
        X = X[correct_y_indices]

    # Standardize predictors
    if normalize:
        # Split numerical and other predictors
        X_num = X.select_dtypes(include="number")
        X_cat = X.select_dtypes(exclude="number")

        # Normalize numerical predictors
        scaler = StandardScaler()
        column_names = X_num.columns
        index_names = X_num.index
        X_num = scaler.fit_transform(X_num)
        X_num = pd.DataFrame(X_num, columns=column_names, index=index_names)

        # Add non-numerical predictors back
        X = pd.concat([X_num, X_cat], axis=1)

    # Missing values handling in X
    if X.isna().any(axis=None):
        if impute:
            # Split numerical and other predictors
            X_num = X.select_dtypes(include="number")
            X_cat = X.select_dtypes(exclude="number")

            # Drop observations with missing values in non-numerical features
            if X_cat.isna().any(axis=None):
                warnings.warn(
                    "WARNING: X included missing values in non-numerical features. These observations are "
                    "dropped in both y and X",
                    stacklevel=2,
                )
                correct_X_indices = pd.notna(X_cat).all(axis=1)
                y = y[correct_X_indices]
                X = X[correct_X_indices]

                # Split numerical and other predictors
                X_num = X.select_dtypes(include="number")
                X_cat = X.select_dtypes(exclude="number")

            # Impute missing values in numerical features X with imputation_stratgey
            imputer = SimpleImputer(missing_values=np.nan, strategy=imputation_strategy)
            column_names = X_num.columns
            index_names = X_num.index
            X_num = imputer.fit_transform(X_num)
            X_num = pd.DataFrame(X_num, columns=column_names, index=index_names)

            # Add non-numerical predictors back
            X = pd.concat([X_num, X_cat], axis=1)

        else:
            # Remove all observations that have missing X values
            warnings.warn(
                "WARNING: X included missing values. These observations are dropped in both y and X",
                stacklevel=2,
            )
            correct_X_indices = pd.notna(X).all(axis=1)
            y = y[correct_X_indices]
            X = X[correct_X_indices]

    if fit_intercept:
        # Add an intercept
        X["intercept"] = 1  # use broadcast property

    # When non-numerical datatypes are present
    if not X.select_dtypes(exclude="number").empty:
        warnings.warn(
            "WARNING: X included non-numerical features. Dummy variables will be created",
            stacklevel=2,
        )
        for cat_column in X.select_dtypes(exclude="number").columns.tolist():
            # Make dummy variable
            X = pd.concat(
                (X, pd.get_dummies(X[cat_column], drop_first=True, prefix=cat_column)),
                axis=1,
            )
            X = X.drop(cat_column, axis=1)

    # Fit model
    result = sm.OLS(endog=y, exog=X, hasconst=fit_intercept).fit(
        cov_type=cov_type, cov_kwds=cov_kwds
    )

    # Calculate RMSE
    ypred = result.predict(X)
    rmse_model = rmse(y, ypred)

    # Wrap relevant OLS loadings info within a dataframe
    coefficient_values = result.params
    coefficient_values.name = "coef"

    p_values = pd.Series(
        data=result.pvalues, index=coefficient_values.index, name="p-value"
    )

    std_err = pd.Series(data=result.bse, index=coefficient_values.index, name="std-err")

    conf_int_min = pd.Series(
        data=result.conf_int().loc[:, 0], index=coefficient_values.index, name="[0.025"
    )

    conf_int_max = pd.Series(
        data=result.conf_int().loc[:, 1], index=coefficient_values.index, name="0.975]"
    )

    df_coef = pd.concat(
        [coefficient_values, std_err, conf_int_min, conf_int_max, p_values], axis=1
    )

    df_coef["stat-sign"] = df_coef["p-value"] <= 0.05

    # Wrap relevant model info within a dataframe

    # to complete using info from
    # https://towardsdatascience.com/how-do-you-check-the-quality-of-your-regression-model-in-python-fa61759ff685

    df_fitting = pd.DataFrame(
        {
            "target": y.name,
            "n-obs": int(result.nobs),
            "rmse": rmse_model,
            "f-value": result.fvalue,
            "df1": result.df_model,
            "df2": result.df_resid,
            "p-value": result.f_pvalue,
            "R-squared": result.rsquared,
            "Adj. R-squared": result.rsquared_adj,
        },
        index=[0],
    )

    if show:
        print(result.summary())
        pd.set_option("display.max_columns", 10)
        print(
            pd.DataFrame(
                {
                    "Normalized residuals": pd.Series(
                        result.get_influence().resid_studentized_internal
                    ).describe()
                }
            ).T
        )

    diagplot = None

    if plot:
        diagplot = diagnostic_plots(result, figsize=figsize)

    return df_coef, df_fitting, diagplot


def diagnostic_plots(model_fit, figsize=(11.7, 8.27)):
    r""" Function to reproduce the 4 diagnostic plots of an OLS model in R.

    Parameters
    ----------
    model_fit:
        statsmodels.regression.linear_model.RegressionResults obtained via `statsmodels.api.OLS.fit`
    figsize: tuple (default: (11.7, 8.27))
        Width and height of the figure in inches

    Returns
    ----------
    diagplot: matplotlib.figure.Figure
        Figure with four diagnostic plots: residuals vs fitted, QQplot, scale location, residuals vs leverage

    Examples
    ----------
    >>> import statsmodels.api as sm
    >>> data = sns.load_dataset(name="mpg")
    >>> y = data['mpg']
    >>> X = data[['cylinders', 'displacement', 'weight', 'acceleration']]
    >>> results = sm.OLS(endog=y, exog=X).fit()
    >>> _ = diagnostic_plots(results)
    >>> # plt.show()

    Notes
    ----------
    You might want to set `sns.set()` before running the function.

    """
    # model values
    model_fitted_y = model_fit.fittedvalues
    # model residuals
    model_residuals = model_fit.resid
    # absolute residuals
    model_abs_resid = np.abs(model_residuals)

    # normalized residuals
    model_norm_residuals = pd.DataFrame(
        {
            "model_norm_residuals": model_fit.get_influence().resid_studentized_internal,
            "index": model_residuals.index,
        }
    )
    # # Alternative type of residuals: pearson
    # model_norm_residuals = model_fit.resid_pearson
    # model_norm_residuals = pd.DataFrame({'model_norm_residuals': model_norm_residuals,
    #                                      'index': model_residuals.index})
    _ = model_norm_residuals.set_index("index", inplace=True)
    model_norm_residuals = model_norm_residuals["model_norm_residuals"]
    # root square absolute normalized residuals
    model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

    # leverage, from statsmodels internals
    model_leverage = pd.DataFrame(
        {
            "model_leverage": model_fit.get_influence().hat_matrix_diag.transpose(),
            "index": model_residuals.index,
        }
    )
    _ = model_leverage.set_index("index", inplace=True)
    model_leverage = model_leverage["model_leverage"]

    # create figure with 4 subplots
    diagplot, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    _ = plt.subplots_adjust(wspace=0.6, hspace=0.6)
    _ = plt.suptitle("Model diagnostics")

    # First plot: Residuals vs fitted
    _ = sns.regplot(
        model_fitted_y,
        model_residuals,
        scatter=True,
        lowess=True,
        line_kws={"color": "red", "lw": 1, "alpha": 0.8},
        scatter_kws={"color": "#3778bf", "alpha": 0.5},
        ax=diagplot.axes[0],
    )
    x_range = np.linspace(min(model_fitted_y), max(model_fitted_y), 50)
    _ = diagplot.axes[0].plot(
        x_range, np.repeat(0, len(x_range)), lw=1, ls=":", color="black"
    )
    _ = diagplot.axes[0].set_title("Residuals vs Fitted")
    _ = diagplot.axes[0].set_xlabel("Fitted values")
    _ = diagplot.axes[0].set_ylabel("Residuals")
    margin_res = 0.10 * (max(model_residuals) - min(model_residuals))
    _ = diagplot.axes[0].set_ylim(
        min(model_residuals) - margin_res, max(model_residuals) + margin_res
    )

    # annotations: top 3 absolute residuals
    abs_resid_top_3 = model_abs_resid.sort_values(ascending=False)[:3]
    for i in abs_resid_top_3.index:
        _ = diagplot.axes[0].annotate(i, xy=(model_fitted_y[i], model_residuals[i]))

    # Second plot: QQ graph
    QQ = ProbPlot(model_norm_residuals)
    _ = QQ.qqplot(line="s", alpha=0.5, color="#3778bf", lw=1, ax=diagplot.axes[1])
    _ = diagplot.axes[1].get_lines()[1].set_markerfacecolor("r")
    _ = diagplot.axes[1].set_title("Normal Q-Q")
    _ = diagplot.axes[1].set_xlabel("Theoretical Quantiles")
    _ = diagplot.axes[1].set_ylabel("Standardized Residuals")
    # annotations: top 3 absolute normalized residuals
    abs_norm_resid_top_3 = np.abs(model_norm_residuals).sort_values(ascending=False)[:3]
    ordered_norm_residuals = pd.DataFrame(
        {"model_norm_residuals": model_norm_residuals}
    ).sort_values(by="model_norm_residuals")
    _ = ordered_norm_residuals.reset_index(inplace=True)
    for i in abs_norm_resid_top_3.index:
        index = np.where(ordered_norm_residuals["index"] == i)[0][0]
        _ = diagplot.axes[1].annotate(
            i, xy=(QQ.theoretical_quantiles.transpose()[index], model_norm_residuals[i])
        )

    # Third plot: scale location
    _ = sns.regplot(
        model_fitted_y,
        model_norm_residuals_abs_sqrt,
        scatter=True,
        ci=False,
        lowess=True,
        line_kws={"color": "red", "lw": 1, "alpha": 0.8},
        scatter_kws={"color": "#3778bf", "alpha": 0.5},
        ax=diagplot.axes[2],
    )
    _ = diagplot.axes[2].set_title("Scale-Location")
    _ = diagplot.axes[2].set_xlabel("Fitted values")
    _ = diagplot.axes[2].set_ylabel(r"$\sqrt{|Standardized Residuals|}$")
    # annotations: top 3 absolute normalized residuals
    for i in abs_norm_resid_top_3.index:
        _ = diagplot.axes[2].annotate(
            i, xy=(model_fitted_y[i], model_norm_residuals_abs_sqrt[i])
        )

    # Fourth plot: residuals vs leverages
    _ = sns.regplot(
        model_leverage,
        model_norm_residuals,
        scatter=True,
        ci=False,
        lowess=True,
        line_kws={"color": "red", "lw": 1, "alpha": 0.8},
        scatter_kws={"color": "#3778bf", "alpha": 0.5},
        ax=diagplot.axes[3],
    )
    _ = diagplot.axes[3].set_xlim(0, max(model_leverage) + 0.01)
    _ = diagplot.axes[3].set_ylim(
        min(model_norm_residuals) - 0.5, max(model_norm_residuals) + 0.5
    )
    _ = diagplot.axes[3].set_title("Residuals vs Leverage")
    _ = diagplot.axes[3].set_xlabel("Leverage")
    _ = diagplot.axes[3].set_ylabel("Standardized Residuals")
    # annotations: top 3 levarages
    leverage_top_3 = model_leverage.sort_values(ascending=False)[:3]
    for i in leverage_top_3.index:
        _ = diagplot.axes[3].annotate(
            i, xy=(model_leverage[i], model_norm_residuals[i])
        )
    # extra lines to indicate Cook's distances
    p = len(model_fit.params)  # number of model parameters
    x_range = np.linspace(0.001, max(model_leverage), 50)

    def cooksdistances(boundary):
        return lambda x: np.sqrt((boundary * p * (1 - x)) / x)

    for line in [0.5, 1]:
        l_formula = cooksdistances(line)
        for place in [1, -1]:
            cooks_line = plt.plot(
                x_range, place * l_formula(x_range), lw=1, ls="--", color="red"
            )
            y_text = place * l_formula(max(model_leverage) + 0.01)
            if (
                min(model_norm_residuals) - 0.5
                < y_text
                < max(model_norm_residuals) + 0.5
            ):
                _ = plt.text(max(model_leverage) + 0.01, y_text, str(line), color="red")
    _ = diagplot.axes[3].legend(
        cooks_line[:2], ["Cook's distance"], handlelength=3, loc="upper right"
    )

    return diagplot


def correct_pvalues(
    pvals,
    alpha: float = 0.05,
    method: str = "fdr_bh",
    plot: bool = False,
    labels=None,
    title: str = "",
    figsize: tuple = (10, 5),
):
    r"""
    Correct an array-like with pvalues using `method`, wrapper for `statsmodels.stats.multitest.multipletests`

    Parameters
    ----------
    pvals: array-like, 1d
        uncorrected pvalues
    alpha: float
        FWER, family-wise error rate
    method: str, one of {'bonferroni', 'sidak', 'holm-sidak', 'holm', 'simes-hochberg', 'hommel', 'fdr_bh',
    'fdr_by', 'fdr_tsbh', 'fdr_tsbky'}
    plot: bool
        whether to plot the results
    title: str
        title to show above the plot
    labels: array-like, 1d
        labels for the uncorrected pvalues
    figsize: tuple
        size for the Figure

    Returns
    ----------
    reject: numpy.array, bool
        true for hypothesis that can be rejected for given alpha
    corrected_p: numpy.array
        p-values corrected for multiple tests
    pvalues_plot: matplotlib.figure.Figure (optional)
        Figure if plot == True, else None

    """

    if isinstance(pvals, pd.Series):
        pvals = pvals.values

    if labels is not None:
        if len(pvals) != len(labels):
            raise ValueError("Lengths of the pvals and the pvals_labels does not match")
        if isinstance(labels, pd.Series):
            labels = labels.values

    reject, corrected_p, _, _ = multipletests(
        pvals=pvals, alpha=alpha, method=method, returnsorted=True
    )

    # Sort the pvalues and the labels (correct pvalues are sorted already)
    sort_order = pvals.argsort()
    pvals = pvals[sort_order]
    if labels is not None:
        labels = labels[sort_order]
        labels = np.insert(labels, [0], [""])

    # Get colors for all pvalues
    colors = ["#2167C5" if i else "#EB5E23" for i in reject]

    pvalues_plot = None

    if plot:
        pvalues_plot, ax = plt.subplots(figsize=figsize)
        x = 1
        # Plot pvalues and corrected pvalues, color dependent on 'reject'
        for p, cp, c in zip(pvals, corrected_p, colors):
            _ = plt.plot(x, p, "o", c=c)
            _ = plt.plot(x, cp, "x", c=c)
            x += 1

        # Variable for pvalues length and number of pvalue
        n = len(pvals)
        i = np.arange(n) + 1

        # Plot line at familywise p value
        familywise_p = np.repeat(alpha, n)
        _ = plt.plot(i, familywise_p, "k--")

        # Add legend elements
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="k",
                label="Original p-values",
                linestyle="none",
            ),
            Line2D(
                [0],
                [0],
                marker="x",
                color="k",
                label="Corrected p-values",
                linestyle="none",
            ),
            Line2D(
                [0],
                [0],
                marker="X",
                color="#EB5E23",
                label="Non-significant",
                linestyle="none",
            ),
            Line2D(
                [0],
                [0],
                marker="X",
                color="#2167C5",
                label="Significant",
                linestyle="none",
            ),
            Line2D(
                [0], [0], marker="", color="k", label=f"Alpha = {alpha}", linestyle="--"
            ),
        ]

        if method == "fdr_bh":
            # Plot a diagonal line to show the boundary pvalue
            optimum_p = alpha * i / n
            _ = plt.plot(i, optimum_p, "k-")
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="",
                    color="k",
                    label="Benjamini-Hochberg decision line",
                    linestyle="-",
                )
            )

        # Add labels and legend
        _ = plt.xlabel("$i$")
        _ = plt.ylabel("$p$")
        _ = plt.title(title)

        _ = ax.legend(handles=legend_elements)

        if labels is not None:
            _ = plt.xticks(plt.xticks()[0], labels)

    return reject, corrected_p, pvalues_plot


def bland_altman_plot(
    data: pd.DataFrame,
    column1: str,
    column2: str,
    hue_column: str = None,
    sd_limit: float = 1.96,
    scatter_kwds: dict = dict(),
    mean_line_kwds: dict = dict(),
    limit_lines_kwds: dict = dict(),
    plt_title: str = None,
    figsize: tuple = (12, 7),
):
    r"""Construct a Bland-Altman Mean Difference Plot.

    The Bland-Altman Plot is a graphical method to analyze
    the differences between two methods of measurement.
    It is a way of quantifying the agreement of two clinical measures.
    The mean of the measures is plotted against their difference.

    For more information see
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4470095/

    Parameters
    ----------

    data: pd.DataFrame
        The data frame from which data will be used for the plot.
    column1: str
        The name of the column of the first measure.
    column2: str
        The name of the column of the second measure.
    hue_column: str
        The name of the column used to color groups in the Bland Altman Plot.
    sd_limit: float
        The limit of agreements expressed in terms of the standard deviation of
        the differences. If `md` is the mean of the differences, and `sd` is
        the standard deviation of those differences, then the limits of
        agreement that will be plotted are md +/- sd_limit * sd.
        The default of 1.96 will produce 95% confidence intervals for the means
        of the differences. If sd_limit = 0, no limits will be plotted, and
        the ylimit of the plot defaults to 3 standard deviations on either
        side of the mean.
    scatter_kwds: dict
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.scatter plotting method.
    mean_line_kwds: dict
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.axhline plotting method.
    limit_lines_kwds: dict
        Options to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.axhline plotting method.
    plt_title: str
        An optional string to overwrite the default title.
    figsize: tuple
        A tuple of numbers to be passed to the plt.figure(figsize= ) function
        to manually specify the size of the plot.

    Returns
    -------
    fig: matplotlib.figure.Figure
        The matplotlib.figure.Figure object that is shown when running the function
    ax: matplotlib.axes._subplots.AxesSubplot
        The associated matplotlib.axes._subplots.AxesSubplot object of the plotted figure
    means: pd.Series
        The series of means that are plotted
    diffs: pd.Series
        The series of differences that are plotted

    References
    ----------
    Bland JM, Altman DG (1986). "Statistical methods for assessing agreement
    between two methods of clinical measurement"
    Giavarina D. (2015). Understanding Bland Altman analysis. Biochemia medica, 25(2), 141–151. https://doi.org/10.11613/BM.2015.015

    Examples
    --------

    >>> # Load relevant libraries.
    >>> from neuropy.frequentist_statistics import bland_altman_plot
    >>> import numpy as np
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt

    >>> # Making a bland-altman difference plot.
    >>> # Seed the random number generator.
    >>> # This ensures that the results below are reproducible.
    >>> np.random.seed(69420)
    >>> measure_1 = np.random.random(100)
    >>> measure_2 = np.random.random(100)
    >>> other_group_difference = np.random.choice([0, 1], 100)

    >>> df = pd.DataFrame({"m1": measure_1, "m2": measure_2, "hue_column": other_group_difference})

    >>> # Example 1: plotting the bland altman plot whilst incorporating a hue column
    >>> fig, ax, means, diffs = bland_altman_plot(data=df, column1="m1", column2="m2", hue_column="hue_column", sd_limit=1.96,
    >>>                                              scatter_kwds=None, mean_line_kwds=None, limit_lines_kwds=None,
    >>>                                              plt_title="Bland Altman plot of \n two different measures",
    >>>                                              figsize=(7,5))
    >>> _ = plt.show()
    >>> # Example 2: plotting the bland altman plot whilst incorporating specific keywords for the scatter plot
    >>> fig, ax, means, diffs = bland_altman_plot(data=df, column1="m1", column2="m2",  hue_column=None,
    >>>                                           sd_limit=1.96, scatter_kwds={'color':'navy'},
    >>>                                           mean_line_kwds=None, limit_lines_kwds=None,
    >>>                                           plt_title="Bland Altman plot of \n two different measures",
    >>>                                           figsize=(7,5))
    >>> _ = plt.show()

    """

    if not isinstance(data, pd.DataFrame):
        raise ValueError("data is not a valid pd.DataFrame object.")
    if (data.count().to_numpy()[0] != data.count().to_numpy()).any():
        sample_size_num = data.count().min()
        sample_size_col = data.count()[
            data.count() == data.count().min()
        ].index.tolist()[0]
        warnings.warn(
            f"The amount of data in the columns is not equal across columns, hence the amount of data"
            f" shown in the plot is equal to {sample_size_num}, column == {sample_size_col}"
        )
    elif data.isna().any().any():
        warnings.warn(
            f"There is missing data in the columns, hence the amount of data"
            f" shown in the plot is equal to {data.count().min()}"
        )
    if sd_limit < 0:
        raise ValueError(f"sd_limit ({sd_limit}) is less than 0.")

    means = data[[column1, column2]].mean(axis=1, skipna=False)
    diffs = data[column2] - data[column1]
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, axis=0)

    if "s" not in scatter_kwds:
        scatter_kwds["s"] = 20
    for kwds in [mean_line_kwds, limit_lines_kwds]:
        if "color" not in kwds:
            kwds["color"] = "gray"
        if "linewidth" not in kwds:
            kwds["linewidth"] = 1
    if "linestyle" not in mean_line_kwds:
        kwds["linestyle"] = "--"
    if "linestyle" not in limit_lines_kwds:
        kwds["linestyle"] = ":"

    fig, ax = plt.subplots(figsize=figsize)
    if hue_column is not None:
        _ = sns.scatterplot(
            x=means, y=diffs, hue=data[hue_column], ax=ax, **scatter_kwds
        )
        _ = ax.legend(title=hue_column, loc="lower left")
    else:
        _ = sns.scatterplot(x=means, y=diffs, ax=ax, **scatter_kwds)

    _ = ax.axhline(mean_diff, **mean_line_kwds)  # draw mean line.

    props = dict(boxstyle="square", facecolor="white", alpha=0.5)

    # Annotate mean line with mean difference.
    _ = ax.annotate(
        f"mean\ndifference:\n{mean_diff:.2f}",
        xy=(0.975, 0.5),
        horizontalalignment="right",
        verticalalignment="center",
        fontsize=14,
        xycoords="axes fraction",
        bbox=props,
    )

    if sd_limit > 0:
        half_ylim = (1.5 * sd_limit) * std_diff
        _ = ax.set_ylim(mean_diff - half_ylim, mean_diff + half_ylim)
        limit_of_agreement = sd_limit * std_diff
        lower = mean_diff - limit_of_agreement
        upper = mean_diff + limit_of_agreement

        for sign, lim, pos in zip(["-", "+"], [lower, upper], [0.07, 0.92]):
            _ = ax.axhline(lim, **limit_lines_kwds)
            _ = ax.annotate(
                f"{sign}SD({sd_limit}): {lim:.2f}",
                xy=(0.975, pos),
                horizontalalignment="right",
                verticalalignment="bottom",
                fontsize=14,
                xycoords="axes fraction",
                bbox=props,
            )

    else:
        half_ylim = 3 * std_diff
        _ = ax.set_ylim(mean_diff - half_ylim, mean_diff + half_ylim)

    if plt_title is None:
        _ = ax.set_title("Bland Altman plot of two timepoints", fontsize=15)
    else:
        _ = ax.set_title(plt_title, fontsize=15)

    _ = ax.set_ylabel("Difference", fontsize=15)
    _ = ax.set_xlabel("Means", fontsize=15)
    _ = ax.tick_params(labelsize=13)
    _ = fig.tight_layout()
    return fig, ax, means, diffs

def robust_rfecv(  # noqa: C901
    X,
    y,
    model_list,
    step=1,
    cv=KFold(3),
    scoring=None,
    preprocessing_pipe=None,
    groups=None,
    show=False,
    plot=True,
    njobs=-4,
    **kwargs,
):
    r"""Run Recursive Feature Elimination (RFE) using multiple Machine learning models.
        The function is suitable for both classification and regression problem.
    Parameters
    ----------
    X: pd.DataFrame
        Contains training vectors.
    y: pd.Series
        Target values.
    model_list: List
        Estimators that have either coef. or feature importance. object.
    step: int, optional (default=1)
        It refers to the (integer) number of features to remove at each iteration.
    cv: cross-validation generator (default: KFold(3))
        Determines the cross-validation splitting strategy.
        For more details see https://scikit-learn.org/stable/modules/cross_validation.html
    scoring: string, callable or None, optional
        A string or a scorer callable object / function with signature.
        For more details see https://scikit-learn.org/stable/modules/model_evaluation.html
    preprocessing_pipe: Pipeline object, or Nones
        Preprocessing steps needed before the estimator (e.g. StandardScaler)
    groups: pd.Series, or Nones
        Group labels for the samples used while splitting the dataset into train/test set.
        Only used in conjunction with a “Group” cv instance (e.g., GroupKFold).
        For more details see https://scikit-learn.org/stable/modules/cross_validation.html
        section 3.1.2.2. Cross-validation iterators with stratification based on class labels
    show: Boolean
        Whether to print the results.
    plot: Boolean
        Whether to plot the rfe summary plot.
    njobs: int
        The amount of cores to use when running the RFECV worker function.
    kwargs: key, value pairings
        Additional keyword arguments for plot setting matplotlib style. Available setting:
            figsize  : tuple (default: (7, 5))
            textsize : int (default: 15) adjust size for xticks label, title and legend
            xlim : tuple. Set the x limits of the current axes.
            ylim : tuple. Set the y limits of the current axes.
    Returns
    -------
    df_summary: pd.DataFrame
        Dataframe which show they number of optimal feature, score value and score type e.g., r2, auc.
    df_feature:
        DataFrame containing the selected features for each ML model and the corresponding intersection
    fig: matplotlib.figure.Figure
        matplotlib object
    Displays
    --------
    - results will be printed if show = True
    - plots will be printed if plot = True
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    >>> from neuropy.dimensionality_reduction import robust_rfecv
    >>> from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.impute import SimpleImputer
    >>> # load dataset
    >>> cancer = load_breast_cancer()
    >>> df = pd.DataFrame(data=np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))
    >>> X = df.drop(columns=['target'])
    >>> y = df['target']
    >>> # prepare pipeline and classifies
    >>> pipe = Pipeline([
    ...    ('inputer', SimpleImputer(strategy='median')),
    ...    ('scaler', StandardScaler())
    ... ])
    >>> clf = [AdaBoostClassifier(n_estimators=20), RandomForestClassifier(n_estimators=20,max_depth=5), LinearDiscriminantAnalysis()]
    >>> df_summary, df_feature, fig = robust_rfecv(X, y, model_list=clf, preprocessing_pipe=pipe, scoring='roc_auc')
    """

    X = X.copy()
    y = y.copy()

    if not model_list:
        raise AttributeError("No ML model found as input.")

    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "X should be of type pd.DataFrame, currently the function cannot handle np.array"
        )

    # detect if regression of classification problem based on the type of model
    check_model_type = pd.DataFrame(columns=["clf", "is_classifier", "is_regressor"])
    for m in model_list:
        check_model_type = check_model_type.append(
            {
                "clf": m.__class__.__name__,
                "is_classifier": is_classifier(m),
                "is_regressor": is_regressor(m),
            },
            ignore_index=True,
        )
    if (
        len(check_model_type["is_classifier"])
        == check_model_type["is_classifier"].sum()
    ):
        # all model are classifiers
        if scoring is None:
            scoring = "roc_auc"  # assign default scoring
        print(
            f"rfecv for classification problem with {len(model_list)} ML models with {scoring} score"
        )
        title_plot = "Recursive Feature Elimination (Classification)"
    elif (
        len(check_model_type["is_regressor"]) == check_model_type["is_regressor"].sum()
    ):
        # all model are regressors
        if scoring is None:
            scoring = "neg_root_mean_squared_error"  # assign default scoring
        print(
            f"rfecv for regression problem with {len(model_list)} ML model mo with {scoring} score"
        )
        title_plot = "Recursive Feature Elimination (Regression)"
    else:
        raise ValueError(
            "'model_list' can contain either only classifiers or only regressors"
        )

    if plot:
        # set figure dimension based on kwargs
        if "figsize" in kwargs.keys():
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=kwargs["figsize"])
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

        # set figure dimension based on kwargs
        if "textsize" in kwargs.keys():
            textsize = kwargs["textsize"]
        else:
            textsize = 10

    else:
        fig = None

    scores = []
    features = []
    n_features = []
    classifier_name = []

    if preprocessing_pipe:
        pipe_names_list = (
            np.array(
                [
                    [x.lower(), y.__class__.__name__.lower()]
                    for x, y in preprocessing_pipe.steps
                ]
            )
            .flatten()
            .tolist()
        )

        if any("impute" in s for s in pipe_names_list):
            X = pd.DataFrame(preprocessing_pipe.fit_transform(X), columns=X.columns)
        else:
            X = pd.DataFrame(
                preprocessing_pipe.fit_transform(X.dropna(axis=0)),
                columns=X.dropna(axis=0).columns,
            )

    if X.shape != X.dropna(axis=0).shape:
        X = X.dropna(axis=0)
        warnings.warn(
            f"""The dataframe passed as X included nan's and there was either no pipe passed or the pipe
                      did not include an imputer, therefore the nan's have been dropped row wise. The shape of
                      the resulting data frame is {X.shape}"""
        )

    # select the same rows in y that are in X
    y = y[X.index]

    for clf in model_list:
        # extract the name of the classifier
        name = clf.__class__.__name__
        classifier_name.append(name)

        # rfecv for model_list
        rfecv = RFECV(estimator=clf, step=step, cv=cv, scoring=scoring, n_jobs=njobs)

        _ = rfecv.fit(X, y, groups=groups)

        # make list from chosen features
        features_rfe = [f for f, s in zip(X.columns, rfecv.support_) if s]
        features.append(features_rfe)

        # make list from best grid scores
        score = round(np.mean(rfecv.grid_scores_), 2)
        scores.append(score)

        # make list from optimal n. of features
        n_feature = rfecv.n_features_
        n_features.append(n_feature)

        if plot:

            try:
                x_range = np.arange(X.shape[1], 0, -step)[::-1]
                plt.plot(
                    x_range, rfecv.grid_scores_, label=name, marker="o", markersize=4
                )
            except ValueError:
                x_range = np.arange(X.shape[1], 0, -step)
                x_range = np.append(x_range, 1)[::-1]
                plt.plot(
                    x_range, rfecv.grid_scores_, label=name, marker="o", markersize=4
                )

    df_summary = pd.DataFrame(
        data={
            "optimal_n_features": np.array(n_features).astype(int),
            "highest_performance": np.array(scores),
            "score": scoring,
        },
        index=[classifier_name],
    ).sort_values(by=["highest_performance"], ascending=False)

    # compute intersection features
    classifier_name.append("intersection")
    intersection = set(features[0]).intersection(*features)
    features.append(list(intersection))

    # create dataframe which contains the feature list
    dict_feature = dict()
    for name_col, feature_col in zip(classifier_name, features):
        feature_col.sort()  # arrange values alphabetically
        dict_feature[name_col] = feature_col
    df_feature = pd.DataFrame(
        dict([(k, pd.Series(v)) for k, v in dict_feature.items()])
    )

    if plot:
        _ = ax.set_xlabel("Number of features selected")
        _ = ax.set_ylabel(f"cv mean {scoring}")
        _ = ax.set_title(title_plot)
        _ = ax.legend()
        _ = plt.setp(ax.get_xticklabels(), fontsize=textsize)
        _ = plt.setp(ax.get_yticklabels(), fontsize=textsize)
        _ = ax.yaxis.label.set_size(textsize)
        _ = ax.xaxis.label.set_size(textsize)
        _ = ax.title.set_size(textsize + 2)
        if "xlim" in kwargs.keys():
            plt.xlim(kwargs["xlim"])
        if "ylim" in kwargs.keys():
            plt.ylim(kwargs["ylim"])

        ymin, ymax = plt.ylim()
        _ = plt.axvline(
            x=len(df_feature["intersection"].dropna()),
            linestyle="-.",
            color="grey",
            linewidth=1.5,
            ymin=ymin,
            ymax=ymax,
            label="optimal number of features",
        )  #
        _ = ax.legend(loc="lower right", fontsize=textsize - 1)

    if show:
        display(df_summary)

    return df_summary, df_feature, fig