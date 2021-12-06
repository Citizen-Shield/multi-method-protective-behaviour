"""Submodule machine_learning_classification.py includes the following functions:
  - plot_decision_boundary():  Generate a simple plot of the decision boundary of a classifier.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.metrics import (accuracy_score,
                             confusion_matrix,
                             f1_score,
                             roc_auc_score,
                             roc_curve,
                             auc
                             )

# set plotting options
cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm

def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=y, marker='_', lw=lw, cmap=cmap_data)

    ax.scatter(range(len(X)), [ii + 2.5] * len(X),
               c=group, marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_splits)) + ['class', 'group']
    ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits+2.2, -.2], xlim=[0, len(X)])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax

def plot_decision_boundary(
    X: pd.DataFrame,
    y: pd.Series,
    clf,
    title: str,
    legend_title: str,
    h=0.05,
    figsize=(11.7, 8.27),
):
    """
    Generate a simple plot of the decision boundary of a classifier.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Classifier vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples)
        Target relative to X for classification. Datatype should be integers.
    clf : scikit-learn algorithm
        An object that has the `predict` and `predict_proba` methods
    h : int (default: 0.05)
        Step size in the mesh
    title : string
        Title for the plot.
    legend_title : string
        Legend title for the plot.
    figsize: tuple (default: (11.7, 8.27))
        Width and height of the figure in inches
    Returns
    ----------
    boundaries: Figure
        Properties of the figure can be changed later, e.g. use `boundaries.axes[0].set_ylim(0,100)` to change ylim
    Examples
    ----------
    >>> import seaborn as sns
    >>> from sklearn.svm import SVC
    >>> data = sns.load_dataset("iris")
    >>> # convert the target from string to category to numeric as sklearn cannot handle strings as target
    >>> y = data["species"]
    >>> X = data[["sepal_length", "sepal_width"]]
    >>> clf = SVC(kernel="rbf", gamma=2, C=1, probability=True)
    >>> _ = plot_decision_boundary(X=X, y=y, clf=clf, title = 'Decision Boundary', legend_title = "Species")
    >>> # plt.show()
    """

    if X.shape[1] != 2:
        raise ValueError("X must contains only two features.")

    if not (
        pd.api.types.is_integer_dtype(y)
        or pd.api.types.is_object_dtype(y)
        or pd.api.types.is_categorical_dtype(y)
    ):
        raise TypeError(
            "The target variable y can only have the following dtype: [int, object, category]."
        )

    label_0 = X.columns.tolist()[0]
    label_1 = X.columns.tolist()[1]
    
    X = X.copy()
    y = y.copy()

    X = X.values
    y = y.astype("category").cat.codes.values
    
#     full_col_list = list(sns.color_palette("husl", len(np.unique(y))))
    full_col_list = list(sns.color_palette())
    
    if len(np.unique(y)) > len(full_col_list):
        raise ValueError(
            "More labels in the data then colors in the color list. Either reduce the number of labels or expend the color list"
        )
        
    sub_col_list = full_col_list[0 : len(np.unique(y))]
    cmap_bold = ListedColormap(sub_col_list)

    # Try to include a mapping in a later release (+ show categorical labels in the legend)

    _ = clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    Z_proba = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z_max = Z_proba.max(axis=1)  # Take the class with highest probability
    Z_max = Z_max.reshape(xx.shape)

    # Put the result into a color plot
    boundaries, ax = plt.subplots(figsize=figsize)
    _ = ax.contour(xx, yy, Z, cmap=cmap_bold)
    _ = ax.scatter(
        xx, yy, s=(Z_max ** 2 / h), c=Z, cmap=cmap_bold, alpha=1, edgecolors="none"
    )

    # Plot also the training points
    training = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolors="black")
    _ = plt.xlim(xx.min(), xx.max())
    _ = plt.ylim(yy.min(), yy.max())
    _ = plt.title(title)
    _ = plt.subplots_adjust(right=0.8)
    _ = plt.xlabel(label_0)
    _ = plt.ylabel(label_1)

    # Add legend colors
    leg1 = plt.legend(
        *training.legend_elements(),
        frameon=False,
        fontsize=12,
        borderaxespad=0,
        bbox_to_anchor=(1, 0.5),
        handlelength=2,
        handletextpad=1,
        title=legend_title,
    )

    # Add legend sizes
    l1 = plt.scatter([], [], c="black", s=0.4 ** 2 / h, edgecolors="none")
    l2 = plt.scatter([], [], c="black", s=0.6 ** 2 / h, edgecolors="none")
    l3 = plt.scatter([], [], c="black", s=0.8 ** 2 / h, edgecolors="none")
    l4 = plt.scatter([], [], c="black", s=1 ** 2 / h, edgecolors="none")

    labels = ["0.4", "0.6", "0.8", "1"]
    _ = plt.legend(
        [l1, l2, l3, l4],
        labels,
        frameon=False,
        fontsize=12,
        borderaxespad=0,
        bbox_to_anchor=(1, 1),
        handlelength=2,
        handletextpad=1,
        title="Probabilities",
        scatterpoints=1,
    )
    plt.gca().add_artist(leg1)

    return boundaries
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Import libraries necessary for functions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
import seaborn as sns
#from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import learning_curve

# Import a list of models you would like to use
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from sklearn import metrics
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.feature_selection import RFE

# set plotting options
cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm

def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=y, marker='_', lw=lw, cmap=cmap_data)

    ax.scatter(range(len(X)), [ii + 2.5] * len(X),
               c=group, marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_splits)) + ['class', 'group']
    ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits+2.2, -.2], xlim=[0, len(X)])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax

def plot_learning_curve(estimator, title, X, y, groups, cross_color, test_color, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), figsize=(7,5)):
    """
    Generate a simple plot of the test and training learning curve.
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    title : string
        Title for the chart.
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
        
    cross_color : string
        Signifies the color of the cross validation in the plot
        
    test_color : string
        Signifies the color of the test set in the plot
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    fig = plt.figure(figsize=figsize)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, groups=groups, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, random_state=42)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color=test_color)
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color=cross_color)
    plt.plot(train_sizes, train_scores_mean, 'o-', color=test_color,
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color=cross_color,
             label="Cross-validation score")

    plt.legend(loc="best")
    return fig

# create a dictionary of models
dict_of_models = [
{
    'label': 'Logistic Regression',
    'model': LogisticRegression(solver="lbfgs"),
},
{
    'label': 'Gradient Boosting',
    'model': GradientBoostingClassifier(),
},
{
    'label': 'K_Neighbors Classifier',
     'model': KNeighborsClassifier(3),
},
{
    'label': 'SVM Classifier (linear)',
     'model': SVC(kernel="linear", C=0.025, probability=True),
},
{
    'label': 'SVM Classifier (Radial Basis Function; RBF)',
     'model': SVC(kernel="rbf", gamma=2, C=1, probability=True),
},
{
    'label': 'Gaussian Process Classifier',
     'model': GaussianProcessClassifier(1.0 * RBF(1.0)),
},
{
    'label': 'Decision Tree (depth=5)',
     'model': DecisionTreeClassifier(max_depth=5),
},
{
    'label': 'Random Forest Classifier(depth=5)',
     'model': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
},
{
    'label': 'Multilayer Perceptron (MLP) Classifier',
     'model': MLPClassifier(alpha=1, max_iter=1000),
},
{
    'label': 'AdaBoost Classifier',
     'model': AdaBoostClassifier(),
},
{
    'label': 'Naive Bayes (Gaussian) Classifier',
     'model': GaussianNB(),
},
{
    'label': 'Quadratic Discriminant Analysis Classifier',
    'model': QuadraticDiscriminantAnalysis(),
}
]

# # create a plot_decision boundary function
# def plot_decision_boundary(X, y, clf, title, legend_title):
#     h = .05 # step size in the mesh
    
#     # Create color map
#     full_col_list = ["#F0871B","#2167C5","#308EC5","#F5A75D","#EB5E23","#E2E1E1","#9B9B9B","#4A4A4A","#FFFFFF","#F0F0F0","#2768C1"]
#     #full_col_list = ['#FF0000', '#00FF00', '#0000FF', '#FF8000', '#FF0080']
#     sub_col_list = full_col_list[0:len(np.unique(y))]
#     cmap_bold = ListedColormap(sub_col_list)
#     #cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FF8000', '#FF0080'])

#     # we create an instance of Neighbours Classifier and fit the data.
#     #clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
#     # create a numeric outcome measure if the data type is categorical
# #     if y.dtype == "category":
# #         y_cat_codes = y.cat.codes.values
# #     else:
# #         y_cat_codes = y
#     y_cat_codes = y
#     clf.fit(X, y_cat_codes)

#     # Plot the decision boundary. For that, we will assign a color to each
#     # point in the mesh [x_min, x_max]x[y_min, y_max].
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                          np.arange(y_min, y_max, h))
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     Z_proba = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()]) #NEED TO FIND THE PROBA FOR PREDICTED CLASS ONLY
#     Z_max = Z_proba.max(axis=1) # Take the class with highest probability
#     Z_max = Z_max.reshape(xx.shape)

#     # Put the result into a color plot
#     #fig, ax = plt.subplots()
#     plt.figure(figsize=(10,8))
#     plt.contour(xx, yy, Z, cmap=cmap_bold)
#     plt.scatter(xx, yy, s=(Z_max**2/h), c=Z, cmap=cmap_bold, alpha=0.75, edgecolors='none')

#     # Plot also the training points
#     a = plt.scatter(X[:, 0], X[:, 1], c=y_cat_codes, cmap=cmap_bold, edgecolors='black')
#     plt.xlim(xx.min(), xx.max())
#     plt.ylim(yy.min(), yy.max())
#     plt.title(title)

#     plt.subplots_adjust(right=0.8)
#     # Add legend colors
#     leg1 = plt.legend(*a.legend_elements(), frameon=False, fontsize=12,
#                      borderaxespad=0, bbox_to_anchor=(1, 0.5),
#                      handlelength=2, handletextpad=1, title=legend_title)


#     # Add legend sizes
#     l1 = plt.scatter([], [], c='black', s=0.4**2/h, edgecolors='none')
#     l2 = plt.scatter([], [], c='black', s=0.6**2/h, edgecolors='none')
#     l3 = plt.scatter([], [], c='black', s=0.8**2/h, edgecolors='none')
#     l4 = plt.scatter([], [], c='black', s=1**2/h, edgecolors='none')

#     labels = ["0.4", "0.6", "0.8", "1"]
#     leg = plt.legend([l1, l2, l3, l4], labels, frameon=False, fontsize=12,
#                      borderaxespad=0, bbox_to_anchor=(1, 1),
#                      handlelength=2, handletextpad=1, title='Probabilities', scatterpoints=1)
#     plt.gca().add_artist (leg1)
# plt.show()

# create a multi roc curve plot function
def multi_roc_auc_plot(X, y, models):
    # create a numeric outcome measure if the data type is categorical
#     if y.dtype == "category":
#         y_cat_codes = y.cat.codes.values
#     else:
#         y_cat_codes = y
    y_cat_codes = y
    # scale the data and create training and test sets of the data
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat_codes, test_size=.3, random_state=42)
    _ = plt.figure(figsize=(7,7))
    # Below for loop iterates through your models list
    for m in models:
        model = m['model'] # select the model
        model.fit(X_train, y_train) # train the model
        y_pred = model.predict(X_test) # predict the test data
        # Compute False postive rate, and True positive rate
        fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(X_test)[:,1])
        # Calculate Area under the curve to display on the plot
        auc = metrics.roc_auc_score(y_test, model.predict(X_test), average="macro")
        # Now, plot the computed values
        plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (m['label'], auc))
    # Custom settings for the plot
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity(False Positive Rate)')
    plt.ylabel('Sensitivity(True Positive Rate)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()   # Display
    
    
def RFE_opt_rf(X , y, n_features_to_select, max_depth, n_estimators) :
    # Perform a 75% training and 25% test data split
    X_train, X_test, y_train, y_test = train_test_split(X, y,   
                                                        test_size    = 0.3, 
                                                        stratify     = y, 
                                                        random_state = 42)
    # Instanciate Random Forest
    rf = RandomForestClassifier(random_state = 42,
                                oob_score    = False) # use oob_score with many trees
    
    # Define params_dt
    params_rf = {'max_depth'    : max_depth, 
                 'n_estimators' : n_estimators,
                 'max_features' : ['log2', 'auto', 'sqrt'],
                 'criterion'    : ['gini', 'entropy']}

    # Instantiate grid_dt
    grid_dt = GridSearchCV(estimator  = rf,
                           param_grid = params_rf,
                           scoring    = 'roc_auc',
                           cv         = 3,
                           n_jobs     = -2)

    # Optimize hyperparameter
    _ = grid_dt.fit(X_train, y_train)
    
    # Extract the best estimator
    optimized_rf = grid_dt.best_estimator_
          
    # Create the RFE with a optimized random forest
    rfe = RFE(estimator            = optimized_rf, 
              n_features_to_select = n_features_to_select,
              verbose              = 1)

    # Fit the eliminator to the data
    _ = rfe.fit(X_train, y_train)
    
    # create dataframe with features ranking (high = dropped early on)
    feature_ranking = pd.DataFrame(data  = dict(zip(X.columns, rfe.ranking_)) , 
                                   index = np.arange(0, len(X.columns)))
    feature_ranking = feature_ranking.loc[0,:].sort_values()

    # create dataframe with feature selected
    feature_selected = X.columns[rfe.support_].to_list()
          
    # create dataframe with importances per feature
    feature_importance = pd.Series(dict(zip(X.columns, optimized_rf.feature_importances_.round(2)))) 
          
    # Calculates the test set accuracy
    acc = accuracy_score(y_test, rfe.predict(X_test))
    
    print("\n- Sizes :")
    print(f"- X shape = {X.shape}")
    print(f"- y shape = {y.shape}")
    print(f"- X_train shape = {X_train.shape}")
    print(f"- X_test shape = {X_test.shape}")
    print(f"- y_train shape = {y_train.shape}")
    print(f"- y_test shape = {y_test.shape}")
    
    print("\n- Optimal Parameters :")
    print(f"- max_depth = {optimized_rf.get_params()['max_depth']}")
    print(f"- n_estimators = {optimized_rf.get_params()['n_estimators']}")
    print(f"- max_features = {optimized_rf.get_params()['max_features']}")
    print(f"- criterion = {optimized_rf.get_params()['criterion']}")
    print(f"- Selected feature list = {feature_selected}")
    print("- Accuracy score on test set = {0:.1%}".format(acc))  
    
    max_depth = optimized_rf.get_params()['max_depth']
    n_estimators = optimized_rf.get_params()['n_estimators']
    max_features = optimized_rf.get_params()['max_features']
    criterion = optimized_rf.get_params()['criterion']
                    
    return (feature_ranking, feature_selected, feature_importance, max_depth, n_estimators, max_features, criterion)

# Create a function which will silence printing when called
import os
import sys
from contextlib import contextmanager # these three are needed to create the silence output function
@contextmanager
def silence_stdout():
    new_target = open(os.devnull, "w")
    old_target = sys.stdout
    sys.stdout = new_target
    try:
        yield new_target
    finally:
        sys.stdout = old_target
        
        
def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)
        


def summary_performance_metrics_classification(
    y_true, y_pred
):
    """Summary of different evaluation metrics specific to a single class classification learning problem.
    Notes
    -----
    The function returns the following metrics:
    - true positive (TP): The model classifies the example as positive, and the actual label also positive.
    - false positive (FP): The model classifies the example as positive, but the actual label is negative.
    - true negative (TN): The model classifies the example as negative, and the actual label is also negative.
    - false negative (FN): The model classifies the example as negative, but the label is actually positive.
    - accuracy: The fractions of predictions the model got right.
    - prevalance: The proportion of positive examples. Where y=1.
    - sensitivity: The probability that our test outputs positive given that the case is actually positive.
    - specificity: The probability that the test outputs negative given that the case is actually negative.
    - positive predictive value: The proportion of positive predictions that are true positives.
    - negative predictive value: The proportion of negative predictions that are true negatives.
    - auc: A measure of goodness of fit.
    - F1: The harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
    Examples
    --------
    >>> from sklearn import datasets
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> import pandas as pd
    >>> data = datasets.load_breast_cancer()
    >>> df = pd.DataFrame(data.data, columns=data.feature_names)
    >>> df['target'] = data.target
    >>> X = data.data
    >>> y = data.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> clf = KNeighborsClassifier(n_neighbors=6)
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_test)
    >>> summary_performance_metrics_classification(y_true=y_test, y_pred=y_pred)
    Parameters
    ----------
    y_true: pd.Series or np.arrays
        Binary true values.
    y_pred: pd.Series or np.arrays
        Binary predictions of model.
    
    Returns
    -------
    pd.DataFrame
    """
    # TP, TN, FP, FN
    confusion_matrix_metric = confusion_matrix(y_true, y_pred)
    TN = confusion_matrix_metric[0][0]
    FP = confusion_matrix_metric[0][1]
    FN = confusion_matrix_metric[1][0]
    TP = confusion_matrix_metric[1][1]

    # accuracy
    accuracy_score_metric = accuracy_score(y_true, y_pred)

    # prevalance
    prevalence = np.mean(y_true == 1)

    # sensitivity
    sensitivity = TP / (TP + FN)

    # specificity
    specificity = TN / (TN + FP)
    
    # balanced accuracy
    bacc = ((sensitivity + specificity) / 2)

    # positive predictive value
    PPV = TP / (TP + FP)

    # negative predictive value
    NPV = TN / (TN + FN)

    # auc
    auc_score = roc_auc_score(y_true, y_pred)

    # F1
    f1 = f1_score(y_true, y_pred)

    df_metrics = pd.DataFrame(
        {
            "TN": TN,
            "FP": FP,
            "FN": FN,
            "TP": TP,
            "Accuracy": accuracy_score_metric,
            "Balanced Accuracy": bacc,
            "Prevalence": prevalence,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "PPV": PPV,
            "NPV": NPV,
            "auc": auc_score,
            "F1": f1,
        },
        index=["scores"],
    )

    return df_metrics