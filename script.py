"""
File: script.py
Author: Benjamin Henning
Project: ISTA 131 Final Project
Section: 01 (Tingting)
Purpose: This program takes data from the FBI Crime Database and uses machine learning models to make predictions about crime rates up to 2035. There is some redundant code because I wanted to allow the user to input data from any dataset from the FBI crime database and be able to apply the model to it. The most relevant functions will be make_frames(), which will import all the datasets from the given directory and add them to a dictionary that maps the filename (minus the file extension) to a pandas dataframe of the dataset, sorted by year. The best model I found was a log transformation of a linear regression, which can be used with the graph_ols_log() function. The graph_prediction function uses the log model to make a 15 year prediction for that dataset.
"""

import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, statsmodels.api as sm

def make_frames(directory):
    """
    This function loops through a directory and makes a dictionary mapping filenames to pandas dataframe of the data contained in each file.

    Parameters
    ----------
    directory : string
        Path of a directory containing csv's.

    Returns
    -------
    frames : Dictionary
        Maps strings of filenames to dataframes containing the data in those files.
    """
    frames = {}
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        frames[filename[:-4]] = pd.read_csv(file_path, index_col="Year").sort_values(by="Year", axis='index')
    
    return frames

def graph_previs(frames, crime_key, loc_key, title, colors):
    """
    Graphs the pre-visualization data to show trends of data before applying machine-learning predictions.

    Parameters
    ----------
    frames : Dictionary
        A dictionary of pandas DataFrames holding crime rate data.
    crime_key : string
        Either property or violent. We use this key to separate violent crime data and property crime data into two graphs.
    loc_key : string
        A Location to filter the dataframe by so we can get only Arizona crime data or only United States crime data.
    title : string
        A Title for the graph.
    colors : tuple
        A tuple of hex strings to color the graph by.

    Returns
    -------
    None.

    """
    
    # filter keys by crime_key:
    filtered_keys = [key for key in frames.keys() if crime_key in key and 'total' not in key]
    
    # filter frames by location
    filtered_frames = {}
    for filename in filtered_keys:
        frame = frames[filename]
        is_loc = frame['Location'] == loc_key
        filtered_frames[filename] = list(frame[is_loc].loc[:,'Rate'])
        
    # sort frames by values
    sorted_labels = sorted(filtered_frames, key=filtered_frames.get, reverse=True)
    for i in range(len(sorted_labels)):
        label = sorted_labels[i]
        strs = label.split("_")
        new_label = strs[0]
        if len(strs) > 2:
            for item in strs[1:-1]:
                new_label += " " + item
        sorted_labels[i] = new_label
            
        
    x = range(1985, 2020)
    y = list(sorted(filtered_frames.values(), reverse=True))

    plt.stackplot(x, y, colors=colors, labels=sorted_labels)
    plt.legend(fontsize='small', markerfirst=False, framealpha=0)
    plt.xlabel('Year')
    plt.ylabel('Crime Rate per 100,000 people')
    plt.title(title)
    plt.show()
    
def graph_total(total_frame, loc_key, title, color):
    """
    Makes a simple line graph of total crime rates for property crime and violent crime.

    Parameters
    ----------
    total_frame : DataFrame
        DataFrame of the total crime rates.
    loc_key : string
        A Location to filter the dataframe by so we can get only Arizona crime data or only United States crime data.
    title : string
        A Title for the graph.
    colors : tuple
        A tuple of hex strings to color the graph by.

    Returns
    -------
    None.

    """
    values = list(total_frame[total_frame['Location'] == loc_key].loc[:, 'Rate'])
    x = range(1985, 2020)
    y = values
    plt.plot(x, y, color=color)
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Crime Rate per 100,000 people')
    plt.show()

def graph_ols_linear(total_frame, loc_key, title, color):
    """
    Graphs the linear OLS model on top of the total crime data

    Parameters
    ----------
    total_frame : DataFrame
        DataFrame of the total crime rates.
    loc_key : string
        A Location to filter the dataframe by so we can get only Arizona crime data or only United States crime data.
    title : string
        A Title for the graph.
    colors : tuple
        A tuple of hex strings to color the graph by.

    Returns
    -------
    None.

    """

    values = list(total_frame[total_frame['Location'] == loc_key].loc[:, 'Rate'])
    xs = np.arange(1985, 2020)
    
    x = sm.add_constant(xs)
    model = sm.OLS(values, x)
    results = model.fit()
    
    
    ys = results.params[1] * xs + results.params[0]
    plt.plot(xs, values, color=color)
    plt.plot(xs, ys, linewidth='4', color='orange')
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Crime Rate per 100,000 people')
    plt.show()
    print(results.summary())
    
def graph_ols_poly(total_frame, loc_key, title, color):
    """
    Graphs the polynomial OLS model on top of the total crime data

    Parameters
    ----------
    total_frame : DataFrame
        DataFrame of the total crime rates.
    loc_key : string
        A Location to filter the dataframe by so we can get only Arizona crime data or only United States crime data.
    title : string
        A Title for the graph.
    colors : tuple
        A tuple of hex strings to color the graph by.

    Returns
    -------
    None.

    """

    values = list(total_frame[total_frame['Location'] == loc_key].loc[:, 'Rate'])
    xs = np.arange(1985, 2020)
    
    x = np.column_stack([xs ** i for i in range(5)])
    model = sm.OLS(values, x)
    results = model.fit()
    
    ys = results.params[0] + results.params[1] * xs \
        + results.params[2] * xs ** 2 \
        + results.params[3] * xs ** 3 \
        + results.params[4] * xs ** 4
        
    plt.plot(xs, values, color=color)
    plt.plot(xs, ys, linewidth='4', color='orange')
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Crime Rate per 100,000 people')
    plt.show()
    print(results.summary())
    
def graph_ols_log(total_frame, loc_key, title, color):
    """
    Graphs the log transformation of the linear OLS model on top of the total crime data

    Parameters
    ----------
    total_frame : DataFrame
        DataFrame of the total crime rates.
    loc_key : string
        A Location to filter the dataframe by so we can get only Arizona crime data or only United States crime data.
    title : string
        A Title for the graph.
    colors : tuple
        A tuple of hex strings to color the graph by.

    Returns
    -------
    None.

    """

    values = list(total_frame[total_frame['Location'] == loc_key].loc[:, 'Rate'])
    xs = np.arange(1985, 2020)
    
    y = np.log(values)
    x = sm.add_constant(xs)
    model = sm.OLS(y, x)
    results = model.fit()
    
    
    ys = np.exp(results.params[0] + results.params[1] * xs)
    plt.plot(xs, values, color=color)
    plt.plot(xs, ys, linewidth='4', color='orange')
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Crime Rate per 100,000 people')
    plt.show()
    print(results.summary())
    
def graph_prediction(total_frame, loc_key, title, color):
    """
    Graphs the log transformation of the linear OLS model on top of the total crime data, with the range extended to 2035.

    Parameters
    ----------
    total_frame : DataFrame
        DataFrame of the total crime rates.
    loc_key : string
        A Location to filter the dataframe by so we can get only Arizona crime data or only United States crime data.
    title : string
        A Title for the graph.
    colors : tuple
        A tuple of hex strings to color the graph by.

    Returns
    -------
    None.
    """

    
    values = list(total_frame[total_frame['Location'] == loc_key].loc[:, 'Rate'])
    xs = np.arange(1985, 2020)
    
    y = np.log(values)
    x = sm.add_constant(xs)
    model = sm.OLS(y, x)
    results = model.fit()
    
    x_pred = np.arange(1985, 2036)
    ys = np.exp(y)
    ys_pred = np.exp(results.params[0] + results.params[1] * x_pred)
    
    plt.plot(xs, ys, color=color)
    plt.plot(x_pred, ys_pred, linewidth='4', color='orange')
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Crime Rate per 100,000 people')
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    
    plt.show()
        
    
    

def main():
    datasets_path = "datasets/"
    frames = make_frames(datasets_path)
    
    prop_colors = ('#997444', '#CCB698', '#FFE3D8', '#98CCA5')
    vio_colors = ('#141012', '#B43B49', '#347978')
    graph_previs(frames, 'property', 'Arizona', 'Arizona Property Crime', prop_colors)
    graph_previs(frames, 'violent', 'Arizona', 'Arizona Violent Crime', vio_colors)
    
    graph_total(frames['total_property'], 'Arizona', 'Arizona Total Property Crime', '#4C4C4C')
    graph_total(frames['total_violent'], 'Arizona', 'Arizona Total Violent Crime', '#1F1F1F')
    
    graph_ols_linear(frames['total_property'], 'Arizona', 'Arizona Total Property Crime', '#4C4C4C')
    graph_ols_linear(frames['total_violent'], 'Arizona', 'Arizona Total Violent Crime', '#1F1F1F')
    
    graph_ols_poly(frames['total_property'], 'Arizona', 'Arizona Total Property Crime', '#4C4C4C')
    graph_ols_poly(frames['total_violent'], 'Arizona', 'Arizona Total Violent Crime', '#1F1F1F')
    
    graph_ols_log(frames['total_property'], 'Arizona', 'Arizona Total Property Crime', '#4C4C4C')
    graph_ols_log(frames['total_violent'], 'Arizona', 'Arizona Total Violent Crime', '#1F1F1F')
    
    graph_prediction(frames['total_property'], 'Arizona', 'Arizona Total Property Crime', '#4C4C4C')
    graph_prediction(frames['total_violent'], 'Arizona', 'Arizona Total Violent Crime', '#1F1F1F')
    
    graph_ols_log(frames['total_property'], 'United States', 'US Total Property Crime', '#4C4C4C')
    graph_ols_log(frames['total_violent'], 'United States', 'US Total Violent Crime', '#1F1F1F')
    

main()