#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import json

data_path = {'local': "COVID-19/csse_covid_19_data/csse_covid_19_time_series/",
             'live' : "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
             }
default_data_location = 'local'
use_data_location = default_data_location

country_string = "Country/Region"
province_string = "Province/State"
state_string = "Province_State"  # for some reason the US data uses a different column name
long_string = "Long"
lat_string = "Lat"

def use_JHU_github_live_data():
    global use_data_location
    use_data_location = 'live'

def read_population_data():
    '''
    Read population-by-country data from JSON file.

    output
    ------
    pop_dict : dictionary with key of the country name (string)
               and value of the population (float)
    '''


    country_json = open("country-data/country-by-population.json")
    pop_data = json.load(country_json)
    pop_dict = {}
    for country_data in pop_data:
        if country_data['population'] is not None:
            pop_dict[country_data['country']] = float(country_data['population'])

    return pop_dict

pop_data = read_population_data()


def read_data(region, data_location = None):
    if data_location is None:
        data_location = use_data_location
    confirmed = pd.read_csv(data_path[data_location] + "time_series_covid19_confirmed_" + region + ".csv")
    deaths = pd.read_csv(data_path[data_location] + "time_series_covid19_deaths_" + region + ".csv")
    testing = None # pd.read_csv(data_path[data_location] + "time_series_covid19_testing_" + region + ".csv")

    return confirmed, deaths, testing

def read_data_usa(data_location = None):
    if data_location is None:
        data_location = use_data_location
    return read_data("US", data_location)

def read_data_global(data_location = None):
    if data_location is None:
        data_location = use_data_location
    return read_data("global", data_location)

def resolve_data_location(data_location):

    if data_location is None:
        data_location = use_data_location

    print("using " + data_location + " data")

    return data_location


def parse_country_data(data_location = None):

    confirmed, deaths, testing = read_data_global(resolve_data_location(data_location))

    drop_columns = [lat_string, long_string, province_string]
    confirmed = confirmed.drop(drop_columns, axis=1)
    deaths = deaths.drop(drop_columns, axis=1)
    # testing = testing.drop(drop_columns, axis=1)

    confirmed = confirmed.groupby(country_string).agg("sum")
    deaths = deaths.groupby(country_string).agg("sum")
    testing = None # testing.groupby(country_string).agg("sum")

    return confirmed, deaths, testing

def parse_us_state_data(data_location = None):

    confirmed, deaths, testing = read_data_usa(resolve_data_location(data_location))

    drop_columns = ['UID','iso2','iso3','code3','FIPS','Admin2','Country_Region','Lat','Long_','Combined_Key']
    confirmed = confirmed.drop(drop_columns, axis=1)
    deaths = deaths.drop(drop_columns, axis=1)
    deaths = deaths.drop(['Population'], axis=1)
    # testing = testing.drop(drop_columns, axis=1)

    
    confirmed = confirmed.groupby(state_string).agg("sum")
    deaths = deaths.groupby(state_string).agg("sum")

    testing = None # testing.groupby(state_string).agg("sum")

    return confirmed, deaths, testing

def parse_us_county_data(state, data_location = None):

    confirmed, deaths, testing = read_data_usa(resolve_data_location(data_location))

    drop_columns = ['UID','iso2','iso3','code3','FIPS','Country_Region','Lat','Long_','Combined_Key']
    confirmed = confirmed.drop(drop_columns, axis=1)
    deaths = deaths.drop(drop_columns, axis=1)
    deaths = deaths.drop(['Population'], axis=1)
    # testing = testing.drop(drop_columns, axis=1)

    confirmed = confirmed[confirmed[state_string] == state].set_index('Admin2')
    deaths = deaths[deaths[state_string] == state].set_index('Admin2')

    confirmed = confirmed.drop([state_string], axis=1)
    deaths = deaths.drop([state_string], axis=1)
    testing = None # testing.groupby(state_string).agg("sum")

    return confirmed, deaths, testing

def select_region_data(data_source,
                       region,
                       population = 1,
                       threshold=100):

    """
    Select data for the given country, divide by its population and 
    trim to greater than some threshold.

    inputs
    ------
    data_source : a pandas data frame with country column as index
    country : the name of the country
    population : the population of the country
                 (default: 1, i.e. do not divide by population)
    threahsold : only data greater than this threshold is kept in the data
                 Note: the threshold should be scaled appropriately with 
                       the population
                 (default: 100)
    """

    plot_data = np.array(data_source[data_source.index.isin([region])].values.tolist()[0])
    plot_data = plot_data/population
    plot_data = plot_data[plot_data>threshold]

    return plot_data

def fit_region_data(plot_data, fit_info):
    """
    Perform a semilog fit to the data based on the definition of the fit,
    to generate a time constant.

    inputs
    ------
    plot_data : numpy array of daily data
    fit_info : defining the way that the fit will be calculated
               'constant' : the multiple for which the time constant is evaluated.
                            A value of 10 means that the reported time constant
                            will be for a growth of 10x. (Default = 10)
               'length' : Measure used to determine how much data to use in the fit.

                          For "first" and "last" fit types, this is the maximum 
                          number of data points to use in creating the time
                          constant fit.  As countries "flatten their curve" a
                          single exponential fit will not represent the early
                          time constant (which is, debatably, more
                          interesting).  It may be prudent to only consider the
                          first set of points 

                          For "exp" fit type, this is the rate at which the
                          exponential weighting of the data falls off for older 
                          data.
 
                          (default = 10000; i.e. all points)
                'type' : String indicating which type of fit:
                        "first" - semi-log fit to first N points
                        "last"  - semi-log fit to last N points
                        "exp"   - semi-log fit to all points with an exponentially 
                                  decreasing weight as data is older, with constant 1/N
                                  (default: "first")
    """
    
    if fit_info['type'] == "exp":
        data_size = plot_data.size
        weights = np.exp(-np.array(range(data_size))/fit_info['length'])
        fit_data = np.polyfit(range(data_size),np.log10(plot_data),1,w=np.flip(weights))
    else:
        fit_length = np.min([plot_data.size, fit_info['length']])
        if fit_info['type'] == "last":
            fit_data = np.polyfit(range(fit_length),np.log10(plot_data[-fit_length:]),1)
        else:
            fit_data = np.polyfit(range(fit_length),np.log10(plot_data[:fit_length]),1)

    time_constant = 1/(fit_data[0]/np.log10(fit_info['constant']))

    return time_constant

def generate_legend_label(fit_info):
    
    if fit_info['type'] == "exp":
        legend_label = "Time constants based on \nexponentially weighted fit \n" + \
                       "with exp constant {}.".format(fit_info['length'])
    else:
        legend_label = "Time constants based on \n {} {} data points.".format(fit_info['type'],
                                                                              fit_info['length'])

    return legend_label
    
def calculate_guideline(axis, doubling_time):
    """
    Calculate a doubling time guideline that fits within the bounds of the plot.

    inputs
    ------
    axis: dictionary of axis dimensions, with keys
          'xmin', 'xmax' : extents of x-axis, typically integers
          'ymin', 'ymax' : extends of y-axis
    doubling_time: doubling time selected for this line

    outputs
    -------
    (xmin, xmax) : tuple with first and last points of line on x-axis
    (ymin, ymax) : tuple with first and last points of line on y-axis
    """

    xmin = 0
    xmax = axis['xmax']
    ymin = axis['ymin']

    ymax = ymin * 2**(axis['xmax']/doubling_time)

    if ymax > axis['ymax']:
        ymax = axis['ymax']
        xmax = np.log2(ymax/ymin) * doubling_time

    return np.array((xmin, xmax)), np.array((ymin,ymax))
        
def semilog_per_capita_since(data_region_list,
                             data_type="cases",
                             threshold=1,
                             fit_info = {'constant' : 10,
                                         'length' : 5,
                                         'type' :"exp"}):
    '''
    Entry point for semilog plots for data since some threshold for per-capita data.

    Passes real population data into semilog_since() and scale threshold by 1e6,
    and update axis labels.

    See semilog_since() for definition of parameters.
    '''

    return semilog_since(data_region_list,
                         data_type = data_type,
                         threshold = threshold/1e6,
                         fit_info = fit_info,                        
                         xlabel="Days since {} {} per million people.",
                         ylabel="Number of {} per million people.",
                         yscale=1e6)

def semilog_since(data_region_list,
                  data_type="cases",
                  threshold=100,
                  fit_info = {'constant' : 10,
                              'length' : 5,
                              'type' :"exp"},
                  xlabel="Days since {} cummulative {}",
                  ylabel="Total number of {}.",
                  yscale=1,
                  labels=()):

    '''
    Create a semilog plot of the cases/deaths in each country, measured in days
    since that country first experienced a threshold number of cases (default:
    100).  Poplation data will be used to generate per capita results.

    The plot legend will include the time constant for an increase of a given
    multiple.  See fit_region_data() for usage of fit_info definition.

    inputs
    -------
    data_region_list: tuple that provides the following items in this order
                      - a data frame containing data
                      - a list of region names to extract from the index of that data frame
                      - population data if any
    data_type: string for plot legends indicating the data type being plotted
    threshold: threshold number of cases that determines the start of the data
           set for each country (default = 100)
    fit_info : dictionary to define the way that the fit will be calculated,
               structure is defined in fit_region_data()

    '''

    fig = plt.figure(figsize=(10,7),facecolor="white")
    ax = plt.axes()


    axis = {'xmin':0, 'xmax':0, 'ymin':1e6, 'ymax': 0}

    for (plot_data, region_list, population_data) in data_region_list:
        # fake population data for convenience
        #   - all countries have 1 person for NON-per-capita results
        if population_data is None:
            population_data = dict(zip(region_list,[1]*len(region_list)))
        for region in region_list:
            tmp_data = select_region_data(plot_data, region, population = population_data[region],
                                           threshold=threshold)
            if tmp_data.size > 2:
                time_constant = fit_region_data(tmp_data, fit_info)
                axis['xmax'] = max(axis['xmax'],len(tmp_data))
                axis['ymin'] = min(axis['ymin'],min(tmp_data))
                axis['ymax'] = max(axis['ymax'],max(tmp_data))
                ax.semilogy(range(tmp_data.size),tmp_data*yscale,"o-",
                            label="{} ({}x time: {:.2f} days)".format(region, fit_info['constant'],
                                                                      time_constant))
    doubling_lines = [2,3,4,5,10]
    for doubling_time in doubling_lines:
        guide_x, guide_y = calculate_guideline(axis,doubling_time)
        guide_y *= yscale
        ax.semilogy(guide_x,guide_y,'--',color="silver")
        ax.text(guide_x[1],guide_y[1],'doubles in\n{} days'.format(doubling_time),color="silver")
    ax.set_xlabel(xlabel.format(int(threshold*yscale),data_type))
    ax.set_ylabel(ylabel.format(data_type))
    ax.legend(title=generate_legend_label(fit_info))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    for label in labels:
        ax.annotate(label[4], xy=(label[0], label[1]), xytext=(label[2],label[3]),
                    arrowprops=dict(facecolor='black', shrink=0.005, width=1, headlength=6, headwidth=6))
    
    return fig

    
def generate_all_plots(countries):
    confirmed, deaths, recovered = parse_country_data()

    death_rate = deaths/confirmed

    print(f"A total of {len(confirmed)} countries confirmed at least one case of covid-19")

    def death_rate_by_country(country):
        return death_rate[death_rate.index.isin([country])].T

    for country in countries:
        death_rate_country = death_rate_by_country(country)
        death_rate_country = death_rate_country[death_rate_country > 0]
        print(f"Mean death rate for {country}: {float(death_rate_country.mean()):.4f} (+-{float(death_rate_country.std()):.4f} std)")

    generate_absolute_plot(confirmed, countries, title="Confirmed cases")
    generate_absolute_plot(deaths, countries, title="Deaths")

    generate_absolute_plot(death_rate, countries, "Death rate by day and country")

    generate_log_plot(confirmed, countries, "Total confirmed log-plot")
    generate_log_plot(deaths, countries, "Total deaths log-plot")

    generate_loglog_plot(deaths, countries, "Total deaths loglog-plot")

    generate_log_plot(confirmed.T.diff().T, countries, "New cases by country by day log--plot")

    generate_absolute_plot(deaths.T.diff().T, countries, "New deaths by country by day")
    generate_log_plot(deaths.T.diff().T, countries, "New deaths by country by day log--plot")



def plot_confirmed_cases(countries):
    confirmed, deaths, recovered = parse_country_data()
    return generate_absolute_plot(
        confirmed,
        countries
    )


def plot_deaths(countries):
    confirmed, deaths, recovered = parse_country_data()
    return generate_absolute_plot(
        deaths,
        countries
    )


def plot_new_deaths_per_day(countries):
    confirmed, deaths, recovered = parse_country_data()

    return generate_absolute_plot(
        deaths.T.diff().T,
        countries
    )


def plot_death_rate(countries):
    confirmed, deaths, recovered = parse_country_data()
    death_rate = deaths.T.diff().T/confirmed
    return generate_absolute_plot(
        death_rate,
        countries
    )


def plot_newly_confirmed_per_day(countries):
    confirmed, deaths, recovered = parse_country_data()
    return generate_absolute_plot(confirmed.T.diff().T,
                           countries)

def semilog_cases_since(countries):
    """
    convenience interface used in all_plots below
    """
    
    confirmed, deaths, recovered = parse_country_data()

    return semilog_since(((confirmed, countries, None),))
    
def generate_absolute_plot(data, countries, title=None):
    return data[data.index.isin(countries)].replace(np.nan, 0).T.plot(title=title).get_figure()


def generate_log_plot(data, countries, title=None):
    return data[data.index.isin(countries)].replace(np.nan, 0).T.plot(logy=True, title=title).get_figure()


def generate_loglog_plot(data, countries, title=None):
    return data[data.index.isin(countries)].replace(np.nan, 0).T.plot(loglog=True, title=title).get_figure()

all_plots = [
    {
        "title": "New Deaths By Confirmed Cases",
        "description": "New deaths divided by total confirmed cases at respective day",
        "fn": plot_death_rate
    },
    {
        "title": "Total Count Of Confirmed Cases",
        "description": "A plot of the total count of confirmed cases by day and country.",
        "fn": plot_confirmed_cases
    },
    {
        "title": "Time to 10-fold",
        "description": "",
        "fn": semilog_cases_since
    },
    {
        "title": "Total Count Of Deaths",
        "description": "New deaths divided by total confirmed cases at respective day",
        "fn": plot_deaths
    },
    {
        "title": "New Cases Per Day By Country",
        "description": "A plot of the total count of new confirmed cases by day and country.",
        "fn": plot_newly_confirmed_per_day
    },
    {
        "title": "New Deaths Per Day By Country",
        "description": "A plot of the total count of new deaths by day and country.",
        "fn": plot_new_deaths_per_day
    },
]
