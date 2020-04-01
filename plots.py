#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

data_path = "COVID-19/csse_covid_19_data/csse_covid_19_time_series/"

country_string = "Country/Region"
province_string = "Province/State"
long_string = "Long"
lat_string = "Lat"


def use_JHU_github_live_data():
    override_file_prefix("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/")


def override_file_prefix(path):
    '''Override the data path used to load data files
    '''
    global data_path
    data_path = path


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

def read_data_usa():
    confirmed = pd.read_csv(data_path + "time_series_covid19_confirmed_US.csv")
    deaths = pd.read_csv(data_path + "time_series_covid19_deaths_US.csv")
    testing = None # pd.read_csv(data_path + "time_series_covid19_testing_US.csv")

    return confirmed, deaths, testing

def read_data_global():
    confirmed = pd.read_csv(data_path + "time_series_covid19_confirmed_global.csv")
    deaths = pd.read_csv(data_path + "time_series_covid19_deaths_global.csv")
    testing = None # pd.read_csv(data_path + "time_series_covid19_testing_global.csv")

    return confirmed, deaths, testing

def read_data():
    #confirmed_us, deaths_us, testing_us = read_data_usa()
    confirmed_global, deaths_global, testing_global = read_data_global()
    # according to https://github.com/CSSEGISandData/COVID-19/issues/1250 they
    # replaced US states by a single US entry in global. We drop it and merge
    # it back into a unified dataframe
    confirmed = confirmed_global# [confirmed_global[country_string] != "US"].append(confirmed_us)
    deaths = deaths_global# [deaths_global[country_string] != "US"].append(deaths_us)
    testing = None # testing_global[testing_global[country_string] != "US"].append(testing_us)

    return confirmed, deaths, None #  testing is not yet published

def parse_country_data():
    confirmed, deaths, testing = read_data()

    confirmed = confirmed.drop([lat_string, long_string, province_string], axis=1)
    deaths = deaths.drop([lat_string, long_string, province_string], axis=1)
    # testing = testing.drop([lat_string, long_string, province_string], axis=1)

    confirmed = confirmed.groupby(country_string).agg("sum")
    deaths = deaths.groupby(country_string).agg("sum")
    testing = None # testing.groupby(country_string).agg("sum")

    return confirmed, deaths, testing

def parse_state_data(country):
    confirmed, deaths, testing = read_us_data()

    confirmed = confirmed[confirmed[country_string] == country]
    deaths = deaths[deaths[country_string] == country]
    testing = None # recovered[recovered[country_string] == country]

    return confirmed, deaths, testing


def semilog_per_capita_country_since(plot_data, countries, data_type="cases",
                                     threshold=100,
                                     fit_info = {'constant' : 10,
                                                 'length' : 5,
                                                 'type' :"exp"}):
    '''Create a semilog plot of the per capita number of cases in each country,
    measured in days since that country first experienced a threshold number
    of cases (default: 100).

    The plot legend will include the time constant for an increase of a given
    multiple (default: 10).  The number of datapoints over which this time
    constant is evaluated can be changed in order to capture the initial trend
    (default: 10000).  The fit can be applied to the first N data points or
    the last N data points (default: first).

    inputs
    -------
     plot_data: data frame containing the data to be plotted/analyzed.  Typically
               either total cases or deaths
    countries: list of strings representing valid countries in the data set
    data_type: string for plot legends indicating the data type being plotted
    threshold_per_capita: threshold per capita number of cases per million people
                          that determines the start of the data set for each
                          country (default = 1)
    time_constant_type: the multiple for which the time constant is evaluated.
                        A value of 10 means that the reported time constant
                        will be for a growth of 10x. (Default = 10)
    fit_length_constant: Measure used to determine how much data to use in the fit.

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
    fit_type: String indicating which type of fit:
              "first" - semi-log fit to first N points
              "last"  - semi-log fit to last N points
              "exp"   - semi-log fit to all points with an exponentially 
                        decreasing weight as data is older, with constant 1/N
              (default: "first")

    '''

    fig = plt.figure(figsize=(10,7),facecolor="white")
    ax = plt.axes()

    for country in countries:
        tmp_data, fit_data = evaluate_country_per_capita_data(plot_data, country, threshold, fit_info)
        time_constant = fit_data[0]
        legend_label = fit_data[1]
        ax.semilogy(range(tmp_data.size),tmp_data,"o-",
                     label="{} ({}x time: {:.2f} days)".format(country, fit_info['constant'],
                                                               time_constant))
    ax.set_xlabel("Days since {}/1,000,000 per capita {}.".format(threshold, data_type))
    ax.set_ylabel("Number of {} per million people.".format(data_type))
    ax.legend(title=legend_label)

def semilog_us_deaths_since(states, counties, threshold_num_cases=100,
                        time_constant_type=10, fit_length_constant=10000,
                        fit_type="first"):

    cases, deaths, recovered = parse_state_data()

    return semilog_us_data_since(deaths, states, counties, data_type="deaths",
                       threshold_num_cases=threshold_num_cases,
                       time_constant_type=time_constant_type,
                       fit_length_constant=fit_length_constant,
                       fit_type=fit_type)

def semilog_us_cases_since(states, counties, threshold_num_cases=100,
                        time_constant_type=10, fit_length_constant=10000,
                        fit_type="first"):

    cases, deaths, recovered = parse_state_data()

    return semilog_us_data_since(cases, states, counties, data_type="cases",
                       threshold_num_cases=threshold_num_cases,
                       time_constant_type=time_constant_type,
                       fit_length_constant=fit_length_constant,
                       fit_type=fit_type)

def evaluate_country_per_capita_data(data_source,
                                     country,
                                     threshold=100,
                                     fit_info = {'constant' : 10,
                                                 'length' : 5,
                                                 'type' : "exp"}):
    
    pop_data = read_population_data()

    plot_data = np.array(data_source[data_source.index.isin([country])].values.tolist()[0])
    plot_data = plot_data/pop_data[country]
    plot_data = plot_data[plot_data>(threshold/1e6)]

    return plot_data, fit_country_data(plot_data, fit_info)

def evaluate_country_data(data_source,
                          country,
                          threshold=100,
                          fit_info = {'constant' : 10,
                                      'length' : 5,
                                      'type' : "exp"}):
    
    plot_data = np.array(data_source[data_source.index.isin([country])].values.tolist()[0])
    plot_data = plot_data[plot_data>threshold]

    return plot_data, fit_country_data(plot_data, fit_info)

def fit_country_data(plot_data, fit_info):
    
    if fit_info['type'] == "exp":
        data_size = plot_data.size
        weights = np.exp(-np.array(range(data_size))/fit_info['length'])
        fit_data = np.polyfit(range(data_size),np.log10(plot_data),1,w=np.flip(weights))
        legend_label = "Time constants based on \nexponentially weighted fit \n" + \
                       "with exp constant {}.".format(fit_info['length'])
    else:
        fit_length = np.min([plot_data.size, fit_info['length']])
        if fit_info['type'] == "last":
            fit_data = np.polyfit(range(fit_length),np.log10(plot_data[-fit_length:]),1)
        else:
            fit_data = np.polyfit(range(fit_length),np.log10(plot_data[:fit_length]),1)
        legend_label = "Time constants based on \n {} {} data points.".format(fit_info['type'],
                                                                              fit_info['length'])

    time_constant = 1/(fit_data[0]/np.log10(fit_info['constant']))

    return time_constant, legend_label

def semilog_country_since(plot_data, countries,
                          data_type="cases",
                          threshold=100,
                          fit_info = {'constant' : 10,
                                      'length' : 5,
                                      'type' :"exp"}):

    '''Create a semilog plot of the total number of cases in each country,
    measured in days since that country first experienced a threshold number
    of cases (default: 100).

    The plot legend will include the time constant for an increase of a given
    multiple (default: 10).  The number of datapoints over which this time
    constant is evaluated can be changed in order to capture the initial trend
    (default: 10000).  The fit can be applied to the first N data points or
    the last N data points (default: first).

    inputs
    -------
    plot_data: data frame containing the data to be plotted/analyzed.  Typically
               either total cases or deaths
    countries: list of strings representing valid countries in the data set
    data_type: string for plot legends indicating the data type being plotted
    threshold: threshold number of cases that determines the start of the data
           set for each country (default = 100)
    time_constant_type: the multiple for which the time constant is evaluated.
                        A value of 10 means that the reported time constant
                        will be for a growth of 10x. (Default = 10)
    fit_length_constant: Measure used to determine how much data to use in the fit.

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
    fit_type: String indicating which type of fit:
              "first" - semi-log fit to first N points
              "last"  - semi-log fit to last N points
              "exp"   - semi-log fit to all points with an exponentially 
                        decreasing weight as data is older, with constant 1/N
              (default: "first")
    '''

    fig = plt.figure(figsize=(10,7),facecolor="white")
    ax = plt.axes()

    for country in countries:
        tmp_data, fit_data = evaluate_country_data(plot_data, country, threshold, fit_info)
        time_constant = fit_data[0]
        legend_label = fit_data[1]
        ax.semilogy(range(tmp_data.size),tmp_data,"o-",
                     label="{} ({}x time: {:.2f} days)".format(country, fit_info['constant'],
                                                               time_constant))
    ax.set_xlabel("Days since {} cummulative {}.".format(threshold,data_type))
    ax.set_ylabel("Total number of {}.".format(data_type))
    ax.legend(title=legend_label)

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

    return semilog_country_since(confirmed, countries)
    
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
