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
    testing = pd.read_csv(data_path + "time_series_covid19_testing_US.csv")

    return confirmed, deaths, testing

def read_data_global():
    confirmed = pd.read_csv(data_path + "time_series_covid19_confirmed_global.csv")
    deaths = pd.read_csv(data_path + "time_series_covid19_deaths_global.csv")
    testing = pd.read_csv(data_path + "time_series_covid19_testing_global.csv")

    return confirmed, deaths, testing

def read_data():
    confirmed_us, deaths_us, testing_us = read_data_usa()
    confirmed_global, deaths_global, testing_global = read_data_global()
    # according to https://github.com/CSSEGISandData/COVID-19/issues/1250 they
    # replaced US states by a single US entry in global. We drop it and merge
    # it back into a unified dataframe
    confirmed = confirmed_global[confirmed_global[country_string] != "US"].append(confirmed_us)
    deaths = deaths_global[deaths_global[country_string] != "US"].append(deaths_us)
    testing = testing_global[testing_global[country_string] != "US"].append(testing_us)

    return confirmed, deaths, testing

def parse_country_data():
    confirmed, deaths, recovered = read_data()

    confirmed = confirmed.drop([lat_string, long_string, province_string], axis=1)
    deaths = deaths.drop([lat_string, long_string, province_string], axis=1)
    recovered = recovered.drop([lat_string, long_string, province_string], axis=1)

    confirmed = confirmed.groupby(country_string).agg("sum")
    deaths = deaths.groupby(country_string).agg("sum")
    recovered = recovered.groupby(country_string).agg("sum")

    return confirmed, deaths, recovered

def parse_province_data(country):
    confirmed, deaths, recovered = read_data()

    confirmed = confirmed[confirmed[country_string] == country]
    deaths = deaths[deaths[country_string] == country]
    recovered = recovered[recovered[country_string] == country]

    return confirmed, deaths, recovered

def semilog_per_capita_since(countries, threshold_per_capita=1,
                             time_constant_type=10, num_datapoints_fit=10000,
                             fit_first_last="first"):
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
    countries: list of strings representing valid countries in the data set
    threshold_per_capita: threshold per capita number of cases per million people
                          that determines the start of the data set for each
                          country (default = 1)
    time_constant_type: the multiple for which the time constant is evaluated.
                        A value of 10 means that the reported time constant
                        will be for a growth of 10x. (Default = 10)
    num_datapoints_fit: The maximum number of data points to use in creating
                        the time constant fit.  As countries "flatten their curve"
                        a single exponential fit will not represent the early
                        time constant (which is, debatably, more interesting).
                        It may be prudent to only consider the first set of
                        points (default = 10000; all points)
    fit_first_last: String indicating whether the fit should occur over the "first"
                    or "last" N data points. (default = "first")
    '''

    cases, deaths, recovered = parse_country_data()
    pop_data = read_population_data()

    plt.figure(figsize=(10,7),facecolor="white")

    for country in countries:
        tmp_data = np.array(cases[cases.index.isin([country])].values.tolist()[0])
        tmp_data = tmp_data/pop_data[country]
        tmp_data = tmp_data[tmp_data>(threshold_per_capita/1e6)]
        fit_length = np.min([tmp_data.size,num_datapoints_fit])
        if fit_first_last == "last":
            fit_data = np.polyfit(range(fit_length),np.log10(tmp_data[-fit_length:]),1)
        else:
            fit_data = np.polyfit(range(fit_length),np.log10(tmp_data[:fit_length]),1)
        time_constant = 1/(fit_data[0]/np.log10(time_constant_type))
        plt.semilogy(range(tmp_data.size),tmp_data*1e6,
                     label="{} ({}x time: {:.2f} days)".format(country, time_constant_type,
                                                               time_constant))
    plt.xlabel("Days since {}/1,000,000 per capita cases.".format(threshold_per_capita))
    plt.ylabel("Number of cases per million people")
    plt.legend(title="Time constants based on \n {} {} data points.".format(fit_first_last,
                                                                            num_datapoints_fit))



def semilog_deaths_since(countries, threshold_num_cases=100,
                        time_constant_type=10, num_datapoints_fit=10000,
                        fit_first_last="first"):

    cases, deaths, recovered = parse_country_data()

    return semilog_data_since(deaths, countries, data_type="deaths",
                       threshold_num_cases=threshold_num_cases,
                       time_constant_type=time_constant_type,
                       num_datapoints_fit=num_datapoints_fit,
                       fit_first_last=fit_first_last)

def semilog_cases_since(countries, threshold_num_cases=100,
                        time_constant_type=10, num_datapoints_fit=10000,
                        fit_first_last="first"):

    cases, deaths, recovered = parse_country_data()

    return semilog_data_since(cases, countries, data_type="cases",
                       threshold_num_cases=threshold_num_cases,
                       time_constant_type=time_constant_type,
                       num_datapoints_fit=num_datapoints_fit,
                       fit_first_last=fit_first_last)

def semilog_data_since(plot_data, countries, data_type="cases",
                       threshold_num_cases=100, time_constant_type=10,
                       num_datapoints_fit=10000, fit_first_last="first"):

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
    threshold_num_cases: threshold number of cases that determines the start of the data
           set for each country (default = 100)
    time_constant_type: the multiple for which the time constant is evaluated.
                        A value of 10 means that the reported time constant
                        will be for a growth of 10x. (Default = 10)
    num_datapoints_fit: The maximum number of data points to use in creating
                        the time constant fit.  As countries "flatten their curve"
                        a single exponential fit will not represent the early
                        time constant (which is, debatably, more interesting).
                        It may be prudent to only consider the first set of
                        points (default = 10000; all points)
    fit_first_last: String indicating whether the fit should occur over the "first"
                    or "last" N data points. (default = "first")
    '''

    fig = plt.figure()
    ax = plt.axes()

    for country in countries:
        tmp_data = np.array(plot_data[plot_data.index.isin([country])].values.tolist()[0])
        tmp_data = tmp_data[tmp_data>threshold_num_cases]
        fit_length = np.min([tmp_data.size,num_datapoints_fit])
        if fit_first_last == "last":
            fit_data = np.polyfit(range(fit_length),np.log10(tmp_data[-fit_length:]),1)
        else:
            fit_data = np.polyfit(range(fit_length),np.log10(tmp_data[:fit_length]),1)
        time_constant = 1/(fit_data[0]/np.log10(time_constant_type))
        ax.semilogy(range(tmp_data.size),tmp_data,
                     label="{} ({}x time: {:.2f} days)".format(country, time_constant_type,
                                                               time_constant))
    ax.set_xlabel("Days since {} cummulative {}.".format(threshold_num_cases,data_type))
    ax.set_ylabel("Total number of {}.".format(data_type))
    ax.legend(title="Time constants based on \n {} {} data points.".format(fit_first_last,
                                                                            num_datapoints_fit))

    return ax

def generate_all_plots(countries):
    confirmed, deaths, recovered = parse_country_data()

    death_rate = deaths/confirmed
    recovery_rate = recovered/confirmed

    print(f"A total of {len(confirmed)} countries confirmed at least one case of covid-19")

    def death_rate_by_country(country):
        return death_rate[death_rate.index.isin([country])].T

    for country in countries:
        death_rate_country = death_rate_by_country(country)
        death_rate_country = death_rate_country[death_rate_country > 0]
        print(f"Mean death rate for {country}: {float(death_rate_country.mean()):.4f} (+-{float(death_rate_country.std()):.4f} std)")

    generate_absolute_plot(confirmed, countries, title="Confirmed cases")
    generate_absolute_plot(deaths, countries, title="Deaths")
    generate_absolute_plot(recovered, countries, title="Recovered cases")

    generate_absolute_plot(death_rate, countries, "Death rate by day and country")
    generate_absolute_plot(recovery_rate, countries, "Recovery rate per day by country")

    # log_confirmed = np.log(confirmed.replace(0, 1))

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

def generate_absolute_plot(data, countries, title=None):
    return data[data.index.isin(countries)].replace(np.nan, 0).T.plot(title=title)


def generate_log_plot(data, countries, title=None):
    return data[data.index.isin(countries)].replace(np.nan, 0).T.plot(logy=True, title=title)


def generate_loglog_plot(data, countries, title=None):
    return data[data.index.isin(countries)].replace(np.nan, 0).T.plot(loglog=True, title=title)

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
