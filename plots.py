#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

country_string = "Country/Region"
province_string = "Province/State"

def read_data():
    data_path = "COVID-19/csse_covid_19_data/csse_covid_19_time_series/"
    
    confirmed = pd.read_csv(data_path + "time_series_19-covid-Confirmed.csv")
    deaths = pd.read_csv(data_path + "time_series_19-covid-Deaths.csv")
    recovered = pd.read_csv(data_path + "time_series_19-covid-Recovered.csv")

    return confirmed, deaths, recovered
    
def parse_country_data():
    confirmed, deaths, recovered = read_data()

    confirmed = confirmed.drop(["Lat", "Long", province_string], axis=1)
    deaths = deaths.drop(["Lat", "Long", province_string], axis=1)
    recovered = recovered.drop(["Lat", "Long", province_string], axis=1)

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

def semilog_cases_since(countries, num_cases=100, time_constant_type=10, num_datapoints_fit=10000, fit_first_last="first"):
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
    countries: list of strings representing valid countries in the data set
    num_cases: threshold number of cases that determines the start of the data 
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

    cases, deaths, recovered = parse_country_data()

    for country in countries:
        tmp_data = np.array(cases[cases.index.isin([country])].values.tolist()[0])
        tmp_data = tmp_data[tmp_data>num_cases]
        fit_length = np.min([tmp_data.size,num_datapoints_fit])
        if fit_first_last == "last":
            fit_data = np.polyfit(range(fit_length),np.log10(tmp_data[-fit_length:]),1)
        else:
            fit_data = np.polyfit(range(fit_length),np.log10(tmp_data[:fit_length]),1)
        time_constant = 1/(fit_data[0]/np.log10(time_constant_type))
        plt.semilogy(range(tmp_data.size),tmp_data,
                     label="{} ({}x time: {:.2f} days)".format(country, time_constant_type,
                                                               time_constant))
    plt.xlabel("Days since {} cummulative cases.".format(num_cases))
    plt.ylabel("Total number of cases.")
    plt.legend(title="Time constants based on \n {} {} data points.".format(fit_first_last,
                                                                            num_datapoints_fit))    

def generate_all_plots(countries):
    confirmed, deaths, recovered = parse_country_data()

    death_rate = deaths/confirmed
    recovery_rate = recovered/confirmed

    print(f"A total of {len(confirmed)} countries confirmed at least one case of covid-19")

    def death_rate_by_country(country):
        return death_rate[death_rate.index.isin([country])].T

    for country in countries:
        death_rate_country = death_rate_by_country(country)
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
    
    generate_absolute_plot(confirmed.T.diff().T, countries, "New cases by country by day")
    generate_log_plot(confirmed.T.diff().T, countries, "New cases by country by day log--plot")
    
    generate_absolute_plot(deaths.T.diff().T, countries, "New deaths by country by day")
    generate_log_plot(deaths.T.diff().T, countries, "New deaths by country by day log--plot")

def generate_absolute_plot(data, countries, title=None):
    data[data.index.isin(countries)].replace(np.nan, 0).T.plot(title=title)

def generate_log_plot(data, countries, title=None):
    data[data.index.isin(countries)].replace(np.nan, 0).T.plot(logy=True, title=title)
    
def generate_loglog_plot(data, countries, title=None):
    data[data.index.isin(countries)].replace(np.nan, 0).T.plot(loglog=True, title=title)
