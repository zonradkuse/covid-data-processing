#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_data():
    confirmed = pd.read_csv("COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv")
    deaths = pd.read_csv("COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv")
    recovered = pd.read_csv("COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv")

    confirmed = confirmed.drop(["Lat", "Long", "Province/State"], axis=1)
    deaths = deaths.drop(["Lat", "Long", "Province/State"], axis=1)
    recovered = recovered.drop(["Lat", "Long", "Province/State"], axis=1)

    confirmed = confirmed.groupby("Country/Region").agg("sum")
    deaths = deaths.groupby("Country/Region").agg("sum")
    recovered = recovered.groupby("Country/Region").agg("sum")

    return confirmed, deaths, recovered

def generate_registered_plots(countries):
    cases, deaths, recovered = parse_data()

    for country in countries:
        tmp_data = np.array(cases[cases.index.isin([country])].values.tolist()[0])
        tmp_data = tmp_data[tmp_data>100]
        fit_length = np.min([tmp_data.size,1000])
        fit_data = np.polyfit(range(fit_length),np.log2(tmp_data[:fit_length]),1)
        doubling = 1/fit_data[0]
        plt.semilogy(range(tmp_data.size),tmp_data,label="{} {:.2f}".format(country,doubling))
    plt.legend()    

def generate_all_plots(countries):
    confirmed, deaths, recovered = parse_data()

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
