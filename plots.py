#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

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

def generate_plots(countries):
    confirmed, deaths, recovered = parse_data()

    death_rate = deaths/confirmed
    recovery_rate = recovered/confirmed

    print(f"A total of {len(confirmed)} countries confirmed at least one case of covid-19")

    confirmed[confirmed.index.isin(countries)].T.plot(title="Confirmed cases")
    deaths[deaths.index.isin(countries)].T.plot(title="Deaths")
    recovered[recovered.index.isin(countries)].T.plot(title="Recovered cases")

    selected_death_rate = death_rate[death_rate.index.isin(countries)]
    selected_death_rate = selected_death_rate.replace(np.nan, 0)
    selected_recovery_ratio = recovery_rate[recovery_rate.index.isin(countries)]
    selected_recovery_ratio = selected_recovery_ratio.replace(np.nan, 0)

    def death_rate_by_country(country):
        return selected_death_rate[selected_death_rate.index.isin([country])].T

    for country in countries:
        death_rate_country = death_rate_by_country(country)
        print(f"Mean death rate for {country}: {float(death_rate_country.mean()):.4f} (+-{float(death_rate_country.std()):.4f} std)")

    selected_death_rate[selected_death_rate.index.isin(countries)].T.plot(title="Death rate by day and country")
    selected_recovery_ratio.T.plot(title="Recovery rate per day by country")

    log_confirmed = np.log(confirmed.replace(0, 1))

    confirmed[confirmed.index.isin(countries)].T.plot(logy=True, title="Total confirmed log-plot")
    deaths[deaths.index.isin(countries)].T.plot(logy=True, title="Total deaths log-plot")
