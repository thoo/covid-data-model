import logging
import math
import json
import datetime
import numbers

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from libs.CovidDatasets import JHUDataset as LegacyJHUDataset
from libs.datasets import JHUDataset
from libs.datasets import FIPSPopulation
from libs.datasets import DHBeds
from libs.datasets.dataset_utils import AggregationLevel

# from .epi_models.HarvardEpi import (
from .epi_models.TalusSEIR import (
    seir,
    dataframe_ify,
    generate_epi_params,
    generate_r0,
    brute_force_r0,
    L,
)

_logger = logging.getLogger(__name__)

### TODO: need to tie together all the interventions and actuals and get a final
# results set
class ModelRun:
    def __init__(self, state, country="USA", county=None):
        self.state = state
        self.country = country
        self.county = county

        # define constants used in model parameter calculations
        self.observed_daily_growth_rate = 1.17
        self.days_to_model = 365

        # when going back to test hypothetical intervnetions in the past,
        # use this to start the data from this date instead of latest reported
        self.override_model_start = False

        ## Variables for calculating model parameters Hill -> our names/calcs
        # IncubPeriod: Average incubation period, days - presymptomatic_period
        # DurMildInf: Average duration of mild infections, days - duration_mild_infections
        # FracMild: Average fraction of (symptomatic) infections that are mild - (1 - hospitalization_rate)
        # FracSevere: Average fraction of (symptomatic) infections that are severe - hospitalization_rate * hospitalized_cases_requiring_icu_care
        # FracCritical: Average fraction of (symptomatic) infections that are critical - hospitalization_rate * hospitalized_cases_requiring_icu_care
        # CFR: Case fatality rate (fraction of infections that eventually result in death) - case_fatality_rate
        # DurHosp: Average duration of hospitalization (time to recovery) for individuals with severe infection, days - hospital_time_recovery
        # TimeICUDeath: Average duration of ICU admission (until death or recovery), days - icu_time_death

        # LOGIC ON INITIAL CONDITIONS:
        # hospitalized = case load from timeseries on last day of data / 4
        # mild = hospitalized / hospitalization_rate
        # icu = hospitalized * hospitalized_cases_requiring_icu_care
        # expoosed = exposed_infected_ratio * mild

        # Time before exposed are infectious (days)
        self.presymptomatic_period = 3

        # Pecentage of asymptomatic, infectious [A] people
        # out of all those who are infected
        # make 0 to remove this stock
        #'asymp_to_mild_ratio': 0.5,
        self.percent_asymp = 0.3

        # Time mildly infected people stay sick before
        # hospitalization or recovery (days)
        self.duration_mild_infections = 6

        # Time asymptomatically infected people stay
        # infected before recovery (days)
        self.duration_asymp_infections = 6

        # Duration of hospitalization before icu or
        # recovery (days)
        self.hospital_time_recovery = 6

        # Time from ICU admission to death (days)
        self.icu_time_death = 8

        ####################################################
        # BETA: transmission rate (new cases per day).
        # The rate at which infectious cases of various
        # classes cause secondary or new cases.
        ####################################################
        #
        # Transmission rate of infected people with no
        # symptoms [A] (new cases per day)
        # This is really beta * N, but it's easier to talk about this way
        # Default: 0.6
        # Current: Calculated based on observed doubling
        # rates
        self.beta_asymp = 0.3 + ((self.observed_daily_growth_rate - 1.09) / 0.02) * 0.05
        #
        # Transmission rate of infected people with mild
        # symptoms [I_1] (new cases per day)
        # This is really beta * N, but it's easier to talk about this way
        # Default: 0.6
        # Current: Calculated based on observed doubling
        # rates
        self.beta = 0.3 + ((self.observed_daily_growth_rate - 1.09) / 0.02) * 0.05
        #
        # Transmission rate of infected people with severe
        # symptoms [I_2] (new cases per day)
        # This is really beta * N, but it's easier to talk about this way
        # Default: 0.1
        self.beta_hospitalized = 0.1
        #
        # Transmission rate of infected people with severe
        # symptoms [I_3] (new cases per day)
        # This is really beta * N, but it's easier to talk about this way
        # Default: 0.1
        self.beta_icu = 0.1
        #
        ####################################################

        self.hospitalization_rate = 0.2
        self.hospitalized_cases_requiring_icu_care = 0.25

        # changed this from CFR to make the calc of mu clearer
        self.death_rate_for_critical = 0.4

        # CFR is calculated from the input parameters vs. fixed
        self.case_fatality_rate = (
            (1 - self.percent_asymp)
            * self.hospitalization_rate
            * self.hospitalized_cases_requiring_icu_care
            * self.death_rate_for_critical
        )

        # if true we calculatied the exposed initial stock from the infected number vs. leaving it at 0
        self.exposed_from_infected = True
        self.exposed_infected_ratio = 1

        self.hospital_capacity_change_daily_rate = 1.05
        self.max_hospital_capacity_factor = 2.07
        self.initial_hospital_bed_utilization = 0.6
        self.case_fatality_rate_hospitals_overwhelmed = (
            self.hospitalization_rate * self.hospitalized_cases_requiring_icu_care
        )

        self.interventions = {}

    # use only if you're doing a stand-alone run, if you're doing a lot of regions
    # then grab all the data and just call get_data_subset for each run
    def get_data(self, min_date):
        # TODO rope in counties

        self.min_date = min_date

        beds = DHBeds.local().beds()
        population_data = FIPSPopulation.local().population()

        timeseries = (
            JHUDataset.local()
            .timeseries()
            .get_subset(
                AggregationLevel.STATE,
                after=min_date,
                country=self.country,
                state=self.state,
            )
        )

        if self.county is None:
            self.population = population_data.get_state_level(self.country, self.state)
            self.beds = beds_data.get_beds_by_country_state(self.country, self.state)
            self.timeseries = timeseries.get_data(state=self.state)
        else:
            # do county thing
            pass

        if self.override_model_start is False:
            self.start_date = self.timeseries.loc[
                (self.timeseries["cases"] > 0), "date"
            ].max()
        else:
            self.start_date = self.override_model_start

        return

    def get_data_subset(
        self, beds_data, population_data, timeseries, min_date,
    ):
        # TODO rope in counties

        self.min_date = min_date

        timeseries = timeseries.get_subset(
            AggregationLevel.STATE,
            after=min_date,
            country=self.country,
            state=self.state,
        )

        if self.county is None:
            self.population = population_data.get_state_level(self.country, self.state)
            self.beds = beds_data.get_state_level(self.state)
            self.timeseries = timeseries.get_data(state=self.state)
        else:
            # do county thing
            pass

        if self.override_model_start is False:
            self.start_date = self.timeseries.loc[
                (self.timeseries["cases"] > 0), "date"
            ].max()
        else:
            self.start_date = self.override_model_start

        # get a series of the relevant row in the db
        self.past_data = self.timeseries.loc[
            (self.timeseries.date == self.start_date), :
        ].iloc[0]

        return

    def set_epi_model(self, epi_model_type):
        if epi_model_type == "seir":
            self.model_type = "seir"

            self.model_cols = [
                "total",
                "susceptible",
                "exposed",
                "infected",
                "infected_a",
                "infected_b",
                "infected_c",
                "recovered",
                "dead",
            ]

            from libs.epi_models import HarvardEpi as EpiModel

        elif epi_model_type == "asymp":
            self.model_type = "asymp"

            self.model_cols = [
                "total",
                "susceptible",
                "exposed",
                "infected",
                "asymp",
                "infected_a",
                "infected_b",
                "infected_c",
                "recovered",
                "dead",
            ]

            from libs.epi_models.TalusSEIRClass import EpiRun

        self.epi_run = EpiRun("base", self)
        self.epi_run.generate_epi_params()
        self.epi_run.generate_initial_conditions()
        self.epi_run.seir()
        self.epi_run.dataframe_ify()

    def add_intervention(self, intervention):
        if self.model_type == "seir":
            from libs.epi_models.HarvardEpi import Intervention

        elif self.model_type == "asymp":
            from libs.epi_models.TalusSEIRClass import Intervention

        intervention["initial_conditions"] = self.epi_run.results_df[
            (self.epi_run.results_df.index == intervention["start_date"])
        ].iloc[0]

        intervention_name = f"intervention_{self.state}_{intervention['name']}"

        self.interventions[intervention_name] = Intervention(intervention, self)


def plot_df(df_to_plot, cols, title="", y_max=8000000):
    cols.append("date")

    df_to_plot = df_to_plot.loc[:, cols]

    x_dates = df_to_plot["date"].dt.strftime("%Y-%m-%d").sort_values().unique()

    df_to_plot.set_index("date", inplace=True)

    stacked = df_to_plot.stack().reset_index()

    stacked.columns = ["date", "Population", "Number of people"]

    plt.figure(figsize=(15, 8))

    plt.ylim(0, y_max)

    plt.title(title)

    df_plt = sb.lineplot(x="date", y="Number of people", hue="Population", data=stacked)
    # df_plt.set_xticklabels(labels=x_dates, rotation=45, ha='right')

    return df_plt


def prep_plot(prep_df, chart_cols, title, y_max=8000000):
    prep_df["date"] = pd.to_datetime(prep_df["date"])

    first_case_date = prep_df.loc[(prep_df.infected > 0), "date"].min()
    peak_date = prep_df.loc[(prep_df.infected_b == prep_df.infected_b.max())][
        "date"
    ].values[0]
    peak = prep_df.loc[(prep_df.infected_b == prep_df.infected_b.max())][
        "infected_b"
    ].values[0]

    icu_peak_date = prep_df.loc[(prep_df.infected_c == prep_df.infected_c.max())][
        "date"
    ].values[0]
    icu_peak = prep_df.loc[(prep_df.infected_c == prep_df.infected_c.max())][
        "infected_c"
    ].values[0]

    deaths = prep_df.loc[:, "dead"].max()

    print("first case")
    print(first_case_date)
    print("peak in hospitalizations")
    print(peak_date)
    print(f"{peak:,}")
    print("peak in icu")
    print(icu_peak_date)
    print(f"{icu_peak:,}")
    print("deaths")
    print(f"{deaths:,}")

    plot_df(
        prep_df,
        chart_cols,
        f"{title}. Peak hospitalizations: {int(peak):,}. Deaths: {int(deaths):,}",
        y_max,
    )


def report_months(df):
    date_list = [
        datetime.datetime(2020, 5, 1).date(),
        datetime.datetime(2020, 6, 1).date(),
        datetime.datetime(2020, 7, 1).date(),
        datetime.datetime(2020, 8, 1).date(),
        datetime.datetime(2020, 9, 1).date(),
        datetime.datetime(2020, 10, 1).date(),
        datetime.datetime(2020, 11, 1).date(),
        datetime.datetime(2020, 12, 1).date(),
    ]

    cols = {
        "date": "Date",
        "infected_a": "Infected",
        "infected_b": "Hospitalized",
        "infected_c": "ICU",
        "dead": "Deaths",
    }

    report_df = df.loc[(df["date"].isin(date_list)), list(cols.keys())]

    report_df.rename(columns=cols, inplace=True)

    return report_df.T
