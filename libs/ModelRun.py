import logging
import math
import json
import datetime
import numbers

import numpy as np
import pandas as pd

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


class ModelRun:
    def __init__(self, state, country="USA", county=None):
        self.state = state
        self.country = country
        self.county = county

        # define constants used in model parameter calculations
        self.observed_daily_growth_rate = 1.17
        self.days_to_model = 365
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
            self.beds = beds_data.get_beds_by_country_state(self.country, self.state)
            self.timeseries = timeseries.get_data(state=self.state)
        else:
            # do county thing
            pass

        return

    def set_epi_model(self, epi_model_type):
        if epi_model_type == "seir":
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

            from libs.CovidTimeseriesModelSIR import CovidTimeseriesModelSIR as Model
            from libs.epi_models import HarvardEpi as EpiModel

        elif epi_model_type == "asymp":
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

            from libs.CovidTimeseriesModelASYMP import CovidTimeseriesModelSIR as Model
            from libs.epi_models import TalusSEIR as EpiModel

        self.epi_params = EpiModel.generate_epi_params_from_mr(self)

        self.r0 = EpiModel.generate_r0(init_params, model_parameters["population"])

    def generate_intervention_R_effective(
        self, new_params, effective_date, lifted_date
    ):
        return

    def generate_intervention_from_params(
        self, new_params, effective_date, lifted_date
    ):

        return intervention
