import logging
import math
import json
import datetime
import numbers

import numpy as np
import pandas as pd

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

class CovidTimeseriesModelSIR:

    def initialize_parameters(self, model_parameters):
        """Perform all of the necessary setup prior to the model's execution"""
        # want first and last days from the actual values in timeseries
        actual_values = model_parameters["timeseries"].sort_values("date")

        model_parameters["actual_init_date"] = actual_values.iloc[
            0, actual_values.columns.get_loc("date")
        ]

        if "override_model_start" in model_parameters:
            model_parameters["actual_end_date"] = model_parameters[
                "override_model_start"
            ]
        else:
            model_parameters["actual_end_date"] = actual_values.iloc[
                -1, actual_values.columns.get_loc("date")].to_pydatetime().date()


        # TODO: add check for earlier initial date parameter and adjust so that
        # we can start the run earlier than the last data point

        model_parameters["init_date"] = model_parameters["actual_end_date"]

        # Get the last day of the model based on the number of iterations and
        # length of the iterations
        duration = datetime.timedelta(days=(model_parameters["days_to_model"]))
        model_parameters["last_date"] = model_parameters["init_date"] + duration

        return model_parameters

    # get the largest key (intervention date) that is less than the init_date and reurn the relevant r0
    def get_latest_past_intervention(self, interventions, init_date):
        past_dates = [
            interevention_date
            for interevention_date in interventions.keys()
            if interevention_date <= init_date
        ]

        if len(past_dates) > 0:
            return interventions[max(past_dates)]
        else:
            return None

    def run_interventions(
        self, model_parameters, combined_df, seir_params, model_seir_init, r0
    ):
        ## for each intervention (in order)
        ## grab initial conditions (conditions at intervention date)
        ## adjust seir_params based on intervention
        ## run model from that date with initial conditions and new params
        ## merge new dataframes, keep old one as counterfactual for that intervention
        ## rinse, repeat
        interventions = model_parameters["interventions"]
        end_date = model_parameters["last_date"]

        counterfactuals = {}

        for date, intervention_params in interventions.items():
            date = pd.Timestamp(date)

            if (date >= model_parameters["init_date"]) and (date <= end_date):

                counterfactuals[date] = combined_df

                if intervention_params is None:
                    new_r0 = generate_r0(
                        model_seir_init, model_parameters["population"]
                    )

                    new_seir_params = brute_force_r0(
                        seir_params, new_r0, r0, model_parameters["population"]
                    )

                elif isinstance(intervention_params, numbers.Number):
                    new_r0 = intervention_params

                    new_seir_params = brute_force_r0(
                        seir_params, new_r0, r0, model_parameters["population"]
                    )
                else:
                    new_seir_params = {
                        k: (intervention_params[k] if k in intervention_params else v)
                        for (k, v) in seir_params.items()
                    }

                    new_r0 = generate_r0(
                        new_seir_params, model_parameters["population"]
                    )

                _logger.info(f"Intervention date: {date}")
                _logger.info(f"Intervention R-effective: {new_r0}")
                _logger.info(f"Intervention params: {json.dumps(new_seir_params)}")

                pop_dict = {
                    "total": model_parameters["population"],
                    "exposed": combined_df.loc[date, "exposed"],
                    "infected": combined_df.loc[date, "infected"],
                    "recovered": combined_df.loc[date, "recovered"],
                    "deaths": combined_df.loc[date, "dead"],
                }

                if model_parameters["model"] == "asymp":
                    pop_dict["asymp"] = combined_df.loc[date, "asymp"]

                    _logger.debug(f"check beta for right number of elements: {new_seir_params['beta']}")
                    new_seir_params["beta"] = L(list(new_seir_params["beta"]))
                    new_seir_params["beta"].A = new_seir_params["beta"][4]

                if model_parameters["model"] in ["seir", "asymp"]:
                    # this is a dumb way to do this, but it might work
                    combined_df.loc[:, "infected_tmp"] = (
                        combined_df.loc[:, "infected_a"]
                        + combined_df.loc[:, "infected_b"]
                        + combined_df.loc[:, "infected_c"]
                    )

                    combined_df.loc[:, "infected"].fillna(combined_df["infected_tmp"])

                    combined_df.drop("infected_tmp", axis=1, inplace=True)

                    pop_dict["infected_a"] = combined_df.loc[date, "infected_a"]
                    pop_dict["infected_b"] = combined_df.loc[date, "infected_b"]
                    pop_dict["infected_c"] = combined_df.loc[date, "infected_c"]

                print(pop_dict['exposed'])
                print(f"Populations: {json.dumps(pop_dict)}")
                _logger.info(f"Populations: {json.dumps(pop_dict)}")

                (data, steps, ret) = seir(pop_dict, model_parameters, **new_seir_params)

                new_df = dataframe_ify(data, date, end_date, steps,)

                early_combo_df = combined_df.copy().loc[:date]

                combined_df = early_combo_df.append(new_df, sort=True)

                if model_parameters["model"] in ["seir", "asymp"]:
                    # this is a dumb way to do this, but it might work
                    combined_df.loc[:, "infected_tmp"] = (
                        combined_df.loc[:, "infected_a"]
                        + combined_df.loc[:, "infected_b"]
                        + combined_df.loc[:, "infected_c"]
                    )

                    combined_df.loc[:, "infected"].fillna(combined_df["infected_tmp"])

                    combined_df.drop("infected_tmp", axis=1, inplace=True)

        return (combined_df, counterfactuals)

    def iterate_model(self, model_parameters):
        """The guts. Creates the initial conditions, and runs the SIR model for the
        specified number of iterations with the given inputs"""

        ## TODO: nice-to have - counterfactuals for interventions

        timeseries = model_parameters["timeseries"].sort_values("date")

        # calc values if missing
        timeseries.loc[:, ["cases", "deaths", "recovered"]] = timeseries.loc[
            :, ["cases", "deaths", "recovered"]
        ].fillna(0)

        timeseries["active"] = (
            timeseries["cases"] - timeseries["deaths"] - timeseries["recovered"]
        )

        # timeseries["active"] = timeseries["active"].fillna(timeseries["active_calc"])

        model_parameters["timeseries"] = timeseries

        model_parameters = self.initialize_parameters(model_parameters)

        timeseries["dt"] = pd.to_datetime(timeseries["date"]).dt.date
        timeseries.set_index("dt", inplace=True)
        timeseries.sort_index(inplace=True)

        init_date = model_parameters["init_date"]

        # load the initial populations
        pop_dict = {
            "total": model_parameters["population"],
            "infected": timeseries.loc[init_date, "active"],
            "recovered": timeseries.loc[init_date, "recovered"],
            "deaths": timeseries.loc[init_date, "deaths"],
        }

        if model_parameters["exposed_from_infected"]:
            pop_dict["exposed"] = (
                model_parameters["exposed_infected_ratio"] * pop_dict["infected"]
            )

        init_params = generate_epi_params(model_parameters)
        init_params["beta"].append(init_params["beta"][1])

        if model_parameters["interventions"] is not None:
            intervention_params = self.get_latest_past_intervention(
                model_parameters["interventions"], init_date
            )

            # keep these in case we need to go back due to a final
            # "return to normal" intervention
            model_seir_init = init_params.copy()

            if intervention_params is not None:
                if isinstance(intervention_params, numbers.Number):
                    init_params = brute_force_r0(
                        init_params,
                        intervention_params,
                        generate_r0(init_params, model_parameters["population"]),
                        model_parameters["population"],
                    )
                else:
                    init_params = {
                        k: (
                            intervention_params[k]
                            if k in intervention_params
                            else v
                        )
                        for (k, v) in init_params.items()
                    }

        r0 = generate_r0(init_params, model_parameters["population"])

        # sometimes the beta is one of these L things and so this breaks
        # but sometimes it isn't? I think?
        try:
            init_params["beta"] = L(list(init_params["beta"]))
            init_params["beta"].A = init_params["beta"][4]
        except AttributeError:
            pass


        _logger.info(f"Initial R0: {r0}")
        _logger.info(f"Initial params: {json.dumps(init_params)}")

        (data, steps, ret) = seir(pop_dict, model_parameters, **init_params)

        # this dataframe should start on the last day of the actual data
        # and have the same values for those initial days, so we combine it with
        # the slice of timeseries from the actual_init_date to actual_end_date - 1
        sir_df = dataframe_ify(
            data, model_parameters["init_date"], model_parameters["last_date"], steps,
        )

        if model_parameters["model"] == "seir":
            sir_df["infected"] = (
                sir_df["infected_a"] + sir_df["infected_b"] + sir_df["infected_c"]
            )

        sir_df["total"] = pop_dict["total"]

        timeseries["susceptible"] = model_parameters["population"] - (
            timeseries.active + timeseries.recovered + timeseries.deaths
        )

        actual_cols = ["active", "recovered", "deaths"]

        # kill last row that is initial conditions on SEIR
        actuals = timeseries.loc[:, actual_cols].head(-1)

        if "override_model_start" in model_parameters:
            actuals = actuals.loc[
                : model_parameters["override_model_start"],
            ]

        actuals["population"] = model_parameters["population"]

        # it wasn't a df thing, you can rip all this out
        actuals.rename(
            columns={"population": "total", "deaths": "dead", "active": "infected"},
            inplace=True,
        )

        actuals.index = pd.to_datetime(actuals.index, format="%Y-%m-%d")

        all_cols = model_parameters["model_cols"]

        actuals.reindex(columns=all_cols)
        sir_df.reindex(columns=all_cols)

        combined_df = pd.concat([actuals, sir_df])

        if model_parameters["interventions"] is not None:
            (combined_df, counterfactuals) = self.run_interventions(
                model_parameters, combined_df, init_params, model_seir_init, r0
            )

        # this should be done, but belt and suspenders for the diffs()
        combined_df.sort_index(inplace=True)
        combined_df.index.name = "date"
        combined_df.reset_index(inplace=True)

        combined_df["total"] = pop_dict["total"]

        if model_parameters["model"] == "seir":

            # make infected total represent the sum of the infected stocks
            # but don't overwrite the historical
            combined_df.loc[:, "infected_tmp"] = (
                combined_df.loc[:, "infected_a"]
                + combined_df.loc[:, "infected_b"]
                + combined_df.loc[:, "infected_c"]
            )

            # TODO: why isn't this working?
            combined_df["infected"].fillna(combined_df["infected_tmp"])

        elif model_parameters["model"] == "asymp":

            # make infected total represent the sum of the infected stocks
            # but don't overwrite the historical
            combined_df.loc[:, "infected_tmp"] = (
                combined_df.loc[:, "infected_a"]
                + combined_df.loc[:, "infected_b"]
                + combined_df.loc[:, "infected_c"]
                + combined_df.loc[:, "asymp"]
            )

            combined_df.loc[:, "infected"].fillna(combined_df["infected_tmp"])

            combined_df.drop("infected_tmp", axis=1, inplace=True)

        non_susceptible_cols = all_cols.copy()
        non_susceptible_cols.remove("total")
        non_susceptible_cols.remove("susceptible")

        # do this so that we can use the model cols vs. a bunch of ifs
        combined_df["non_susceptible"] = combined_df.loc[:, non_susceptible_cols].sum(
            axis=1
        )

        combined_df["susceptible"] = (
            combined_df["total"] - combined_df["non_susceptible"]
        )
        combined_df.drop("non_susceptible", axis=1, inplace=True)

        combined_df.fillna(0, inplace=True)

        combined_df["pct_change"] = combined_df.loc[:, "infected_b"].pct_change()
        combined_df["doubling_time"] = math.log(2) / combined_df["pct_change"]

        combined_df["beds"] = model_parameters["beds"]

        combined_df = combined_df.loc[
            (combined_df["date"] <= pd.Timestamp(model_parameters["last_date"])), :
        ]

        return [combined_df, ret]

    def forecast_region(self, model_parameters):
        cycle_series = self.iterate_model(model_parameters)

        return cycle_series
