import pandas as pd
from datetime import datetime
import numpy as np
from typing import Tuple, Dict, Any, List


class Team:
    """
    A class representing a team and its attribute entries.

    Attributes:
        id_code (int): The ID code of the team.
        attribute_entries (DataFrame): The attribute entries for the team.

    Methods:
        get_data(data: DataFrame, id_name: str = 'team_api_id') -> None:
            Retrieves and stores the attribute entries for the team from the given DataFrame.
    """

    def __init__(self, id_code):
        self.id_code = id_code
        self.attribute_entries: pd.DataFrame = None

    def get_data(self, data: pd.DataFrame, id_name="team_api_id"):
        """
        Retrieves and stores the attribute entries for the team from the given DataFrame.

        Args:
            data (DataFrame): The DataFrame containing the attribute entries.
            id_name (str, optional): The name of the ID column to filter the entries
            (default: 'team_api_id')

        Returns:
            None
        """
        self.attribute_entries = data.loc[data[id_name] == self.id_code]

    def get_latest_entry(
        self, date: str, cols: list[str], merge_id: str = "team_api_id"
    ) -> pd.DataFrame:
        """
        Returns the latest attribute entry before the specified date.

        Args:
            date (str): The date to compare against in "YYYY-MM-DD" format.

        Returns:
            DataFrame: The latest attribute entry before the specified date.

        """
        entries_before_date = self.attribute_entries[
            self.attribute_entries["date"] < date
        ]
        latest_entry = entries_before_date[
            entries_before_date["date"] == entries_before_date["date"].max()
        ][[merge_id] + cols].set_index(merge_id)

        if len(latest_entry) == 0:
            latest_entry = latest_entry.reindex([self.id_code])
        return latest_entry


class MatchPlayers:
    """
    A class for handling player data and calculating attribute differences in a match.

    Attributes:
        match_data (dict): A dictionary containing the match data in key-value pairs.
        home_players (dict): A dictionary containing the home team's players and goalkeeper.
        away_players (dict): A dictionary containing the away team's players and goalkeeper.
        away_player_pos (dict[int, tuple]): A dictionary mapping player positions (1 to 11) to (X, Y) coordinates for the away team.
        home_player_pos (dict[int, tuple]): A dictionary mapping player positions (1 to 11) to (X, Y) coordinates for the home team.

    Methods:
        __init__(): Initializes the MatchPlayers class with empty data.
        get_data(data: pd.DataFrame): Sets the match_data attribute with a given pandas DataFrame.
        get_player_positions(): Populates the home_player_pos and away_player_pos dictionaries with player positions.
        get_player_ids(): Populates the home_players and away_players dictionaries with Player instances based on player IDs.
        calculate_attribute_difference(attribute: str) -> float: Calculates the difference in the specified attribute
                                                               between home and away teams' players.
        export_player_attributes(cols: List[str], how: str = "all") -> dict: Exports player attributes as a dictionary
                                                                            based on the specified method ('all', 'diff',
                                                                            'avg_diff', 'avg').

    Class Player:
        A nested class representing a player with their attributes.

        Attributes:
            player_id (str): The unique identifier of the player.
            attributes (dict): A dictionary containing the player's attributes.

        Methods:
            __init__(player_id: str): Initializes the Player class with the player's ID.
            get_player_attributes(player_data: pd.DataFrame, date, player_id_name: str = "player_api_id"):
                Retrieves the player's attributes from player_data DataFrame based on the provided date.
    """

    def __init__(self):
        self.match_data: dict = None
        self.home_players: dict
        self.away_players: dict
        self.away_player_pos: dict[int, tuple]
        self.home_player_pos: dict[int, tuple]

    class Player:
        def __init__(self, player_id):
            self.player_id: str = player_id
            self.attributes: dict

        def get_player_attributes(
            self, player_data: pd.DataFrame, date, player_id_name: str = "player_api_id"
        ):
            entries = player_data.loc[player_data[player_id_name] == self.player_id]
            entries_before_date = entries.loc[entries["date"] < date]
            latest_entry = (
                entries_before_date[
                    entries_before_date["date"] == entries_before_date["date"].max()
                ]
                .squeeze()
                .to_dict()
            )
            self.attributes = latest_entry

    def get_data(self, data: pd.DataFrame):
        self.match_data = data.to_dict()

    def get_player_positions(self):
        self.home_player_pos = {}
        self.away_player_pos = {}
        for i in np.arange(1, 12):
            self.home_player_pos[i] = (
                self.match_data["home_player_X" + str(i)],
                self.match_data["home_player_Y" + str(i)],
            )
            self.away_player_pos[i] = (
                self.match_data["away_player_X" + str(i)],
                self.match_data["away_player_Y" + str(i)],
            )

    def get_player_ids(self):
        # initiate empty dictionaries
        self.home_players = {}
        self.away_players = {}

        # Get home player column names and goalkeeper column names
        home_players_num = ["home_player_" + str(i) for i in np.arange(1, 12)]
        goaly_home_num = "home_player_" + str(
            next(key for key, val in self.home_player_pos.items() if val == (1, 1))
        )
        home_players_num.remove(goaly_home_num)

        # Get away player column names and goalkeeper column names
        away_players_num = ["away_player_" + str(i) for i in np.arange(1, 12)]
        goaly_away_num = "away_player_" + str(
            next(key for key, val in self.away_player_pos.items() if val == (1, 1))
        )
        away_players_num.remove(goaly_away_num)

        # Initiate a Player class with it's id for every home player and goalkeeper separately
        self.home_players["players"] = [
            self.Player(self.match_data[player]) for player in home_players_num
        ]
        self.home_players["goaly"] = self.Player(self.match_data[goaly_home_num])

        # Initiate a Player class with it's id for every away player and goalkeeper separately
        self.away_players["players"] = [
            self.Player(self.match_data[player]) for player in away_players_num
        ]
        self.away_players["goaly"] = self.Player(self.match_data[goaly_away_num])

    def calculate_attribute_difference(self, attribute):
        try:
            home_avg = sum(
                player.attributes[attribute] for player in self.home_players["players"]
            )
            away_avg = sum(
                player.attributes[attribute] for player in self.away_players["players"]
            )
            difference = home_avg - away_avg
            return difference
        except Exception:
            return np.nan

    def export_player_attributes(self, cols, how: str = "all"):
        atts = {}

        if how == "all":
            # Add home player attributes
            for i, player in enumerate(self.home_players["players"]):
                for col in cols:
                    atts[col + "_H_" + str(i + 1)] = player.attributes[col]
            for col in cols:
                atts[col + "_H_gk"] = self.home_players["goaly"].attributes[col]

            # Add away player attributes
            for i, player in enumerate(self.away_players["players"]):
                for col in cols:
                    atts[col + "_A_" + str(i + 1)] = player.attributes[col]
            for col in cols:
                atts[col + "_A_gk"] = self.away_players["goaly"].attributes[col]

        if how == "diff":
            # Add home player attributes
            for i, (player_h, player_a) in enumerate(
                zip(self.home_players["players"], self.away_players["players"])
            ):
                for col in cols:
                    try:
                        val = player_h.attributes[col] - player_a.attributes[col]
                    except:
                        val = np.nan
                    atts[col + "_dif_" + str(i + 1)] = val
            for col in cols:
                try:
                    val = (
                        self.home_players["goaly"].attributes[col]
                        - self.away_players["goaly"].attributes[col]
                    )
                except:
                    val = np.nan
                atts[col + "_dif_gk"] = val

        if how == "avg_diff":
            for col in cols:
                atts[col + "_avg_diff"] = self.calculate_attribute_difference(col) / 10
                try:
                    val = (
                        self.home_players["goaly"].attributes[col]
                        - self.away_players["goaly"].attributes[col]
                    )
                except:
                    val = np.nan
                atts[col + "_avg_diff_gk"] = val

        if how == "avg":
            # Add home player attributes
            for col in cols:
                att = 0
                for player in self.home_players["players"]:
                    try:
                        att += player.attributes[col]
                    except:
                        att = np.nan
                atts[col + "_H_avg"] = att / 10
            for col in cols:
                atts[col + "_H_gk"] = self.home_players["goaly"].attributes[col]

            # Add away player attributes
            for col in cols:
                att = 0
                for player in self.away_players["players"]:
                    try:
                        att += player.attributes[col]
                    except:
                        att = np.nan
                atts[col + "_A_avg"] = att / 10
            for col in cols:
                atts[col + "_A_gk"] = self.away_players["goaly"].attributes[col]
        return atts


def outcome_guess_prob_dif(row: pd.Series, coef_a: float, coef_b: float) -> str:
    """
    Predicts the match outcome based on the difference between win and loss probabilities.

    Parameters:
        row (pd.Series): A pandas Series representing a row of data containing 'win' and 'loss' probabilities.
        coef_a (float): The threshold coefficient for predicting a 'Home Win' outcome.
        coef_b (float): The threshold coefficient for predicting a 'Home Loss' outcome.

    Returns:
        str: The predicted match outcome, which can be 'Home Win', 'Home Loss', or 'Tie'.
    """
    dif = row["win"] - row["loss"]
    if dif > coef_a:
        output = "Home Win"
    elif dif < -coef_b:
        output = "Home Loss"
    else:
        output = "Tie"
    return output


def classifier_train_prob_dif(
    params: Dict[str, float], prob_data: pd.DataFrame, y_data: pd.Series
) -> np.ndarray:
    """
    Trains a classifier based on win and loss probabilities using threshold coefficients.

    Parameters:
        params (dict): A dictionary containing the threshold coefficients 'coef_a' and 'coef_b'.
        prob_data (pd.DataFrame): A pandas DataFrame containing 'win' and 'loss' probabilities.
        y_data (pd.Series): A pandas Series representing the actual match outcomes.

    Returns:
        np.ndarray: An array of integers indicating whether the predictions match the actual outcomes.
    """
    coef_a = params["coef_a"]
    coef_b = params["coef_b"]
    guess = prob_data.apply(outcome_guess_prob_dif, axis=1, args=(coef_a, coef_b))
    sum_false = ~(guess.values == y_data.values)
    return sum_false.astype(int)


def outcome_guess_prob_win(x: float, coef_win: float, coef_loss: float) -> str:
    """
    Assigns a match outcome value based on the probability of home team win
    using two threshold coefficients.

    Parameters:
        x (float): The probability of home team win.
        coef_win (float): The threshold coefficient for predicting a 'Home Win' outcome.
        coef_loss (float): The threshold coefficient for predicting a 'Home Loss' outcome.

    Returns:
        str: The predicted match outcome, which can be 'Home Win', 'Home Loss', or 'Tie'.
    """
    if x >= 1 - coef_win:
        y = "Home Win"
    elif x <= coef_loss:
        y = "Home Loss"
    else:
        y = "Tie"
    return y


def classifier_train_prob_win(
    params: Dict[str, float], probs: pd.Series, y_data: pd.Series
) -> np.ndarray:
    """
    Trains a classifier based on home win probabilities and threshold coefficients.

    Parameters:
        params (dict): A dictionary containing the threshold coefficients 'coef_win' and 'coef_loss'.
        probs (pd.Series): A pandas Series containing home win probabilities.
        y_data (pd.Series): A pandas Series representing the actual match outcomes.

    Returns:
        np.ndarray: An array of integers indicating whether the predictions match the actual outcomes.
    """
    coef_win = params["coef_win"]
    coef_loss = params["coef_loss"]
    guess = probs.apply(outcome_guess_prob_win, args=(coef_win, coef_loss))
    sum_false = ~(guess.values == y_data.values)
    return sum_false.astype(int)


def bet_home(row: Dict[str, Any]) -> float:
    """
    Calculate the profit for a bet on a home win.

    Parameters:
        row (dict): A dictionary representing the row of data with the following keys:
                    - 'home_win_pred' (int): 1 if the prediction is a home win, 0 otherwise.
                    - 'outcome' (str): The actual outcome of the match ('Home Win', 'Home Loss', or 'Tie').
                    - 'home_win_coef' (float): The coefficient (odds) for a home win.

    Returns:
        float: The profit (or loss if negative) from the bet on a home win.
    """

    profit = 0
    if row["home_win_pred"] == 1:
        if row["outcome"] == "Home Win":
            profit = row["home_win_coef"] * 100
        else:
            profit = -100
    return profit


def bet_away(row: Dict[str, Any]) -> float:
    """
    Calculate the profit for a bet on an away win.

    Parameters:
        row (dict): A dictionary representing the row of data with the following keys:
                    - 'away_win_pred' (int): 1 if the prediction is an away win, 0 otherwise.
                    - 'outcome' (str): The actual outcome of the match ('Home Win', 'Home Loss', or 'Tie').
                    - 'away_win_coef' (float): The coefficient (odds) for an away win.

    Returns:
        float: The profit (or loss if negative) from the bet on an away win.
    """

    profit = 0
    if row["away_win_pred"] == 1:
        if row["outcome"] == "Home Loss":
            profit = row["away_win_coef"] * 100
        else:
            profit = -100
    return profit


def bet_tie(row: Dict[str, Any]) -> float:
    """
    Calculate the profit for a bet on a tie.

    Parameters:
        row (dict): A dictionary representing the row of data with the following keys:
                    - 'tie_pred' (str): 'Tie' if the prediction is a tie, otherwise another value.
                    - 'outcome' (str): The actual outcome of the match ('Home Win', 'Home Loss', or 'Tie').
                    - 'tie_coef' (float): The coefficient (odds) for a tie.

    Returns:
        float: The profit (or loss if negative) from the bet on a tie.
    """

    profit = 0
    if row["tie_pred"] == "Tie":
        if row["outcome"] == "Tie":
            profit = row["tie_coef"] * 100
        else:
            profit = -100
    return profit
