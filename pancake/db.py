""" Module containing the Pancake Database. """
from typing import Type, Union

import numpy as np
import sqlite3
import time
import threading

from torch.jit import Error

from .detector import Detector
from .detector.backends import Backend
from .tracker import BaseTracker

from .utils.common import fix_path
from .utils.parser import get_config

from .logger import setup_logger


l = setup_logger(__name__)


class DataBase:
    def __init__(self, db_name: str, relations: dict, inserts: dict, *args, **kwargs):
        """ Encapsulates the Pancake database

        Args:
            db_name (str): Database filename
            relations (dict): Contains create queries for custom relationships
            inserts (dict): Holds corresponding insert queries

        Raises:
            ConnectionError: Raised when sqlite3 can't connect to the database
        """        
        try:
            self.con = sqlite3.connect(db_name, check_same_thread=False)
            self.relations = relations
            self.inserts = inserts
        except Exception as e:
            l.error(
                f"Error while connecting to the DB: {e} \n"
                "Proceed tracking without database logging"
            )
            raise ConnectionError

        assert (
            len(self.relations.items()) > 0
        ), "No relationships found to build database from!"

        self.create_tables()
        self.initial_insert(DBT=kwargs)  # Detector, Backend, Tracker

    def create_tables(self) -> None:
        """ Creates tables for provided relationships in self.relations.

        Raises:
            ConnectionError: Raised when an error occurs while trying to create \
                                table.
        """        
        cursor = self.con.cursor()

        for rel_name, query in self.relations.items():
            if type(query) is str:
                try:
                    cursor.execute(query)
                except Exception as e:
                    l.error(
                        f"Error occured while executing query for {rel_name}: {e} \n"
                        "Proceed tracking without database logging"
                    )
                    raise ConnectionError

    def initial_insert(self, *args, **kwargs) -> None:
        """ Method to log the static application setup.
        """        
        cursor = self.con.cursor()

        if "DETECTOR" in kwargs["DBT"]:
            detector: Type[Detector] = kwargs["DBT"]["DETECTOR"]

            # DETECTOR insert
            try:
                #!! ADAPT TO CUSTOM SCHEMA IF NECESSARY!!
                cursor.execute(
                    self.inserts["DETECTOR"],
                    (
                        1,
                        detector.__class__.__name__,
                        detector.weights if hasattr(detector, "weights") else "NULL",
                    ),
                )
            except Exception as e:
                l.info(f"{e}")
                l.info("Skip initial query for DETECTOR")

            # CLASSES insert
            try:
                #!! ADAPT TO CUSTOM SCHEMA IF NECESSARY!!
                for id, label in enumerate(detector.model.names):
                    cursor.execute(self.inserts["CLASSES"], (id + 1, label))
            except Exception as e:
                l.info(f"{e}")
                l.info("Skip initial query for CLASSES")

        if "BACKEND" in kwargs["DBT"]:
            backend: Type[Backend] = kwargs["DBT"]["BACKEND"]

            # BACKEND insert
            try:
                #!! ADAPT TO CUSTOM SCHEMA IF NECESSARY!!
                cursor.execute(
                    self.inserts["BACKEND"],
                    (1, backend.__class__.__name__),
                )
            except Exception as e:
                l.info(f"{e}")
                l.info("Skip initial query for BACKEND")

        if "TRACKER" in kwargs["DBT"]:
            tracker: Type[BaseTracker] = kwargs["DBT"]["TRACKER"]

            # TRACKER insert
            try:
                #!! ADAPT TO CUSTOM SCHEMA IF NECESSARY!!
                cursor.execute(
                    self.inserts["TRACKER"],
                    (1, tracker.__class__.__name__),
                )
            except Exception as e:
                l.info(f"{e}")
                l.info("Skip initial query for TRACKER")

        self.con.commit()

    def insert_tracks(self, *args, **kwargs):
        """ Calls a thread to execute the query with provided arguments.
        """        
        t = threading.Thread(target=self.run_insert_tracks, args=args, kwargs=kwargs)
        t.start()

    def run_insert_tracks(self, tracks: np.ndarray, timestamp: float):
        """ Executes query with provided tracks matrix.

        If no timestamp is provided, we take the current timestamp.
        Reorders the matrix to match the database relation in order to be
        inserted via .executemany().
        Skips the query when an error occurs.

        Args:
            tracks (np.ndarray): Tracks on [x1, y1, x2, y2, centre x, centre y, id, cls]
            timestamp (float): A timestamp of format time.time() timestamp
        """        
        if len(tracks) < 1:
            l.debug("Tracks are empty, skipping insert.")
            return

        if not timestamp:
            timestamp = time.time()
        cursor = self.con.cursor()

        # append timestamp to 0th column, 0 if timestamp is None, 69 is a dummy entry for cls
        insertable = np.c_[
            np.full((tracks.shape[0]), timestamp if timestamp else 0),
            tracks,
            # np.full((tracks.shape[0]), 69),
        ]

        #!! ADAPT TO CUSTOM SCHEMA IF NECESSARY!!
        # order: [id, ts, cx, cy, x1, y1, x2, y2, cls] (for "extended_db.yaml")
        new_order = [7, 0, 5, 6, 1, 2, 3, 4, 8]
        insertable = insertable[:, new_order]

        try:
            cursor.executemany(self.inserts["TRACKS"], insertable)
            self.con.commit()
        except Exception as e:
            l.error(f"{e} - bad query..")


def setup_database(
    cfg: dict,
    detector: Type[Detector] = None,
    backend: Type[Backend] = None,
    tracker: Type[BaseTracker] = None,
) -> Union[DataBase, None]:
    """Helper function to set up the Pancake database.

    Description:
        The different objects are parsed in order to log the static application
        setup. They are inserted during the initialization procedure of the database.

    Args:
        cfg (dict): Dictionary containing configurations.
        detector (Type[Detector], optional): A detector instance. Defaults to None.
        backend (Type[Backend], optional): A backend instance. Defaults to None.
        tracker (Type[BaseTracker], optional): A tracker instance. Defaults to None.

    Returns:
        Union[DataBase, None]: An instance of the pancake database if the flag was set,
                                None otherwise.
    """    
    db_schema = get_config(config_file=fix_path(cfg.SCHEME_PATH)).PANCAKE_DB

    relations = db_schema.RELATIONS
    inserts = db_schema.INSERTS

    return (
        DataBase(
            cfg.FILENAME,
            relations,
            inserts,
            DETECTOR=detector,
            BACKEND=backend,
            TRACKER=tracker,
        )
        if cfg.STORE
        else None
    )