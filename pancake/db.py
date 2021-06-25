import numpy as np
import sqlite3
import torch
from typing import Type
from multiprocessing.pool import ThreadPool

from .detector import Detector
from .detector.backends import Backend
from .tracker import BaseTracker

from .utils.common import fix_path
from .utils.parser import get_config

from .logger import setup_logger


l = setup_logger(__name__)


def setup_database(
    cfg: dict,
    detector: Type[Detector] = None,
    backend: Type[Backend] = None,
    tracker: Type[BaseTracker] = None,
):
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


class DataBase:
    """Encapsulates the Pancake database"""

    def __init__(self, db_name: str, relations: dict, inserts: dict, *args, **kwargs):
        """
        :param db_name (str):       Database filename
        :param relations (dict):    contains create queries for custom relationships
        :param inserts (dict):      holds corresponding insert queries
        """
        try:
            self.con = sqlite3.connect(db_name)
            self.relations = relations
            self.inserts = inserts
        except Error as e:
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
        """ 
        Creates tables for provided relationships in self.relations.
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
        """
        :param kwargs (dict): Holds objects for logging of the general setup. 
        """
        cursor = self.con.cursor()

        if "DETECTOR" in kwargs["DBT"]:
            detector: Type[Detector] = kwargs["DBT"]["DETECTOR"]

            # DETECTOR insert
            try:
                #!! ADAPT TO CUSTOM SCHEME IF NECESSARY!! 
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
                #!! ADAPT TO CUSTOM SCHEME IF NECESSARY!!
                for id, label in enumerate(detector.model.names):
                    cursor.execute(
                        self.inserts["CLASSES"], 
                        (id + 1, label)
                    )
            except Exception as e:
                l.info(f"{e}")
                l.info("Skip initial query for CLASSES")

        if "BACKEND" in kwargs["DBT"]:
            backend: Type[Backend] = kwargs["DBT"]["BACKEND"]

            # BACKEND insert
            try:
                #!! ADAPT TO CUSTOM SCHEME IF NECESSARY!!  
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
                #!! ADAPT TO CUSTOM SCHEME IF NECESSARY!! 
                cursor.execute(
                    self.inserts["TRACKER"],
                    (1, tracker.__class__.__name__),
                )
            except Exception as e:
                l.info(f"{e}")
                l.info("Skip initial query for TRACKER")

        self.con.commit()

    def insert_tracks(self, tracks: np.ndarray, timestamp: float):
        """
        :param tracks (np.ndarray): [tracks][x1, y1, x2, y2, centre x, centre y, id, cls]
        :param timestamp (float):     time.time() timestamp
        """
        cursor = self.con.cursor()

        # append timestamp to 0th column, 0 if timestamp is None, 69 is a dummy entry for cls
        insertable = np.c_[
            np.full((tracks.shape[0]), timestamp if timestamp else 0), 
            tracks,
            np.full((tracks.shape[0]), 69)
        ]

        #!! ADAPT TO CUSTOM SCHEME IF NECESSARY!! 
        # order: [id, ts, cx, cy, x1, y1, x2, y2, cls] (for "extended_db.yaml")
        new_order = [7, 0, 5, 6, 1, 2, 3, 4, 8]
        insertable = insertable[:, new_order]

        try:
            cursor.executemany(self.inserts["TRACKS"], insertable)
            self.con.commit()
        except Exception as e:
            l.error(f"{e} - bad query..")
