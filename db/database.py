import sqlite3 as lite
import os

import sys
sys.path.insert(0,'../db')
from populate_images import populate_db_with_images
from flask import g


class DB(object):
    DB_ROOT = os.path.join("..", "db")
    DATABASE = os.path.join(DB_ROOT, "test.db")

    def __init__(self):
        if not os.path.isfile(self.DATABASE):
            conn = self.get_connection()
            self.__generate_tables(conn)
            self.__populate_tables(conn)

    def get_connection(self):
        db = getattr(g, '_database', None)
        if db is None:
            db = g._database = lite.connect(self.DATABASE)
        return db

    def __generate_tables(self, conn):
        with open(os.path.join(self.DB_ROOT, "init.sql"), "r") as generation_script:
            conn.execute("PRAGMA foreign_keys=on;")
            conn.commit()
            conn.executescript(generation_script.read())
            conn.commit()

    def __populate_tables(self, conn):
        # populate images
        image_list_path = os.path.join(self.DB_ROOT, "..", "data", "ddsm_raw_image_lists")
        populate_db_with_images(conn, image_list_path)
        conn.commit()
