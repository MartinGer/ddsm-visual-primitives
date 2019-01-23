import sqlite3 as lite
import os

import sys
sys.path.insert(0,'../db')
from populate_images import populate_db_with_images

class DB(object):
    class __DB:
        def __init__(self, filename, db_root="../db/"):
            self._db_root = db_root
            db_file_path = os.path.join(self._db_root, filename)
            if os.path.isfile(db_file_path):
                self.__conn = lite.connect(db_file_path)
            else:
                self.__conn = lite.connect(db_file_path)
                self.__generate_tables()
                self.__populate_tables()

        def get_connection(self):
            return self.__conn

        def __generate_tables(self):
            with open(os.path.join(self._db_root, "init.sql"), "r") as generation_script:
                self.__conn.execute("PRAGMA foreign_keys=on;")
                self.__conn.commit()
                self.__conn.executescript(generation_script.read())
                self.__conn.commit()

        def __populate_tables(self):
            # populate images
            image_list_path = os.path.join(self._db_root, "..", "data", "ddsm_raw_image_lists")
            populate_db_with_images(self.__conn, image_list_path)

            # TODO: development only, populate with real nets later
            self.__conn.execute("INSERT INTO net(id, net, filename) VALUES('id_placeholder', 'resnet152', 'filename_placeholder');")
            self.__conn.commit()

    _instance = None

    def __init__(self):
        if not DB._instance:
            DB._instance = DB.__DB("test.db")

    def __getattr__(self, name):
        return getattr(self._instance, name)
