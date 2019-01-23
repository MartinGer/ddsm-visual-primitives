from db.database import DB

def _is_doctor_existing(username):
    db = DB()
    conn = db.get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM doctor WHERE name='{}'".format(username))
    return c.fetchone() is not None


def _insert_doctor(username):
    insert_statement = "INSERT INTO doctor(name) VALUES (?)"
    db = DB()
    conn = db.get_connection()
    conn.execute(insert_statement, (username, ))
    conn.commit()


def insert_doctor_into_db_if_not_exists(username):
    if _is_doctor_existing(username):
        return
    _insert_doctor(username)
