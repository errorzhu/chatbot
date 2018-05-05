# coding:utf-8

import sqlite3
import datetime
import random
import os

tmp_path = os.path.split(os.path.realpath(__file__))[0]
# print(os.path.dirname(tmp_path))
db_file = os.path.join(tmp_path, 'dialog.db')


class DbClient():
    def __init__(self):
        self.connection = sqlite3.connect(db_file, check_same_thread=False)

    def connect(self):
        self.connection = sqlite3.connect(db_file, check_same_thread=False)

    def close(self):
        self.connection.close()

    def get(self):
        cursor = self.connection.cursor()
        proxys = cursor.execute('SELECT ip FROM proxy').fetchall()
        result = random.choice(proxys)[0]
        cursor.close()
        return result

    def get_all(self):
        cursor = self.connection.cursor()
        time = datetime.datetime.now().strftime('%Y-%m-%d')
        print(time)
        proxys = cursor.execute('SELECT text FROM dialog where time = "%s"' % time).fetchall()
        cursor.close()
        return [x[0] for x in proxys]

    def put(self, data):
        self.connection.executemany(
            'INSERT INTO dialog VALUES (?,?)',
            [(data, datetime.datetime.now().strftime('%Y-%m-%d'))]
        )
        self.connection.commit()


if __name__ == "__main__":
    client = DbClient()
    print(client.get_all())
