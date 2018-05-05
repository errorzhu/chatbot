# coding:utf-8
from db import DbClient

import train
import time

if __name__ == "__main__":
    db = DbClient()
    text = db.get_all()
    with open('samples/answer', 'a') as f:
        for i, t in enumerate(text):
            if i % 2 == 0:
                t = t.strip()
                f.write(t)
                f.write('\n')
    with open('samples/question', 'a') as f:
        for i, t in enumerate(text):
            if i % 2 != 0:
                t = t.strip()
                f.write(t)
                f.write('\n')

    time.sleep(60 * 5)
    train.train()
