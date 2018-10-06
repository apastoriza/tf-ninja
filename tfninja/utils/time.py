# coding=utf-8

import time
from datetime import datetime


def current_time_in_millis():
    return int(round(time.time() * 1000))


def current_time_in_microsecs():
    now = datetime.now()
    return now.microsecond
