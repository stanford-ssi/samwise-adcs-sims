import pytest
import datetime
from simwise.utils.time import dt_utc_to_jd, jd_to_dt_utc


def test_jd_back_and_forth():
    for year in range(30):
        for day in range(365):
            dt = datetime.datetime(2000 + year, 1, 1) + datetime.timedelta(days=day)
            print(dt)
            jd = dt_utc_to_jd(dt)
            print(jd)
            dt2 = jd_to_dt_utc(jd)
            assert dt == dt2