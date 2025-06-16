from datetime import datetime, timedelta

TIME_FORMAT = "%Y-%m-%d %H:%M:%S:%f"

def get_date(data):
    beg = datetime.strptime(data[0]['TIME'], TIME_FORMAT)
    end = datetime.strptime(data[-1]['TIME'], TIME_FORMAT)

    return beg, end, (end - beg).total_seconds()


def date_formt(s, formt=TIME_FORMAT):
    try:
        return datetime.strptime(s, formt)
    except:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def sec2date(x):
    return timedelta(seconds=x)