from datetime import datetime


def fix_ts():
    """ ts format: YYYY-MM-DDThh:mm:ss """

    return datetime.now().isoformat().split('.')[0]


def log(s):
    """ Print a LOG message """

    print('[LOG ' + fix_ts() + '] ' + s)
