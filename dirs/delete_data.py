import os
import shutil
from dirs.dirs import *


def delete_data(model):
    for d in [MODEL_DOMAIN_DIR, MODEL_RUNS_DIR, INTERFACE_RUNS_DIR, EMBEDDING_RUNS_DIR]:
        path = os.path.join(d, model)
        try:
            shutil.rmtree(path)
            print('del: {}'.format(path))
        except FileNotFoundError:
            print('not found: {}'.format(path))
