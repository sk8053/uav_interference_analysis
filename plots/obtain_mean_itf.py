import numpy as np
import pandas as pd

import sys
import pathlib
ab_path = pathlib.Path().absolute().parent.__str__()

sys.path.append(ab_path+'/uav_interference_analysis/src/')

from src.network_channel import Channel_Info, Network
chan_info = Channel_Info()
