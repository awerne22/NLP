import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from time import time
from concurrent.future import ThreadPoolExecutor
from scipy.optimize import curv_fit