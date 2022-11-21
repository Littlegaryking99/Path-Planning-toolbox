import sys
print(sys.version)
import time
import numpy as np
import matplotlib
import quadprog
from quadprog import solve_qp
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from dynamics import Dyn
from agent import Agent
from obstacle import Sphere, Ellipsoid, Wall
from gurobipy import *
from goal import Goal
from sim_plots import Cbf_data, Clf_data, ColorManager
from params import *
from scipy.sparse import csr_matrix, lil_matrix
import constants as const
import math
import re
import warnings

class YVariable(object):
    def __init__(self, name, value = 0, lowBound=None, upBound=None):
        self.name = name
        self.lowBound = lowBound
        self.upBound = upBound
        self.value = value

    def getName(self):
        return self.name

    def getLowBound(self):
        return self.lowBound

    def getUpBound(self):
        return self.upBound
    
class YExpression(object):
    def __init__(self, e=None, constant=0, name=None):
        self.name = name
        self.e = e
        self.constant = constant
    
    def 