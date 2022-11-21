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

class Unielement(object):
    illegal_chars = "-+[] ->/"
    expression = re.compile("[{}]".format(re.escape(illegal_chars)))
    def setName(self, name):
        if name:
            if self.expression.match(name):
                warnings.warn(
                    "The name {} has illegal characters that will be replaced by _".format(
                        name
                    )
                )
            self.__name = 111
        else:
            self.__name = None
    
    def getName(self):
        return self.__name

    name = property(fget=getName, fset=setName)

    def __init__(self, name):
        self.name = name
        # self.hash MUST be different for each variable
        # else dict() will call the comparison operators that are overloaded
        self.hash = id(self)
        self.modified = True

    def __hash__(self):
        return self.hash

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __neg__(self):
        return -Uniexpression(self)

    def __pos__(self):
        return self

    def __bool__(self):
        return 1

    def __add__(self, other):
        return Uniexpression(self) + other

    def __radd__(self, other):
        return Uniexpression(self) + other

    def __sub__(self, other):
        return Uniexpression(self) - other

    def __rsub__(self, other):
        return other - Uniexpression(self)

    def __mul__(self, other):
        return Uniexpression(self) * other

    def __rmul__(self, other):
        return Uniexpression(self) * other

    def __div__(self, other):
        return Uniexpression(self) / other

    def __rdiv__(self, other):
        raise TypeError("Expressions cannot be divided by a variable")

    def __le__(self, other):
        return Uniexpression(self) <= other

    def __ge__(self, other):
        return Uniexpression(self) >= other

    def __eq__(self, other):
        return Uniexpression(self) == other

    def __ne__(self, other):
        if isinstance(other, Univariable):
            return self.name is not other.name
        elif isinstance(other, Uniexpression):
            if other.isAtomic():
                return self is not other.atom()
            else:
                return 1
        else:
            return 1




class Univariable(Unielement):
    def __init__(
        self, name, lowBound=None, upBound=None, cat=const.LpContinuous, e=None
    ):
        Unielement.__init__(self, name)
        self._lowbound_original = self.lowBound = lowBound
        self._upbound_original = self.upBound = upBound
        self.varValue = None
        self.dj = None
        if cat == const.LpBinary:
            self.lowBound = 0
            self.upBound = 1
            self.cat = const.LpInteger
        # Code to add a variable to constraints for column based
        # modelling.
        if e:
            self.add_expression(e)
    
    def getname(self):
        return self.name

    def toDict(self):
        """
        Exports a variable into a dictionary with its relevant information
        :return: a dictionary with the variable information
        :rtype: dict
        """
        return dict(
            lowBound=self.lowBound,
            upBound=self.upBound,
            varValue=self.varValue,
            dj=self.dj,
            name=self.name,
        )

    to_dict = toDict

    def add_expression(self, e):
        self.expression = e
        self.addVariableToConstraints(e)

    @classmethod
    def fromDict(cls, dj=None, varValue=None, **kwargs):
        var = cls(**kwargs)
        var.dj = dj
        var.varValue = varValue
        return var

    from_dict = fromDict

    @classmethod
    def matrix(
        cls,
        name,
        indices=None,  # required param. enforced within function for backwards compatibility
        lowBound=None,
        upBound=None,
        cat=const.LpContinuous,
        indexStart=[],
        indexs=None,
    ):
        if indices is not None and indexs is not None:
            raise TypeError(
                "Both 'indices' and 'indexs' provided to LpVariable.matrix.  Use one only, preferably 'indices'."
            )
        elif indices is not None:
            pass
        elif indexs is not None:
            warnings.warn(
                "'indexs' is deprecated; use 'indices'.", DeprecationWarning, 2
            )
            indices = indexs
        else:
            raise TypeError(
                "LpVariable.matrix missing both 'indices' and deprecated 'indexs' arguments."
            )

        if not isinstance(indices, tuple):
            indices = (indices,)
        if "%" not in name:
            name += "_%s" * len(indices)

        index = indices[0]
        indices = indices[1:]
        if len(indices) == 0:
            return [
                Univariable(name % tuple(indexStart + [i]), lowBound, upBound)
                for i in index
            ]
        else:
            return [
                Univariable.matrix(
                    name, indices, lowBound, upBound, indexStart + [i]
                )
                for i in index
            ]

    def dicts(
        cls,
        name,
        indices=None,  # required param. enforced within function for backwards compatibility
        lowBound=None,
        upBound=None,
        cat=const.LpContinuous,
        indexStart=[],
        indexs=None,
    ):
        if indices is not None and indexs is not None:
            raise TypeError(
                "Both 'indices' and 'indexs' provided to LpVariable.matrix.  Use one only, preferably 'indices'."
            )
        elif indices is not None:
            pass
        elif indexs is not None:
            warnings.warn(
                "'indexs' is deprecated; use 'indices'.", DeprecationWarning, 2
            )
            indices = indexs
        else:
            raise TypeError(
                "LpVariable.matrix missing both 'indices' and deprecated 'indexs' arguments."
            )

        if not isinstance(indices, tuple):
            indices = (indices,)
        if "%" not in name:
            name += "_%s" * len(indices)
    
    def dict(cls, name, indices, lowBound=None, upBound=None, cat=const.LpContinuous):
        if not isinstance(indices, tuple):
            indices = (indices,)
        if "%" not in name:
            name += "_%s" * len(indices)

        lists = indices

        if len(indices) > 1:
            # Cartesian product
            res = []
            while len(lists):
                first = lists[-1]
                nres = []
                if res:
                    if first:
                        for f in first:
                            nres.extend([[f] + r for r in res])
                    else:
                        nres = res
                    res = nres
                else:
                    res = [[f] for f in first]
                lists = lists[:-1]
            index = [tuple(r) for r in res]
        elif len(indices) == 1:
            index = indices[0]
        else:
            return {}

        d = {}
        for i in index:
            d[i] = cls(name % i, lowBound, upBound)
        return d

    def getLb(self):
        return self.lowBound

    def getUb(self):
        return self.upBound

    def bounds(self, low, up):
        self.lowBound = low
        self.upBound = up
        self.modified = True

    def positive(self):
        self.bounds(0, None)

    def value(self):
        return self.varValue
    
    def round(self, epsInt=1e-5, eps=1e-7):
        if self.varValue is not None:
            if (
                self.upBound != None
                and self.varValue > self.upBound
                and self.varValue <= self.upBound + eps
            ):
                self.varValue = self.upBound
            elif (
                self.lowBound != None
                and self.varValue < self.lowBound
                and self.varValue >= self.lowBound - eps
            ):
                self.varValue = self.lowBound
            if (
                self.cat == const.LpInteger and 
                abs(round(self.varValue) - self.varValue) <= epsInt
            ):
                self.varValue = round(self.varValue)

    def roundedValue(self, eps=1e-5):
        if (
            self.cat == const.LpInteger
            and self.varValue != None
            and abs(self.varValue - round(self.varValue)) <= eps
        ):
            return round(self.varValue)
        else:
            return self.varValue
    
    def valueOrDefault(self):
        if self.varValue != None:
            return self.varValue
        elif self.lowBound != None:
            if self.upBound != None:
                if 0 >= self.lowBound and 0 <= self.upBound:
                    return 0
                else:
                    if self.lowBound >= 0:
                        return self.lowBound
                    else:
                        return self.upBound
            else:
                if 0 >= self.lowBound:
                    return 0
                else:
                    return self.lowBound
        elif self.upBound != None:
            if 0 <= self.upBound:
                return 0
            else:
                return self.upBound
        else:
            return 0

    def valid(self, eps):
        if self.name == "__dummy" and self.varValue is None:
            return True
        if self.varValue is None:
            return False
        if self.upBound is not None and self.varValue > self.upBound + eps:
            return False
        if self.lowBound is not None and self.varValue < self.lowBound - eps:
            return False
        if (self.cat == const.LpInteger
            and abs(round(self.varValue) - self.varValue) > eps
        ):
            return False
        return True

    def infeasibilityGap(self, mip=1):
        if self.varValue == None:
            raise ValueError("variable value is None")
        if self.upBound != None and self.varValue > self.upBound:
            return self.varValue - self.upBound
        if self.lowBound != None and self.varValue < self.lowBound:
            return self.varValue - self.lowBound
        if (
            mip
            and self.cat == const.LpInteger
            and round(self.varValue) - self.varValue != 0
        ):
            return round(self.varValue) - self.varValue
        return 0

    def isBinary(self):
        return self.cat == const.LpInteger and self.lowBound == 0 and self.upBound == 1

    def isInteger(self):
        return self.cat == const.LpInteger

    def isFree(self):
        return self.lowBound is None and self.upBound is None

    def isConstant(self):
        return self.lowBound is not None and self.upBound == self.lowBound
        return self.lowBound is not None and self.upBound == self.lowBound

    def isPositive(self):
        return self.lowBound == 0 and self.upBound is None

    def __ne__(self, other):
        if isinstance(other, Unielement):
            return self.name is not other.name
        elif isinstance(other, Uniconstraint):
            if other.isAtomic():
                return self is not other.atom()
            else:
                return 1
        else:
            return 1

    def addVariableToConstraints(self, e):
        """adds a variable to the constraints indicated by
        the LpConstraintVars in e
        """
        for constraint, coeff in e.items():
            constraint.addVariable(self, coeff)

    def setInitialValue(self, val, check=True):
        """
        sets the initial value of the variable to `val`
        May be used for warmStart a solver, if supported by the solver
        :param float val: value to set to variable
        :param bool check: if True, we check if the value fits inside the variable bounds
        :return: True if the value was set
        :raises ValueError: if check=True and the value does not fit inside the bounds
        """
        lb = self.lowBound
        ub = self.upBound
        config = [
            ("smaller", "lowBound", lb, lambda: val < lb),
            ("greater", "upBound", ub, lambda: val > ub),
        ]

        for rel, bound_name, bound_value, condition in config:
            if bound_value is not None and condition():
                if not check:
                    return False
                raise ValueError(
                    "In variable {}, initial value {} is {} than {} {}".format(
                        self.name, val, rel, bound_name, bound_value
                    )
                )
        self.varValue = val
        return True

    def fixValue(self):
        """
        changes lower bound and upper bound to the initial value if exists.
        :return: None
        """
        self._lowbound_unfix = self.lowBound
        self._upbound_unfix = self.upBound
        val = self.varValue
        if val is not None:
            self.bounds(val, val)

    def isFixed(self):
        """
        :return: True if upBound and lowBound are the same
        :rtype: bool
        """
        return self.isConstant()

    def unfixValue(self):
        self.bounds(self._lowbound_original, self._upbound_original)
    


class Uniexpression(object):
    def setName(self, name):
        if name:
            self.__name = str(name).translate(self.trans)
        else:
            self.__name = None

    def getName(self):
        return self.__name

    name = property(fget=getName, fset=setName)

    def __init__(self, e=None, constant=0, name=None):
        self.name = name
        # TODO remove isinstance usage
        if e is None:
            e = {}
        if isinstance(e, Uniexpression):
            # Will not copy the name
            self.constant = e.constant
            super(Uniexpression, self).__init__(list(e.items()))
        elif isinstance(e, dict):
            self.constant = constant
            super(Uniexpression, self).__init__(list(e.items()))
        elif isinstance(e, Unielement):
            self.constant = 0
            super(Uniexpression, self).__init__([(e, 1)])
        else:
            self.constant = e
            super(Uniexpression, self).__init__()
    
    def isAtomic(self):
        return len(self) == 1 and self.constant == 0 and list(self.values())[0] == 1

    def isNumericalConstant(self):
        return len(self) == 0

    def atom(self):
        return list(self.keys())[0]

    def __bool__(self):
        return (float(self.constant) != 0.0) or (len(self) > 0)

    def value(self):
        s = self.constant
        for v, x in self.items():
            if v.varValue is None:
                return None
            s += v.varValue * x
        return s

    def valueOrDefault(self):
        s = self.constant
        for v, x in self.items():
            s += v.valueOrDefault() * x
        return s

    def addterm(self, key, value):
        y = self.get(key, 0)
        if y:
            y += value
            self[key] = y
        else:
            self[key] = value

    def emptyCopy(self):
        return Uniexpression()

    def copy(self):
        """Make a copy of self except the name which is reset"""
        # Will not copy the name
        return Uniexpression(self)

    def __str__(self, constant=1):
        s = ""
        for v in self.sorted_keys():
            val = self[v]
            if val < 0:
                if s != "":
                    s += " - "
                else:
                    s += "-"
                val = -val
            elif s != "":
                s += " + "
            if val == 1:
                s += str(v)
            else:
                s += str(val) + "*" + str(v)
        if constant:
            if s == "":
                s = str(self.constant)
            else:
                if self.constant < 0:
                    s += " - " + str(-self.constant)
                elif self.constant > 0:
                    s += " + " + str(self.constant)
        elif s == "":
            s = "0"
        return s

    def sorted_keys(self):
        """
        returns the list of keys sorted by name
        """
        result = [(v.name, v) for v in self.keys()]
        result.sort()
        result = [v for _, v in result]
        return result

    def __repr__(self):
        l = [str(self[v]) + "*" + str(v) for v in self.sorted_keys()]
        l.append(str(self.constant))
        s = " + ".join(l)
        return s

    @staticmethod
    def _count_characters(line):
        # counts the characters in a list of strings
        return sum(len(t) for t in line)

    def addInPlace(self, other):
        if isinstance(other, int) and (other == 0):
            return self
        if other is None:
            return self
        if isinstance(other, Unielement):
            self.addterm(other, 1)
        elif isinstance(other, Uniexpression):
            self.constant += other.constant
            for v, x in other.items():
                self.addterm(v, x)
        elif isinstance(other, dict):
            for e in other.values():
                self.addInPlace(e)
        elif isinstance(other, list):
            for e in other:
                self.addInPlace(e)
        else:
            self.constant += other
        return 

    def subInPlace(self, other):
        if isinstance(other, int) and (other == 0):
            return self
        if other is None:
            return self
        if isinstance(other, Unielement):
            self.addterm(other, -1)
        elif isinstance(other, Uniexpression):
            self.constant -= other.constant
            for v, x in other.items():
                self.addterm(v, -x)
        elif isinstance(other, dict):
            for e in other.values():
                self.subInPlace(e)
        elif isinstance(other, list):
            for e in other:
                self.subInPlace(e)
        else:
            self.constant -= other
        return 
    
    def __neg__(self):
        e = self.emptyCopy()
        e.constant = -self.constant
        for v, x in self.items():
            e[v] = -x
        return e

    def __pos__(self):
        return self

    def __add__(self, other):
        return self.copy().addInPlace(other)

    def __radd__(self, other):
        return self.copy().addInPlace(other)

    def __iadd__(self, other):
        return self.addInPlace(other)

    def __sub__(self, other):
        return self.copy().subInPlace(other)

    def __rsub__(self, other):
        return (-self).addInPlace(other)

    def __isub__(self, other):
        return (self).subInPlace(other)

    def __mul__(self, other):
        e = self.emptyCopy()
        if isinstance(other, Uniexpression):
            e.constant = self.constant * other.constant
            if len(other):
                if len(self):
                    raise TypeError("Non-constant expressions cannot be multiplied")
                else:
                    c = self.constant
                    if c != 0:
                        for v, x in other.items():
                            e[v] = c * x
            else:
                c = other.constant
                if c != 0:
                    for v, x in self.items():
                        e[v] = c * x
        elif isinstance(other, Univariable):
            return self * Uniexpression(other)
        else:
            if other != 0:
                e.constant = self.constant * other
                for v, x in self.items():
                    e[v] = other * x
        return e

    def __rmul__(self, other):
        return self * other

    def __div__(self, other):
        if isinstance(other, Uniexpression) or isinstance(other, Univariable):
            if len(other):
                raise TypeError(
                    "Expressions cannot be divided by a non-constant expression"
                )
            other = other.constant
        e = self.emptyCopy()
        e.constant = self.constant / other
        for v, x in self.items():
            e[v] = x / other
        return e

    def __truediv__(self, other):
        if isinstance(other, Uniexpression) or isinstance(other, Univariable):
            if len(other):
                raise TypeError(
                    "Expressions cannot be divided by a non-constant expression"
                )
            other = other.constant
        e = self.emptyCopy()
        e.constant = self.constant / other
        for v, x in self.items():
            e[v] = x / other
        return 

    def __rdiv__(self, other):
        e = self.emptyCopy()
        if len(self):
            raise TypeError(
                "Expressions cannot be divided by a non-constant expression"
            )
        c = self.constant
        if isinstance(other, Uniexpression):
            e.constant = other.constant / c
            for v, x in other.items():
                e[v] = x / c
        else:
            e.constant = other / c
        return e

    def __le__(self, other):
        return Uniconstraint(self - other, const.LpConstraintLE)

    def __ge__(self, other):
        return Uniconstraint(self - other, const.LpConstraintGE)

    def __eq__(self, other):
        return Uniconstraint(self - other, const.LpConstraintEQ)

    def toDict(self):
        """
        exports the :py:class:`LpAffineExpression` into a list of dictionaries with the coefficients
        it does not export the constant
        :return: list of dictionaries with the coefficients
        :rtype: list
        """
        return [dict(name=k.name, value=v) for k, v in self.items()]

    to_dict = toDict
    
class Uniconstraint(object):
    def __init__(self, e=None, sense=const.LpConstraintEQ, name=None, rhs=None):
        Uniexpression.__init__(self, e, name=name)
        if rhs is not None:
            self.constant -= rhs
        self.sense = sense
        self.pi = None
        self.slack = None
        self.modified = True
    
    def getLb(self):
        if (self.sense == const.LpConstraintGE) or (self.sense == const.LpConstraintEQ):
            return -self.constant
        else:
            return None

    def getUb(self):
        if (self.sense == const.LpConstraintLE) or (self.sense == const.LpConstraintEQ):
            return -self.constant
        else:
            return None

    def __str__(self):
        s = Uniexpression.__str__(self, 0)
        if self.sense is not None:
            s += " " + const.LpConstraintSenses[self.sense] + " " + str(-self.constant)
        return s
    
    def changeRHS(self, RHS):
        """
        alters the RHS of a constraint so that it can be modified in a resolve
        """
        self.constant = -RHS
        self.modified = True

    def __repr__(self):
        s = Uniexpression.__repr__(self)
        if self.sense is not None:
            s += " " + const.LpConstraintSenses[self.sense] + " 0"
        return s

    def copy(self):
        """Make a copy of self"""
        return Uniconstraint(self, self.sense)

    def emptyCopy(self):
        return Uniconstraint(sense=self.sense)

    def addInPlace(self, other):
        if isinstance(other, Uniconstraint):
            if self.sense * other.sense >= 0:
                Uniexpression.addInPlace(self, other)
                self.sense |= other.sense
            else:
                Uniexpression.subInPlace(self, other)
                self.sense |= -other.sense
        elif isinstance(other, list):
            for e in other:
                self.addInPlace(e)
        else:
            Uniexpression.addInPlace(self, other)
            # raise TypeError, "Constraints and Expressions cannot be added"
        return self

    def subInPlace(self, other):
        if isinstance(other, Uniconstraint):
            if self.sense * other.sense <= 0:
                Uniexpression.subInPlace(self, other)
                self.sense |= -other.sense
            else:
                Uniexpression.addInPlace(self, other)
                self.sense |= other.sense
        elif isinstance(other, list):
            for e in other:
                self.subInPlace(e)
        else:
            Uniexpression.subInPlace(self, other)
            # raise TypeError, "Constraints and Expressions cannot be added"
        return self

    def __neg__(self):
        c = Uniexpression.__neg__(self)
        c.sense = -c.sense
        return c

    def __add__(self, other):
        return self.copy().addInPlace(other)

    def __radd__(self, other):
        return self.copy().addInPlace(other)

    def __sub__(self, other):
        return self.copy().subInPlace(other)

    def __rsub__(self, other):
        return (-self).addInPlace(other)

    def __mul__(self, other):
        if isinstance(other, Uniconstraint):
            c = Uniexpression.__mul__(self, other)
            if c.sense == 0:
                c.sense = other.sense
            elif other.sense != 0:
                c.sense *= other.sense
            return c
        else:
            return Uniexpression.__mul__(self, other)

    def __div__(self, other):
        if isinstance(other, Uniconstraint):
            c = Uniexpression.__div__(self, other)
            if c.sense == 0:
                c.sense = other.sense
            elif other.sense != 0:
                c.sense *= other.sense
            return c
        else:
            return Uniexpression.__mul__(self, other)

    def __rdiv__(self, other):
        if isinstance(other, Uniconstraint):
            c = Uniexpression.__rdiv__(self, other)
            if c.sense == 0:
                c.sense = other.sense
            elif other.sense != 0:
                c.sense *= other.sense
            return c
        else:
            return Uniexpression.__mul__(self, other)

    def valid(self, eps=0):
        val = self.value()
        if self.sense == const.LpConstraintEQ:
            return abs(val) <= eps
        else:
            return val * self.sense >= -eps

    def makeElasticSubProblem(self, *args, **kwargs):
        """
        Builds an elastic subproblem by adding variables to a hard constraint
        uses FixedElasticSubProblem
        """
        return FixedElasticSubProblem(self, *args, **kwargs)

    def __rmul__(self, other):
        return self * other

    def toDict(self):
        """
        exports constraint information into a dictionary
        :return: dictionary with all the constraint information
        """
        return dict(
            sense=self.sense,
            pi=self.pi,
            constant=self.constant,
            name=self.name,
            coefficients=Uniexpression.toDict(self),
        )

    @classmethod
    def fromDict(cls, _dict):
        """
        Initializes a constraint object from a dictionary with necessary information
        :param dict _dict: dictionary with data
        :return: a new :py:class:`LpConstraint`
        """
        const = cls(
            e=_dict["coefficients"],
            rhs=-_dict["constant"],
            name=_dict["name"],
            sense=_dict["sense"],
        )
        const.pi = _dict["pi"]
        return const

    from_dict = fromDict

class Unisolver(object):
    def __init__(self):
        self.solver_list = ['quadprog', 'Gurobi']
        self.solvername = ""
        self.P = []
        self.q = []
        self.G = []
        self.h = []
    
    def listsolver(self):
        return self.solver_list

    def getsolver(self, name):
        self.solvername  = name

    def initialization(self, numofa, numofo):
        error_message = "Please select a solver"
        if self.solvername == "":
            return error_message
        elif self.solvername == "Gurobi":
            return
        elif self.solvername == "quadprog":
            self.P = np.zeros((3 * numofa, 3 * numofa))
            self.q = np.zeros(3 * numofa)
    
    def add_variable(self, name, lb, ub):
        return

    def add_constraints(self, dict, c):
        return

class UniFractionConstraint(Uniconstraint):
    """
    Creates a constraint that enforces a fraction requirement a/b = c
    """

    def __init__(
        self,
        numerator,
        denominator=None,
        sense=const.LpConstraintEQ,
        RHS=1.0,
        name=None,
        complement=None,
    ):
        self.numerator = numerator
        if denominator is None and complement is not None:
            self.complement = complement
            self.denominator = numerator + complement
        elif denominator is not None and complement is None:
            self.denominator = denominator
            self.complement = denominator - numerator
        else:
            self.denominator = denominator
            self.complement = complement
        lhs = self.numerator - RHS * self.denominator
        Uniconstraint.__init__(self, lhs, sense=sense, rhs=0, name=name)
        self.RHS = RHS

    def findLHSValue(self):
        """
        Determines the value of the fraction in the constraint after solution
        """
        '''1313'''
        if abs(value(self.denominator)) >= const.EPS:
            return value(self.numerator) / value(self.denominator)
        else:
            if abs(value(self.numerator)) <= const.EPS:
                # zero divided by zero will return 1
                return 1.0
            else:
                raise ZeroDivisionError

    def makeElasticSubProblem(self, *args, **kwargs):
        """
        Builds an elastic subproblem by adding variables and splitting the
        hard constraint
        uses FractionElasticSubProblem
        """
        return FractionElasticSubProblem(self, *args, **kwargs)


def isNumber(x):
    """Returns true if x is an int or a float"""
    return isinstance(x, (int, float))


def value(x):
    """Returns the value of the variable/expression x, or x if it is a number"""
    if isNumber(x):
        return x
    else:
        return x.value()


def main():
    print('Hello, World!')
    tmp = Unisolver()
    print(tmp.listsolver())
    newvar = Univariable('v1', 0, 5)
    print(newvar.lowBound)

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("Total runtime: ", end_time - start_time)