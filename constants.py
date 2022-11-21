# variable categories
LpContinuous = "Continuous"
LpInteger = "Integer"
LpBinary = "Binary"
LpCategories = {LpContinuous: "Continuous", LpInteger: "Integer", LpBinary: "Binary"}

# objective sense
LpMinimize = 1
LpMaximize = -1
LpSenses = {LpMaximize: "Maximize", LpMinimize: "Minimize"}

# problem status
LpStatusNotSolved = 0
LpStatusOptimal = 1
LpStatusInfeasible = -1
LpStatusUnbounded = -2
LpStatusUndefined = -3
LpStatus = {
    LpStatusNotSolved: "Not Solved",
    LpStatusOptimal: "Optimal",
    LpStatusInfeasible: "Infeasible",
    LpStatusUnbounded: "Unbounded",
    LpStatusUndefined: "Undefined",
}

# solution status
LpSolutionNoSolutionFound = 0
LpSolutionOptimal = 1
LpSolutionIntegerFeasible = 2
LpSolutionInfeasible = -1
LpSolutionUnbounded = -2
LpSolution = {
    LpSolutionNoSolutionFound: "No Solution Found",
    LpSolutionOptimal: "Optimal Solution Found",
    LpSolutionIntegerFeasible: "Solution Found",
    LpSolutionInfeasible: "No Solution Exists",
    LpSolutionUnbounded: "Solution is Unbounded",
}
LpStatusToSolution = {
    LpStatusNotSolved: LpSolutionInfeasible,
    LpStatusOptimal: LpSolutionOptimal,
    LpStatusInfeasible: LpSolutionInfeasible,
    LpStatusUnbounded: LpSolutionUnbounded,
    LpStatusUndefined: LpSolutionInfeasible,
}

# constraint sense
LpConstraintLE = -1
LpConstraintEQ = 0
LpConstraintGE = 1
LpConstraintTypeToMps = {LpConstraintLE: "L", LpConstraintEQ: "E", LpConstraintGE: "G"}
LpConstraintSenses = {LpConstraintEQ: "=", LpConstraintLE: "<=", LpConstraintGE: ">="}
# LP line size
LpCplexLPLineSize = 78


def isiterable(obj):
    try:
        obj = iter(obj)
    except:
        return False
    else:
        return True


class PulpError(Exception):
    """
    Pulp Exception Class
    """

    pass