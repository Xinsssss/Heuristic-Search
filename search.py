# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Grid
from searchProblems import nullHeuristic,PositionSearchProblem,ConstrainedAstarProblem

### You might need to use
from copy import deepcopy


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    util.raiseNotDefined()


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    myPQ = util.PriorityQueue()
    startState = problem.getStartState()
    startNode = (startState, 0, [])
    myPQ.push(startNode, heuristic(startState, problem))
    best_g = dict()
    while not myPQ.isEmpty():
        node = myPQ.pop()
        state, cost, path = node
        if (not state in best_g) or (cost < best_g[state]):
            best_g[state] = cost
            if problem.isGoalState(state):
                return path
            for succ in problem.getSuccessors(state):
                succState, succAction, succCost = succ
                new_cost = cost + succCost
                newNode = (succState, new_cost, path + [succAction])
                myPQ.push(newNode, heuristic(succState, problem) + new_cost)

    return None  # Goal not found



def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    util.raiseNotDefined()


def dijkstra(start, dimWidth, dimHeight,walls):
    """
        Generates the minimum distance from all points on the map to the food'
        Args:
            start (int,int): coordinate of the food
            dimWidth (int): width of the map
            dimHeight (int): height of the map
            walls (game.Grid): layout of walls on the map
        Returns:
            [[int]]: 2-d list of minimum distance from a point on the map to the food
    """

    # Initialise the map with (approximately) infinite value so the heuristic is safe
    distanceMap = [[9999]*dimHeight for _ in range (dimWidth)]
    # Set the food coordinate to distance 0 since it's the goal
    # In this case the heuristic would be goal aware
    distanceMap[start[0]][start[1]] = 0
    dijPQ = util.PriorityQueue()
    node = (0,start)
    dijPQ.push(node,0)
    while not dijPQ.isEmpty():
        node = dijPQ.pop()
        distance, (x, y) = node
        # For each action we move 1 unit in a straight line
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < dimWidth and 0 <= ny < dimHeight and not walls[nx][ny]:
                # Update the minimum distance from the food to all other points on the game board
                new_distance = distance + 1
                if new_distance < distanceMap[nx][ny]:
                    distanceMap[nx][ny] = new_distance
                    dijPQ.push((new_distance, (nx, ny)),new_distance)

    # Return the map containing minimum distance from all points to the food
    return distanceMap


def precompute_map(problem,foodList):
    """
        For each food on the map, we precompute a minimum spanning tree from the food and use as the heurstic
        Args:
            problem (searchProblems.FoodSearchProblem): current path-finding problem
            foodList [(int,int)]: list of coordinates of foods on the map
    """
    walls = problem.walls
    width, height = walls.width, walls.height
    # food: coordinate of one of the food
    # width: width of the game board
    # height: height of the game board
    # walls: distribution of walls on the game board
    distanceMap = {food:dijkstra(food,width,height,walls) for food in foodList}
    # We load the information into the problem object
    problem.heuristicInfo['precompute_map'] = distanceMap


def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    "*** YOUR CODE HERE for TASK1 ***"

    pacman, foodGrid = state
    foodList = foodGrid.asList()

    # The heurstic function uses a fixed minimum spanning tree for all food, therefore
    # we only compute it once on the first time we call foodHeuristic
    if 'precompute_map' not in problem.heuristicInfo:
        precompute_map(problem,foodList)
    
    distanceMap = problem.heuristicInfo['precompute_map']

    # Get the distance of this current pacman to all other food
    heuristics = [distanceMap[food][pacman[0]][pacman[1]] for food in foodList]

    if not heuristics: # If no food
        return 0
    elif len(heuristics)==1: # If there's only one food
        return heuristics[0]
    
    # the maximum distance between any two food items 
    # the remaining distance to visit all other food dots (reaching goal state) is at least this value
    else:
        maxFoodPair = max(
            distanceMap[food1][food2[0]][food2[1]]
            for i, food1 in enumerate(foodList)
            for food2 in foodList[i+1:]
            )
        return min(heuristics) + maxFoodPair
    # util.raiseNotDefined()

######################################################
# This part includes helper functions used in part 2 #
######################################################

def midConflict(path1, path2):
    """
        Check for vertex and swapping collision between two paths
        Return the first conflict, or None if no conflict occus between these two paths
        Args:
            path1 [(int,int)]: list of coordinates of the first path
            path2 [(int,int)]: list of coordinates of the second path
        Returns:
            (int, (int,int)): vertex conflict
            (string,(int,int),(int,int)): swapping conflict, string indicates type and time of swap, tuples are coordinates where
                                          swap occurs
            None: if no conflict detected
    """
    minTime = min(len(path1),len(path2))
    for t in range(minTime):
        if path1[t]==path2[t]: # vertext conflict
            return (t, path1[t])
        if t > 0 and path1[t-1] == path2[t] and path1[t] == path2[t-1]:
            if path1[t][1] == path2[t][1]: # same y value, horizontal swap
                return (str(t)+'h',path1[t], path2[t])
            elif path1[t][0] == path2[t][0]: # same x value, vertical swap
                return (str(t)+'v', path1[t], path2[t])

    return None

def goalConflict(goal,path,endTime):
    """
        Check for goal conflict, meaning that a pacman has stopped on its food, but it's in the way of other pacman
        Args:
            goal (int,int): coordinate of the goal
            path [(int,int)]: coordinates of the path of the travelling pacman
            endTime int: time where the stopped pacman reached its goal
        Returns:
            (int, (int,int)): if conflict detected, return the time of reaching the goal and the coordinate it stops
            None: if no conflict
    """
    for t in range(endTime,len(path)):
        if path[t]==goal:
            return (str(endTime)+'g', goal)
    
    return None

def checkConflictinPath(startPosition, allPaths):
    """
        Check pairwise conflicts in all paths and return conflict type, returns the first conflict detected
        Args:
            startPosition {String:(int,int)}: initial position of all pacman
            allPaths {String:[(int,int)]}: coordinates of current optimal path of each pacman
        Returns:
            (String,String,conflict): returns name of pacmen in the conflict and the conflict information
            None: if no conflict detected among all paths
    """
    pacmanList = list(allPaths.keys())
    pacmanNum = len(pacmanList)
    for i in range(pacmanNum):
        # Swap conflict might happen at the first action, so we need to consider the initial position
        path1 = [startPosition[pacmanList[i]]] + allPaths[pacmanList[i]]
        for j in range(i + 1, pacmanNum):
            path2 = [startPosition[pacmanList[j]]] + allPaths[pacmanList[j]]
            # If one path is longer than the other, then there's a risk of goal conflict
            if len(path1) > len(path2):
                goal = path2[-1]
                endTime = len(path2) - 1
                conflict = goalConflict(goal, path1, endTime)
                if conflict:
                    # Constraint on goal conflict only applies on the node that reaches the goal
                    return (pacmanList[j],None, conflict)
            if len(path1) < len(path2):
                goal = path1[-1]
                endTime = len(path1) - 1
                conflict = goalConflict(goal, path2, endTime)
                if conflict:
                    return (pacmanList[i],None, conflict)   
            # If no goal conflict is detected, we move on to see if there's conflict on the path
            conflict = midConflict(path1, path2)
            if conflict:
                return (pacmanList[i], pacmanList[j], conflict)

    return None

def tupleToDict(constraints):
    """
        Conflicts are initially stored as (time,(coordinate)), for the ease of locating conflict, we translate it
        to a dictoinary
        Args:
            constraints [(int,(int,int))]: list of conflicts and their timestamp
        Returns:
            {int:[(int,int)]}: dictionary with timestamp as the key, and conflicts occured during this time as values
    """
    constraintDict = dict()
    for key,value in constraints:
        if key not in constraintDict:
            constraintDict[key] = []
        constraintDict[key].append(value)
    
    return constraintDict

def aStarVariant(problem, constraints, heuristic=nullHeuristic):
    """
        This algorithm is a variant of the A-STAR algorithm where it considers constraints that should be avoided
        while finding a path
        Args:
            problem(searchProblems.ConstrainedAstarProblem): problem of one pacman targeting its own food on the map
            constraints [(int,(int,int))]: list of constraints applied on this path
        Returns:
            ([(int,int)],[String]): list of coordinates of the optimal path and actions taken to travel through this path
            (None, None): if fail to find a solution
    """
    constraints = tupleToDict(constraints)
    myPQ = util.PriorityQueue()
    startState = problem.getStartState()
    startNode = (startState,0,[],[]) # state, cost, path, coordinate of path
    myPQ.push(startNode,heuristic(startState,problem))
    best_g = dict()
    penalty = 50 # Penalise on the swap conflict

    while not myPQ.isEmpty():
        node  = myPQ.pop()
        state,cost,path,coords = node
        # Time of children = length of current path + 1
        time = len(coords) + 1
        if ((state,time-1) not in best_g) or (cost < best_g[(state,time-1)]):
            best_g[(state,time-1)] = cost
            # Return if found a valid path to the goal
            goalConf = str(time-1)+'g'
            if problem.isGoalState(state):
                if goalConf in constraints.keys():
                    if not state[0] in constraints[goalConf]:
                        return (coords,path)
                else:
                    return (coords,path)
            for succ in problem.getSuccessors(state):
                succState,succAction,succCost = succ
                # Excludes vertex or goal conflict from the child nodes
                if time in constraints.keys() and succState[0] in constraints[time]:
                    continue
                # Check for swap conflict at this timestamp
                horiSwap = str(time)+'h'
                vertSwap = str(time)+'v'
                goalConf = str(time)+'g'
                # Check if move result in goal conflict
                new_cost = cost + succCost
                # Check for horizontal swap
                if horiSwap in constraints.keys():
                    if succState[0] in constraints[horiSwap]: # If swap occurs at this position, we excldues it
                        continue
                    for swapPosition in constraints[horiSwap]:
                        # If it's a horizontaly shift to the other direction, we penalise this action
                        # to avoid swaping conflict in the next timestamp
                        if succState[0][1] == swapPosition[1]:
                            new_cost += penalty 
                # Check for vertical swap
                if vertSwap in constraints.keys():
                    if succState[0] in constraints[vertSwap]: # Excludes
                        continue
                    for swapPosition in constraints[vertSwap]: # Penalise
                        if succState[0][0] == swapPosition[0]:
                            new_cost += penalty
                if goalConf in constraints.keys():
                    if not problem.isGoalState(succState):
                        new_cost += penalty
                # Add the new child node to the queue
                newNode = (succState, new_cost, path + [succAction],coords + [succState[0]])
                myPQ.push(newNode, heuristic(succState, problem) + new_cost)
        
    return (None,None) # Goal not found

def getIndividualFood(foodGrid,pacmanName):
    """
        The initial map contains position of all food, we translate it to only the location of the food
        targeted by the current pacman
        Args:
            foodGrid (game.Grid): initial location of all foods
            String: name of the current pacman
        Return:
            game.Grid: a new grid for the current pacman
    """

    width = foodGrid.width
    height = foodGrid.height
    newFoodGrid = deepcopy(foodGrid)
    for i in range(width):
        for j in range(height):
            if newFoodGrid[i][j] != False and newFoodGrid[i][j] != pacmanName: # Remove all non-targeting food
                newFoodGrid[i][j] = False
            elif newFoodGrid[i][j] == pacmanName: # Change the food to True so we can use asList to generate coordinate
                newFoodGrid[i][j] = True

    return newFoodGrid

def manhattanHeuristic(states, problem):
    """
        Calcualte the manhattan distance of current position to the closest food
        Args:
            states (int,int): current position
            problem(searchProblems.ConstrainedAstarProblem): problem of one pacman targeting its own food on the map
        Return:
            int: minimum distance to a food
    """
    _, foodGrid = problem.getStartState()
    foodList = foodGrid.asList()
    x1,y1 = states[0]
    distances = []
    for food in foodList:
        x2,y2 = food
        distances.append(abs(x1-x2)+abs(y1-y2))
    if distances:
        return min(distances)
    return 0



def findAllPaths(problem, constraints):
    """
        For each iteration, find the optimal path of all pacmen to its food with constraints applied
        Args:
            problem(searchProblems.MAPFProblem): problem of multiple pacmen targeting to their food
            constraints({String:[int,(int,int)]}): record each pacman's constraints
        Return:
            {String:[(int,int)]}: coordinate of path travelled by each pacman
            {String:[String]}: action taken by each pacmen to travel through the path
    """
    pacmanStates, foodGrid = problem.getStartState()
    solution = {}
    states = {}
    pacmanNames = pacmanStates.keys()
    for pacman in pacmanNames:
        # We treat each pacman as an individual food search problem
        newFoodGrid = getIndividualFood(foodGrid,pacman)
        constrainedProblem = ConstrainedAstarProblem(pacmanStates[pacman],newFoodGrid,problem.walls)
        constraint = constraints[pacman]
        (allStates, currPath) = aStarVariant(constrainedProblem,constraint,manhattanHeuristic)
        solution[pacman] = currPath
        states[pacman] = allStates
    
    return (states,solution)

def calculateCost(paths):
    """
        Calculate the cost of this current path
        Args:
            path [String]: list of actions taken
        Return:
            int : total steps taken in the path
    """
    pacmen = paths.keys()
    cost = 0
    for pacman in pacmen:
        path = paths[pacman]
        cost += len(path)
    
    return cost

from searchProblems import MAPFProblem
def conflictBasedSearch(problem: MAPFProblem):
    """
        Conflict-based search algorithm.
        Input: MAPFProblem
        Output(IMPORTANT!!!): A dictionary stores path for each pacman as a list {pacman_name: [a1, a2, ...]}.

        A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
          pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
          foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
        
        Hints:
            You should model the constrained Astar problem as a food search problem instead of a position search problem,
            you can use: ConstrainedAstarProblem
            The stop action may also need to be considered as a valid action for finding an optimal solution
            Also you should model the constraints as vertex constraints and edge constraints

    """
    "*** YOUR CODE HERE for TASK2 ***"

    pacman_positions, _ = problem.getStartState()
    myPQ = util.PriorityQueue()

    pacmen = list(pacman_positions.keys())
    constraints = {pacman:[] for pacman in pacmen}
    (states,solution) = findAllPaths(problem,constraints)
    cost = calculateCost(solution)
    myPQ = util.PriorityQueue()
    myPQ.push((states,solution,constraints),cost)

    while not myPQ.isEmpty():
        (states,solution,constraints) = myPQ.pop()
        conflict = checkConflictinPath(pacman_positions,states)
        if not conflict:
            # Find the optimal path for all without conflict
            return solution
        # Since we only consider one conflict at a time, only two pacmen will be involved in this conflict
        pacman1, pacman2, newConflict = conflict
        newConstraint1 = deepcopy(constraints)
        newConstraint2 = deepcopy(constraints)
        # Goal conflict only applies on one pacman
        hasC2 = True
        if not pacman2: # goal conflict
            hasC2 = False
            newConstraint1[pacman1] = newConstraint1[pacman1] + [newConflict] 
        elif len(newConflict)==2: # vertex conflict
            newConstraint1[pacman1] = newConstraint1[pacman1] + [newConflict]
            newConstraint2[pacman2] = newConstraint2[pacman2] + [newConflict]
        elif len(newConflict)==3: # swap conflict
            newConstraint1[pacman1] = newConstraint1[pacman1] + [(newConflict[0],newConflict[1])]
            newConstraint2[pacman2] = newConstraint2[pacman2] + [(newConflict[0],newConflict[2])]

        (states,solution) = findAllPaths(problem,newConstraint1)
        cost = calculateCost(solution)
        myPQ.push((states,solution,newConstraint1),cost)
        if hasC2:
            (states,solution) = findAllPaths(problem,newConstraint2)
            cost = calculateCost(solution)
            myPQ.push((states,solution,newConstraint2),cost)

    return solution

    # util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
cbs = conflictBasedSearch

