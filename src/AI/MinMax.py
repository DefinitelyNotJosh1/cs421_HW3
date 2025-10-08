# MinMax AI, HW 3 CS-421
# Authors:
# - Makengo Lokombo
# - Joshua Krasnogorov

import random
import sys

sys.path.append("..")  #so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *


##
# NODE
# Description: A node in the search tree; contains a game state, a move, the parent state,
# and the utility of the state.
##
class Node:
    # Use slots for memory optimization and fast attribute access -
    # HOWEVER  - we can't add new attributes dynamically now. This shouldn't be a problem tho
    __slots__ = ['parent', 'move', 'gameState', 'depth', 'evaluation']

    ## __init__
    #
    # Description: Creates a new node
    #
    # Parameters:
    #   parent - the parent node
    #   move - the move that led to this state
    #   gameState - the game state
    #   depth - how many steps to reach from the agent's actual state
    #   evalution - state depth + utility
    ##
    def __init__(self, parent, move, gameState, depth, evaluation):
        self.parent = parent
        self.move = move
        self.gameState = gameState
        self.depth = depth
        self.evaluation = evaluation


##
# rootEval
# Description: Evaluates the root node, base case for miniMax
#
# Parameters:
#   gameState - a game state
#   me - the id of me, the current player
#
# Return: The utility of the state
#
def rootEval(gameState, me):
    if gameState.whoseTurn == me:
        return utility(gameState)
    else:
        return 1.0 - utility(gameState)  # Changed to use consistent 0-1 scale


##
# miniMax with Alpha-Beta Pruning
# Description: Runs the minimax algorithm with alpha-beta pruning on a given node
# 
# Parameters:
#   gameState - a game state
#   depth - the depth of the search
#   alpha - the best value the maximizer can guarantee (initially -inf)
#   beta - the best value the minimizer can guarantee (initially +inf)
#   me - the id of me, the current player
#
# Return: (bestValue, bestMove) tuple
##
def miniMax(gameState, depth, alpha, beta, me):
    if gameState.phase == PLAY_PHASE:
        winner = getWinner(gameState)
        if winner is not None or depth == 0: # This is my base case
            return rootEval(gameState, me), None

    moves = listAllLegalMoves(gameState)

    if not moves:
        return rootEval(gameState, me), None
    
    # Always include END move as an option if not already in moves
    hasEndMove = any(move.moveType == END for move in moves)
    if not hasEndMove:
        moves.append(Move(END, None, None))
    
    # If it's my turn, we want to maximize our score
    if gameState.whoseTurn == me:
        bestValue = float('-inf')
        bestMove = None
        for move in moves:
            newState = getNextStateAdversarial(gameState, move)
            value, _ = miniMax(newState, depth - 1, alpha, beta, me)
            if value > bestValue:
                bestValue = value
                bestMove = move
            
            # Alpha-beta pruning for maximizing player
            alpha = max(alpha, bestValue)
            if beta <= alpha:
                break  # Beta cutoff - prune remaining branches
                
        return bestValue, bestMove
    # If it's the enemy's turn, we want to minimize our score
    else:
        bestValue = float('inf')
        bestMove = None
        for move in moves:
            newState = getNextStateAdversarial(gameState, move)
            value, _ = miniMax(newState, depth - 1, alpha, beta, me)
            if value < bestValue:
                bestValue = value
                bestMove = move
            
            # Alpha-beta pruning for minimizing player
            beta = min(beta, bestValue)
            if beta <= alpha:
                break  # Alpha cutoff - prune remaining branches
                
        return bestValue, bestMove


##
# utility
#
# Description: Calculates the utility of a given game state on a scale of 0 to 1
# HEAVILY WEIGHTED TOWARD FOOD COLLECTION
#
# Parameters:
#   gameState - a game state
#
# Return: The utility of the state
#
##
def utility(gameState):
    # Constants
    me = gameState.whoseTurn
    enemy = 1 - me
    myInv = getCurrPlayerInventory(gameState)
    enemyInv = getEnemyInv(enemy, gameState)
    myAnts = getAntList(gameState, me, (WORKER,DRONE,SOLDIER,R_SOLDIER,QUEEN))
    utility = 0.0
    
    # If I win in this game state; instant win lose
    if gameState.phase == PLAY_PHASE:
        if getWinner(gameState) == me or \
                len(getAntList(gameState, enemy, (QUEEN,))) == 0 or \
                myInv.foodCount == 11 or \
                enemyInv.getAnthill().captureHealth == 0:
            return float(1.0)  # Changed from 100 to 1.0 for consistency
        elif getWinner(gameState) == enemy or \
            len(myAnts) == 0 or \
                enemyInv.foodCount == 11 or \
                myInv.getAnthill().captureHealth == 0:
            return float(0.0)

    # FOOD is now 95% of total utility (increased from 80%)
    foodScore = foodUtility(gameState, myInv, enemyInv, me)
    if foodScore:
        utility += foodScore * 0.95

    # Defense is now only 5% (decreased from 20%)
    defenseScore = defenseUtility(gameState, me)
    utility += defenseScore * 0.05

    return utility


## foodUtility - ENHANCED FOR MORE AGGRESSIVE GATHERING + TUNNEL BLOCKING
# Description: Calculates the utility of the food situation in a game state
# Includes worker utility and tunnel blocking strategy
#
# Parameters:
#   gameState - a game state
#   myInv - the inventory of the current player
#   enemyInv - the inventory of the enemy
#   me - the id of the current player
#
# Return: The utility of the food situation
##
def foodUtility(gameState, myInv, enemyInv, me):
    utility = 0.0
    
    # Get my workers
    myWorkers = getAntList(gameState, me, (WORKER,))
    tunnels = myInv.getTunnels()
    anthill = myInv.getAnthill()
    foodList = getConstrList(gameState, None, (FOOD,))
    numWorkers = len(myWorkers)
    
    # Get enemy tunnel
    enemy = 1 - me
    enemyTunnels = getConstrList(gameState, enemy, (TUNNEL,))
    enemyTunnel = enemyTunnels[0].coords if enemyTunnels else None
    
    # Food count is CRITICAL - 60% of food utility (increased from 50%)
    if myInv.foodCount is not None and enemyInv.foodCount is not None:
        foodScore = 0.0
        # My food count is heavily rewarded
        foodScore += (myInv.foodCount / 11) * 0.8  # Increased from 0.5
        # Enemy food count is penalized
        foodScore -= (enemyInv.foodCount / 11) * 0.2  # Decreased from 0.5
        utility += foodScore * 0.6

    # If we have no workers, this is catastrophic
    if numWorkers == 0:
        return utility * 0.1  # Severe penalty

    # Encourage having 2 workers: 1 blocker + 1 gatherer (PRIORITY: blocking first!)
    if numWorkers == 1:
        utility += 0.05  # Small bonus for first worker (blocker)
    elif numWorkers == 2:
        utility += 0.20  # BIG bonus for two workers (blocker + gatherer)
    elif numWorkers > 2:
        utility -= 0.25  # Penalty for too many workers

    # Worker behavior scoring - 40% of food utility
    # DYNAMIC PRIORITY: Blocking until tunnel blocked, then FOOD becomes priority
    workerScore = 0.0
    blockerScore = 0.0
    gathererScore = 0.0
    tunnelIsBlocked = False
    
    # Check if tunnel is already blocked
    if enemyTunnel and numWorkers > 0:
        for w in myWorkers:
            if w.coords == enemyTunnel:
                tunnelIsBlocked = True
                break
    
    # Precompute drop sites
    dropSites = []
    if anthill:
        dropSites.append(anthill.coords)
    if tunnels:
        dropSites.extend([t.coords for t in tunnels])

    # Filter food to only include food on our side (y <= 3)
    myFoodList = [f for f in foodList if f.coords[1] <= 3]
    
    # Assign roles: first worker blocks (PRIORITY), second worker gathers
    for idx, w in enumerate(myWorkers):
        contrib = 0.0
        
        # First worker (index 0) should block enemy tunnel - TOP PRIORITY!
        if idx == 0 and enemyTunnel:
            # Check if worker is blocking the tunnel
            if w.coords == enemyTunnel:
                # Perfect! Worker is blocking the tunnel
                contrib = 1.0
            else:
                # Worker should move toward enemy tunnel
                distToTunnel = approxDist(w.coords, enemyTunnel)
                if distToTunnel == 1:
                    contrib = 0.95
                elif distToTunnel == 2:
                    contrib = 0.9
                elif distToTunnel <= 3:
                    contrib = 0.85
                elif distToTunnel <= 5:
                    contrib = 0.75
                else:
                    contrib = max(0.6, 1.0 - (distToTunnel / 20.0))
            blockerScore = contrib
        
        # Second worker (index 1) should gather food
        else:
            if w.carrying:
                # Worker is carrying food - MAXIMUM PRIORITY for depositing
                if dropSites:
                    if any(w.coords == d for d in dropSites):
                        # At drop site - MAXIMUM reward!
                        contrib = 1.0
                    else:
                        # Moving toward drop site - heavy reward based on proximity
                        closestDrop = min(approxDist(w.coords, d) for d in dropSites)
                        if closestDrop == 0:
                            contrib = 1.0
                        elif closestDrop == 1:
                            contrib = 0.95
                        elif closestDrop == 2:
                            contrib = 0.9
                        elif closestDrop <= 4:
                            contrib = 0.85
                        elif closestDrop <= 6:
                            contrib = 0.75
                        else:
                            contrib = max(0.5, 1.0 - (closestDrop / 15.0))
                else:
                    contrib = 0.6
            else:
                # Worker not carrying - HIGH priority for getting food
                if myFoodList:
                    closestFood = min(approxDist(w.coords, f.coords) for f in myFoodList)
                    
                    # Check if worker is ON food
                    onFood = any(w.coords == f.coords for f in myFoodList)
                    if onFood:
                        # On food - about to pick up
                        contrib = 0.95
                    elif closestFood == 1:
                        contrib = 0.9
                    elif closestFood == 2:
                        contrib = 0.85
                    elif closestFood <= 3:
                        contrib = 0.75
                    elif closestFood <= 5:
                        contrib = 0.6
                    elif closestFood <= 7:
                        contrib = 0.45
                    else:
                        contrib = max(0.2, 0.7 - (closestFood / 15.0))
                else:
                    # No food available
                    contrib = 0.0
            
            gathererScore = contrib
        
        workerScore += contrib
    
    # Weight blocking vs gathering based on whether tunnel is blocked
    if numWorkers == 1:
        # Only blocker exists - use blocker score
        finalWorkerScore = blockerScore
    elif numWorkers >= 2:
        if tunnelIsBlocked:
            # Tunnel is blocked! FOOD is now the priority
            # 80% gatherer, 20% blocker (maintain position)
            finalWorkerScore = (blockerScore * 0.2) + (gathererScore * 0.8)
        else:
            # Tunnel NOT blocked yet - blocking is priority
            # 70% blocker, 30% gatherer
            finalWorkerScore = (blockerScore * 0.7) + (gathererScore * 0.3)
    else:
        finalWorkerScore = 0.0
    
    # Add worker behavior to utility (40% weight)
    utility += finalWorkerScore * 0.4
    
    # Clamp final utility to valid range
    utility = max(0.0, min(1.0, utility))
    return utility


## defenseUtility
# Description: Calculates the utility of the defense situation in a game state
#
# Parameters:
#   gameState - a game state
#   me - the id of the current player
#
# Return: The utility of the attack situation
##
def defenseUtility(gameState, me):
    enemy = 1 - me
    def on_my_side(coords):
        y = coords[1]
        return (y <= 4)

    # Enemy ants on my side are threats
    threats = [a for a in getAntList(gameState, enemy, (QUEEN, WORKER, DRONE, SOLDIER, R_SOLDIER)) if on_my_side(a.coords)]
    # My attack-capable ants
    defenders = getAntList(gameState, me, (DRONE, SOLDIER, R_SOLDIER))

    # If no threats on my side, defense is perfect
    if not threats:
        return 1.0
    # If there are threats but no defenders, defense is bad
    if not defenders:
        return 0.0

    # Encourage defenders to be close to threats
    # 0 distance -> 1.0 score; distance >= maxDist -> 0.1 score
    maxDist = 10.0
    total = 1.0
    for t in threats:
        minDist = min(approxDist(d.coords, t.coords) for d in defenders)
        score = 2.0 - min(minDist / maxDist, 10.0)
        total += score

    proximityScore = total / len(threats)
    return max(0.0, min(1.0, proximityScore))


## attackUtility
# Description: Calculates the utility of the defense situation in a game state
#
# Parameters:
#   gameState - a game state
#   myInv - the inventory of the current player
#   enemyInv - the inventory of the enemy
#   me - the id of the current player
#
# Return: The utility of the attack situation
##
def attackUtility(gameState, myInv, enemyInv, me):
    enemy = 1 - me
    maxDist = 10.0
    total = 0.0
    def on_enemy_side(coords):
        y = coords[1]
        return y > 4

    # Enemy ants on the enemy side
    attackable = [a for a in getAntList(gameState, enemy, (QUEEN, WORKER, DRONE, SOLDIER, R_SOLDIER)) if on_enemy_side(a.coords)]
    # My attack-capable ants
    attackers = getAntList(gameState, me, (DRONE, SOLDIER, R_SOLDIER))

    enemyAnthill = enemyInv.getAnthill()
    if enemyAnthill:
        enemyAnthill = enemyAnthill.coords
    else:
        return 1.0

    enemyQueen = getAntList(gameState, enemy, (QUEEN,))[0]
    if enemyQueen:
        enemyQueen = enemyQueen.coords
    else:
        return 1.0

    # If there are no enemy's, attack is perfect
    if not attackable:
        for t in attackers:
            minDist = approxDist(enemyQueen, t.coords)
            score = 1.0 - min(minDist / maxDist, 10.0)
            total += score

        proximityScore = total / len(attackers) if total != 0 or len(attackers) != 0 else 0.0
        return max(0.0, min(1.0, proximityScore))
    # If there are threats but no attackers, attackable is bad
    if not attackers:
        return 0.0

    # Encourage attackers to be close to threats
    # 0 distance -> 1.0 score; distance >= maxDist -> 0.1 score
    for t in attackable:
        minDist = approxDist(enemyAnthill, t.coords)
        score = 1.0 - min(minDist / maxDist, 10.0)
        total += score

    proximityScore = total / len(attackable) if total != 0 or len(attackable) != 0 else 0.0
    return max(0.0, min(1.0, proximityScore))


##
# bestMove
#
# Description: Searches a given list of game nodes to find the highest utility move
#
# Parameters:
#   gameState - a game state
#   moves - a list of moves
#
# Return: The state with the highest utility
#
def bestMove(nodes):
    # Initialize the best node with the first node's utility
    bestNodes = [nodes[0]]

    # Iterate through nodes to find the one with the highest utility
    for node in nodes:
        if node.evaluation is None:
            node.evaluation = utility(node.gameState) + node.depth
        if (node.evaluation - node.depth < bestNodes[0].evaluation - bestNodes[0].depth):
            bestNodes = [node]
        elif (node.evaluation - node.depth == bestNodes[0].evaluation - bestNodes[0].depth):
            bestNodes.append(node)

    return random.choice(bestNodes)


##
#AIPlayer
#Description: The responsibility of this class is to interact with the game by
#deciding a valid move based on a given game state. This class has methods that
#will be implemented by students in Dr. Nuxoll's AI course.
#
#Variables:
#   playerId - The id of the player.
##
class AIPlayer(Player):

    #__init__
    #Description: Creates a new Player
    #
    #Parameters:
    #   inputPlayerId - The id to give the new player (int)
    #   cpy           - whether the player is a copy (when playing itself)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer,self).__init__(inputPlayerId, "MiniMax Bot")
        self.playerId = inputPlayerId


    ##
    #getPlacement
    #
    #Description: called during setup phase for each Construction that
    #   must be placed by the player.  These items are: 1 Anthill on
    #   the player's side; 1 tunnel on player's side; 9 grass on the
    #   player's side; and 2 food on the enemy's side.
    #
    #Parameters:
    #   construction - the Construction to be placed.
    #   currentState - the state of the game at this point in time.
    #
    #Return: The coordinates of where the construction is to be placed
    ##
    def getPlacement(self, currentState):
        numToPlace = 0
        #implemented by students to return their next move
        if currentState.phase == SETUP_PHASE_1:    #stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:   #stuff on foe's side
            enemyTunnel = getConstrList(currentState, None, (TUNNEL,))[0]
            enemyHill = getConstrList(currentState, None, (ANTHILL,))[0]

            # find all spots on enemy side of board that are empty
            furthestCoords = []
            for i in range(0, 10):
                for j in range(6, 10):
                    if currentState.board[i][j].constr == None:
                        furthestCoords.append((i,j))

            # sort spots by distance from enemy tunnel
            furthestCoords.sort(key=lambda x:
                        abs(enemyTunnel.coords[0] - x[0]) + abs(enemyTunnel.coords[1] - x[1]) +
                        abs(enemyHill.coords[0] - x[0]) + abs(enemyHill.coords[1] - x[1]))
            moves = []
            # add the two furthest spots to the moves list
            moves.append(furthestCoords[-1])
            moves.append(furthestCoords[-2])
            return moves
        else:
            return [(0, 0)]

    ##
    #getMove
    #Description: Gets the next move from the Player.
    #
    #Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    #Return: The Move to be made
    ##
    def getMove(self, currentState):
        # Run miniMax with alpha-beta pruning
        # Initialize alpha to -infinity and beta to +infinity
        value, move = miniMax(currentState, 3, float('-inf'), float('inf'), self.playerId)
        
        # Ensure we have a valid move
        if move is None:
            move = Move(END, None, None)
        
        return move


    ##
    #getAttack
    #Description: Gets the attack to be made from the Player
    #
    #Parameters:
    #   currentState - A clone of the current state (GameState)
    #   attackingAnt - The ant currently making the attack (Ant)
    #   enemyLocations - The Locations of the Enemies that can be attacked (Location[])
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        #Attack a random enemy.
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]

    ##
    #registerWin
    #
    # This agent doens't learn
    #
    def registerWin(self, hasWon):
        #method templaste, not implemented
        pass


# Unit Tests
totalTests = 4
passedTests = 0

# BEST MOVE TEST
nodes = []
for i in range(10):
    node = Node(None, None, GameState.getBlankState(), 1, None)
    nodes.append(node)
    node.evaluation = i / 10 + node.depth
bestNode = bestMove(nodes)

if bestNode.evaluation == 1.0:
    passedTests += 1
else:
    print(f"| BestMove test failed. Value was {bestNode.evaluation}, expected 1.9")

# UTILITY TEST
gameState = GameState.getBlankState()
util = utility(gameState)
if not 0.0 <= util <= 1.0:
    print(f"| ERROR: utility() returned {util}, expected 0.4")
else:
    passedTests += 1

# FOOD UTILITY TEST
gameState = GameState.getBasicState()
util = foodUtility(gameState, getCurrPlayerInventory(gameState), getEnemyInv(0, gameState), 0)
if not 0.0 <= util <= 1.0:
    print(f"| ERROR: foodUtility() returned {util}, expected 0.0")
else:
    passedTests += 1

# DEFENSE UTILITY TEST
gameState = GameState.getBasicState()
util = defenseUtility(gameState, 0)
if not 0.0 <= util <= 1.0:
    print(f"| ERROR: defenseUtility() returned {util}, expected 1.0")
else:
    passedTests += 1