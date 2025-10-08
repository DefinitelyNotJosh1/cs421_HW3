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
        return -utility(gameState)



##
# miniMax
# Description: Runs the minimax algorithm on a given node
# 
# Parameters:
#   gameState - a game state
#   depth - the depth of the search
#   alpha - the alpha value (not used yet, but I figured I'd at it now)
#   beta - the beta value (also not used yet, but I figured I'd at it now)
#   me - the id of me, the current player
def miniMax(gameState, depth, alpha, beta, me):
    if gameState.phase == PLAY_PHASE:
        winner = getWinner(gameState)
        if winner is not None or depth == 0: # This is my base case
            return rootEval(gameState, me), None

    moves = listAllLegalMoves(gameState)

    if not moves:
        return rootEval(gameState, me), None

    if gameState.whoseTurn == me:
        # Sort descending (best first)
        moves.sort(key=lambda move: rootEval(getNextStateAdversarial(gameState, move), me), reverse=True)
    else:
        # Sort ascending (worst first)
        moves.sort(key=lambda move: rootEval(getNextStateAdversarial(gameState, move), me))
    
    # Limit to top 10 moves for better performance
    moves = moves[:10]

    # If it's my turn, we want to maximize our score
    if gameState.whoseTurn == me:
        bestValue = float('-inf')
        bestMove = None
        for move in moves:
            newState = getNextStateAdversarial(gameState, move)
            value, _ = miniMax(newState, depth-1, alpha, beta, me)

            if value > bestValue:
                bestValue = value
                bestMove = move

                #DEBUG: delete
                if (move.moveType == END) and (len(moves) > 1):
                    print(f"DEBUG: Found END move: {move}")

            # update alpha
            alpha = max(alpha, bestValue)
            # prune
            if beta <= alpha:
                break

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

            # # update beta
            beta = min(beta, bestValue)
            # prune
            if beta <= alpha:
                break

        return bestValue, bestMove




##
# utility
#
# Description: Calculates the utility of a given game state on a scale of 0 to 1
# Reminder: Do not use the board variable
#
# Parameters:
#   gameState - a game state
#
# Return: The utility of the state
#
##
def utility(gameState):
    # Some ideas: from Josh:
    # Food difference - this should absolutely play a decently large role.
    # enemy ants - if the enemy has lots of ants and we don't, that's bad.

    # Constants
    me = gameState.whoseTurn
    enemy = 1 - me
    myInv = getCurrPlayerInventory(gameState)
    enemyInv = getEnemyInv(enemy, gameState)
    myAnts = getAntList(gameState, me, (WORKER,DRONE,SOLDIER,R_SOLDIER,QUEEN))
    enemyWorkers = getAntList(gameState, enemy, (WORKER,))
    enemyAttackers = getAntList(gameState, enemy, (DRONE,SOLDIER,R_SOLDIER))    
    utility = 0.0
    # If I win in this game state; instant win lose
    if gameState.phase == PLAY_PHASE:   #v libby trick
        if getWinner(gameState) == me or \
                len(getAntList(gameState, enemy, (QUEEN,))) == 0 or \
                myInv.foodCount == 11 or \
                enemyInv.getAnthill().captureHealth == 0:
            return float(100.0)  # cost 2 win?
        elif getWinner(gameState) == enemy or \
            len(myAnts) == 0 or \
                enemyInv.foodCount == 11 or \
                myInv.getAnthill().captureHealth == 0:
            return float(0) # float('inf') # cost 2 lose? / best thing ever

    # estimate moves for queen, food, capture hill,... maybe soldiers?


    # food stuff - 60% of total utility
    foodScore = foodUtility(gameState, myInv, enemyInv, me)
    # print(f"Food Score: {foodScore}")q
    # defense stuff - 30% of total utility
    defenseScore = defenseUtility(gameState, me)
    # attack stuff - 10% of total utility
    attackScore = attackUtility(gameState, myInv, enemyInv, me)


    utility = foodScore * (0.6) + defenseScore * (0.3) + attackScore * (0.1)

    return utility


## foodUtility
# Description: Calculates the utility of the food situation in a game state
# Includes worker utility
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
    # Food Weights - 90% of total utility
    if myInv.foodCount is not None and enemyInv.foodCount is not None:
        foodScore = 0.5
        foodScore += (myInv.foodCount / 11) * 0.5 # This is on a scale of 0 - 1 - good, now multiply by multiplier
        foodScore -= (enemyInv.foodCount / 11) * 0.5
        # print(f"Food Score: {foodScore}")
        utility += foodScore * 0.97


        # Some help from ChatGPT
        workerScore = 0.0
        # Get my workers
        myWorkers = getAntList(gameState, me, (WORKER,))
        tunnels = myInv.getTunnels()
        anthill = myInv.getAnthill()
        foodList = getConstrList(gameState, None, (FOOD,))
        numWorkers = len(myWorkers)
        myAnts = getAntList(gameState, me, (QUEEN, WORKER, DRONE, SOLDIER, R_SOLDIER))
        # print(f"Workers: {myWorkers}")
        # print(f"Num Workers: {numWorkers}")

        # If we have no workers, score is 0
        # if numWorkers == 0:
        #     return 0.0

        # If we have too many workers, aka not good
        if numWorkers > 2:
            utility -= 0.1

        # Avoid division by zero; if no workers, score remains 0
        if numWorkers > 0:
            # Precompute drop sites
            dropSites = []
            if anthill:
                dropSites.append(anthill.coords)
            if tunnels:
                dropSites.extend([t.coords for t in tunnels])

            # Normalization constants keep per-worker contribution in [0,1]
            maxFoodDist = 8.0
            maxDropDist = 8.0

            for i, w in enumerate(myWorkers):
                contrib = 0.0

                if w.carrying:
                    # If at drop site: full contribution
                    if dropSites and any(w.coords == d for d in dropSites):
                        contrib = 1.0
                    else:
                        # Positive baseline for carrying so picking up is attractive
                        if dropSites:
                            closestDrop = min(approxDist(w.coords, d) for d in dropSites)
                            progressToDrop = max(0.0, min(1.0, 1.0 - (closestDrop / maxDropDist)))
                        else:
                            progressToDrop = 0.0
                        # Baseline 0.5 plus progress up to 1.0 max
                        contrib = 0.5 + 0.5 * progressToDrop
                else:
                    # Not carrying: incentivize getting closer to nearest food, but cap at 0.5
                    if foodList:
                        closestFood = min(approxDist(w.coords, f.coords) for f in foodList)
                        towardFood = max(0.0, min(1.0, 1.0 - (closestFood / maxFoodDist)))
                        contrib = 0.5 * towardFood
                    else:
                        contrib = 0.0

                # Clamp and average across workers
                contrib = max(0.0, min(1.0, contrib))
                workerScore += contrib / numWorkers
                # print(f"Worker {i} contrib: {contrib}")

        # print(f"Worker Score: {workerScore}")
        # Ensure workerScore in [0,1]
        # print(f"Worker Score: {workerScore}")
        workerScore = max(0.0, min(1.0, workerScore))
        utility += (workerScore * 0.03)
        # utility = min(utility, 1.0)

        # give a bonus for every ant that has moved
        # for ant in myAnts:
        #     if ant.hasMoved:Fb
        #         utility += 0.03
        return utility


## defenseUtility
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
def defenseUtility(gameState, me):
    enemy = 1 - me
    def on_my_side(coords):
        y = coords[1]
        return (y <= 4)

    # Enemy ants on my side are threats
    threats = [a for a in getAntList(gameState, enemy, (QUEEN, WORKER, DRONE, SOLDIER, R_SOLDIER)) if on_my_side(a.coords)]
    # My attack-capable ants
    defenders = getAntList(gameState, me, (SOLDIER,))

    # If no threats on my side, defense is perfect
    if not threats:
        return 1.0
    # If there are threats but no defenders, defense is bad
    if not defenders:
        return 0.0
    
    # print(f"Threats: {threats}")
    # print(f"Defenders: {defenders}")

    # Encourage defenders to be close to threats
    # 0 distance -> 1.0 score; distance >= maxDist -> 0.0 score
    maxDist = 13.0
    total = 0.0
    for t in threats:
        minDist = min(approxDist(d.coords, t.coords) for d in defenders)
        # Use a sharper curve: closer defenders get much higher score
        if minDist >= maxDist:
            score = 0.0
        else:
            score = (maxDist - minDist) / maxDist
            # square to reward being closer
            score = score ** 2
        total += score


    proximityScore = total / len(threats)
    # print(f"Proximity Score: {proximityScore}")
    return proximityScore


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
    maxDist = 18.0
    total = 0.0
 
    # Only consider soldiers for attack utility
    myAnts = getAntList(gameState, me, (SOLDIER,))
    
    # Get enemy workers as primary targets
    enemyWorkers = getAntList(gameState, enemy, (WORKER,))
    
    # If no ants, return 0
    if not myAnts:
        return 0.0
    
    # If no enemy workers, good
    if not enemyWorkers:
        # print(f"No enemy workers")
        return 1.0
    
    # Calculate proximity scores for ants
    for ant in myAnts:
        if enemyWorkers:
            minDist = min(approxDist(ant.coords, worker.coords) for worker in enemyWorkers)
            score = max(0.0, min(1.0, 1.0 - (minDist / maxDist)))
            total += score
    
    # Normalize by number of ants
    proximityScore = total / len(myAnts)
    # print(f"Drone Score: {proximityScore}")

    queenHealth = enemyWorkers[0].health
    queenHealthScore = max(0.0, min(1.0, 1.0 - (queenHealth / 10.0)))

    # Sum health of enemy attacker ants
    enemyAnts = getAntList(gameState, enemy, (WORKER, DRONE, SOLDIER, R_SOLDIER))
    enemyAntHealth = sum(a.health for a in enemyAnts)

    enemyAntHealthScore = enemyAntHealth / 10.0


    return proximityScore# + queenHealthScore + enemyAntHealthScore

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
# def bestMove(nodes):
#     # Initialize the best node with the first node's utility
#     bestNodes = [nodes[0]]

#     # Iterate through nodes to find the one with the highest utility
#     for node in nodes:
#         if node.evaluation is None:
#             node.evaluation = utility(node.gameState) + node.depth
#         if (node.evaluation - node.depth < bestNodes[0].evaluation - bestNodes[0].depth):
#             bestNodes = [node]
#         elif (node.evaluation - node.depth == bestNodes[0].evaluation - bestNodes[0].depth):
#             bestNodes.append(node)

#     return random.choice(bestNodes)


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
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
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
        # run miniMax
        myUtility = round(utility(currentState), 2)
        # round to 2 decimal places
        print(f"Current Utility: {myUtility}")  
        value, move = miniMax(currentState, 3, float('-inf'), float('inf'), self.playerId)

        # DEBUG:
        # print the top 3 best moves at currentState
        moves = listAllLegalMoves(currentState)
        bagOfStates = []
        bagOfUtils = []
        for m in moves:
            newState = getNextStateAdversarial(currentState, m)
            u = utility(newState)
            bagOfStates.append(newState)
            bagOfUtils.append(u)
        # sort by utility
        sortedMoves = sorted(zip(moves, bagOfUtils), key=lambda x: x[1], reverse=True)
        print("Top 3 moves by utility:")
        for m, u in sortedMoves[:3]:
            print(f"Move: {m}, Utility: {round(u, 2)}")



        print(f"Move with value {value} selected: {move}")
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


# # Remove print statements for final version
# # print("-------------------------------- STARTING TESTS -------------------------------- ")
# totalTests = 4
# passedTests = 0
# # BEST MOVE TEST
# # print("| Beginning bestMove test")
# nodes = []
# for i in range(10):
#     node = Node(None, None, GameState.getBlankState(), 1, None)
#     nodes.append(node)
#     node.evaluation = i / 10 + node.depth
# bestNode = bestMove(nodes)

# if bestNode.evaluation == 1.0:
#     # print(f"| BestMove test passed. Value was {bestNode.evaluation}, expected 1.9")
#     passedTests += 1
# else:
#     print(f"| BestMove test failed. Value was {bestNode.evaluation}, expected 1.9")


# # UTILITY TEST
# # print("| Beginning utility test")
# gameState = GameState.getBlankState()
# util = utility(gameState)
# if not 0.0 <= util <= 1.0:
#     print(f"| ERROR: utility() returned {util}, expected 0.4")
# else:
#     # print(f"| Utility test passed. Value was {util}, expected 0.4")
#     passedTests += 1


# # FOOD UTILITY TEST
# # print("| Beginning food utility test")
# gameState = GameState.getBasicState()
# util = foodUtility(gameState, getCurrPlayerInventory(gameState), getEnemyInv(0, gameState), 0)
# if not 0.0 <= util <= 1.0:
#     print(f"| ERROR: foodUtility() returned {util}, expected 0.0")
# else:
#     # print(f"| Food utility test passed. Value was {util}, expected 0.0")
#     passedTests += 1


# # DEFENSE UTILITY TEST
# # print("| Beginning defense utility test")
# gameState = GameState.getBasicState()
# util = defenseUtility(gameState, 0)
# if not 0.0 <= util <= 1.0:
#     print(f"| ERROR: defenseUtility() returned {util}, expected 1.0")
# else:
#     # print(f"| Defense utility test passed. Value was {util}, expected 1.0")
#     passedTests += 1

# # print(f"|----------------------- Passed {passedTests} out of {totalTests} tests --------------------------- ")
# # print("-------------------------------- ENDING TESTS -------------------------------- ")
