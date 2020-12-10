# ==============================================================================
# IMPORTS
# ==============================================================================
# External imports
from docplex.mp.model import Model
from itertools import permutations
from functools import reduce
from operator import mul
from numpy import argmax
from time import time as getTime
import matplotlib.pyplot as plt
import random
import copy
import sys
import os

# Internal Imports
from util import generateRandomDefenders, generateRandomAttackers

# ==============================================================================
# GAME SETTINGS
# ==============================================================================
def writeGames(gameFile, gameCount, tNum, aNum, dNum):
    # Write the number of games in the file for parsing later
    f = open(gameFile, "w")
    f.write(f"{gameCount}\n")
    f.close()
    for gameNum in range(gameCount):
        # Generate the defender and attacker information
        defenderInformation = generateRandomDefenders(dNum,tNum)
        attackerInformation = generateRandomAttackers(aNum,tNum)
        # write the game to the file
        writeGame(gameFile,defenderInformation,attackerInformation)

def writeGame(gameFile, defenderInformation, attackerInformation, mode="a"):
    """
    The files are written with one line per defender, followed by one line per
    attacker
    """
    # Open the file
    f = open(gameFile, mode)
    # Get the information for each
    defenders, dRewards, dPenalties, dCosts = defenderInformation
    aTypes, aRewards, aPenalties, q = attackerInformation
    # Write the number of defenders and attackers
    infoLine = f"{len(defenders)};{len(aTypes)}\n"
    f.write(infoLine)
    # write a line for each defender
    for defender in defenders:
        rewardList = dRewards[defender]
        penaltyList = dPenalties[defender]
        costList = dCosts[defender]
        # write these all in one line, semi-colon separated
        defenderLine = f"{rewardList};{penaltyList};{costList}\n"
        f.write(defenderLine)
    # write a line for each attacker
    for attacker in aTypes:
        rewardList = aRewards[attacker]
        penaltyList = aPenalties[attacker]
        probability = q[attacker]
        attackerLine = f"{rewardList};{penaltyList};{probability}\n"
        f.write(attackerLine)
    # Close the file
    f.close()

def readGame(openFile):
    # Read in the number of defenders and attackers
    defenderNum, attackerNum = [int(x) for x in openFile.readline().split(";")]
    # Initialize data types
    dRewards = {}
    dPenalties = {}
    dCosts = {}
    aRewards = {}
    aPenalties = {}
    q = []
    # Read in defender info
    for defender in range(defenderNum):
        stringLists =  openFile.readline().split(";")
        stringLists = [stringList.replace("[","").replace("]","").split(", ") for stringList in stringLists]
        dReward, dPenalty, dCost = stringLists
        dReward = [int(x) for x in dReward]
        dPenalty = [int(x) for x in dPenalty]
        dCost = [int(x) for x in dCost]
        dRewards[defender] = dReward
        dPenalties[defender] = dPenalty
        dCosts[defender] = dCost
    # Read in attacker info
    for attacker in range(attackerNum):
        stringLists =  openFile.readline().split(";")
        stringLists = [stringList.replace("[","").replace("]","").split(", ") for stringList in stringLists]
        aReward, aPenalty, qVal = stringLists
        aReward = [int(x) for x in aReward]
        aPenalty = [int(x) for x in aPenalty]
        qVal = float(qVal[0].replace("\n",""))
        aRewards[attacker] = aReward
        aPenalties[attacker] = aPenalty
        q.append(qVal)
    return list(range(defenderNum)), dRewards, dPenalties, dCosts, list(range(attackerNum)), aRewards, aPenalties, q

def readGames(gameFile):
    games = []
    # Open the file for reading
    f = open(gameFile, "r")
    # Read the number of games
    gameNum = int(f.readline())
    # Read in the number of defenders and attackers
    for game in range(gameNum):
        games.append((readGame(f)))
    # Close the file
    f.close()
    return gameNum, games

def getFileName(tNum,aNum,dNum):
    directory = "games"
    fileName = f"{tNum}_{aNum}_{dNum}.txt"
    return os.path.join(directory,fileName)


def main():
    # Get the arguments
    args = sys.argv[1:]
    tNum, aNum, dNum, gameCount = args
    tNum, aNum, dNum, gameCount = [int(arg) for arg in [tNum,aNum,dNum,gameCount]]
    # Write the games
    filePath = getFileName(tNum,aNum,dNum)
    writeGames(filePath, gameCount, tNum, aNum, dNum)
    # Read the games
    #games = readGames(FILE_PATH)


# =========
# MAIN CALL
# =========
if __name__ == "__main__":
    main()
