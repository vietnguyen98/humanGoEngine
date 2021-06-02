import sys
import random
import os
import numpy as np
import torch
from datetime import datetime as dt
import glob
from collections import Counter
from sgfmill import sgf
from sgfmill import boards
from sgfmill import ascii_boards
from tqdm import tqdm as tq
from util import buildFeatures, checkSimpleKo, gameToFeatures, filterGame
from multiprocessing import Pool

def processGames(rankIdx, gameCount = 5400):
	ranks = ['1k']
	ranksCheck = ['1çº§']

	dataPath = "../cleanedGoData/" + ranks[rankIdx]
	trainPath = "../cleanedGoData/1ktrain/"
	valPath = "../cleanedGoData/1kval/"
	testPath = "../cleanedGoData/1ktest/"
	files = glob.glob(dataPath + "/*/*.sgf")

	rank = ranks[rankIdx]
	rankCheck = ranksCheck[rankIdx]
	print("Sampling", str(gameCount), "games from", len(files), rank, "games")
	np.random.shuffle(files)

	fileCounter = 0
	idCounter = 1
	for i in tq(range(gameCount)):
		while True:
			file = files[fileCounter]
			fileCounter += 1
			try:
				with open(file, "rb") as f:
					game = sgf.Sgf_game.from_bytes(f.read())
			except Exception as e:
				continue
			try:
				if not filterGame(game, rankCheck):
					continue
			except Exception as e:
				continue    
			try:
				features, labels = gameToFeatures(game)
				if i < 5000:
					for j in range(len(features)):
						torch.save(features[j], trainPath + "data/" + str(idCounter) + ".pt")
						torch.save(labels[j], trainPath + "labels/" + str(idCounter) + ".pt")
						np.savetxt(trainPath + "meta/" + str(idCounter) + ".np", np.array([file]), fmt='%s')
						idCounter += 1
				elif i < 5100:
					for j in range(len(features)):
						torch.save(features[j], valPath + "data/" + str(idCounter) + ".pt")
						torch.save(labels[j], valPath + "labels/" + str(idCounter) + ".pt")
						np.savetxt(valPath + "meta/" + str(idCounter) + ".np", np.array([file]), fmt='%s')
						idCounter += 1
				elif i < 5400: 
					for j in range(len(features)):
						torch.save(features[j], testPath + "data/" + str(idCounter) + ".pt")
						torch.save(labels[j], testPath + "labels/" + str(idCounter) + ".pt")
						np.savetxt(testPath + "meta/" + str(idCounter) + ".np", np.array([file]), fmt='%s')
						idCounter += 1
			except Exception as e:
				continue 
			break
	return

def main():
	processGames(0)

if __name__ == '__main__':
    main()