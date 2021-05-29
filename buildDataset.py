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

def processGames(rankIdx, gameCount = 1800):
	ranks = ['5k', '4k', '3k', '2k', '1k', '1d', '2d', '3d', '4d']
	ranksCheck = ['5çº§', '4çº§', '3çº§', '2çº§', '1çº§', '1æ®µ', '2æ®µ', '3æ®µ', '4æ®µ']

	dataPath = "../cleanedGoData/" + ranks[rankIdx]
	trainPath = "../cleanedGoData/train/"
	valPath = "../cleanedGoData/val/"
	testPath = "../cleanedGoData/test/"
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
			with open(file, "rb") as f:
				game = sgf.Sgf_game.from_bytes(f.read())
			if not filterGame(game, rankCheck):
				continue
			try:
				features, labels = gameToFeatures(game)
				if i < 1500:
					for j in range(len(features)):
						torch.save(features[j], trainPath + "data/" + rank + "/" + str(idCounter) + ".pt")
						torch.save(labels[j], trainPath + "labels/" + rank + "/" + str(idCounter) + ".pt")
						np.savetxt(trainPath + "meta/" + rank + "/" + str(idCounter) + ".np", np.array([file]), fmt='%s')
						idCounter += 1
				elif i < 1650:
					for j in range(len(features)):
						torch.save(features[j], valPath + "data/" + rank + "/" + str(idCounter) + ".pt")
						torch.save(labels[j], valPath + "labels/" + rank + "/" + str(idCounter) + ".pt")
						np.savetxt(valPath + "meta/" + rank + "/" + str(idCounter) + ".np", np.array([file]), fmt='%s')
						idCounter += 1
				elif i < 1800: 
					for j in range(len(features)):
						torch.save(features[j], testPath + "data/" + rank + "/" + str(idCounter) + ".pt")
						torch.save(labels[j], testPath + "labels/" + rank + "/" + str(idCounter) + ".pt")
						np.savetxt(testPath + "meta/" + rank + "/" + str(idCounter) + ".np", np.array([file]), fmt='%s')
						idCounter += 1
			except Exception as e:
				continue 
			break
	return

def main():
	assert len(sys.argv) == 3

	print("using rankIdxs: ", [sys.argv[1], sys.argv[2]])# , sys.argv[3]])
	with Pool(2) as p:
		p.map(processGames, [int(sys.argv[1]), int(sys.argv[2])])#, int(sys.argv[3])])

if __name__ == '__main__':
    main()