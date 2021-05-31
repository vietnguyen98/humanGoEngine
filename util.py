import torch
from torch.utils.data import Dataset
import sys
import random
import os
import numpy as np
from datetime import datetime as dt
import glob
from collections import Counter
from sgfmill import sgf
from sgfmill import boards
from sgfmill import ascii_boards
from tqdm import tqdm as tq

class GoDataset(Dataset):
  # pos_paths: list of paths to position features
  # labels: list of moves played in a given position
  def __init__(self, pos_paths, label_paths):
        self.label_paths = label_paths
        self.pos_paths = pos_paths

  def __len__(self):
        return len(self.pos_paths)

  def __getitem__(self, idx):
        pos = torch.load(self.pos_paths[idx])
        correctClass= torch.load(self.label_paths[idx])
        label = torch.zeros(361)
        label[correctClass[0] * 19 + correctClass[1]] = 1
        return pos, label

def getPaths(prefix):
    fullDataPaths = []
    fullLabelPaths = []
    for rank in ranks:
        data = glob.glob(prefix + "data/" + rank + "/*.pt")
        labels = glob.glob(prefix + "labels/" + rank + "/*.pt")
        for i in range(len(data)):
            assert data[0].rsplit('/', 1)[1] == labels[0].rsplit('/', 1)[1]
            print(rank, len(data), len(labels))
            fullDataPaths.extend(data)
            fullLabelPaths.extend(labels)
    return fullDataPaths, fullLabelPaths

def getCorrectCount(pred, y):
    # pred: batch_size x 361
    # y: batch_size x 361
    return (torch.argmax(pred, dim = 1) == torch.argmax(pred, y, dim = 1)).sum()

def buildFeatures(positions):
    # input: 19 x 19 x 3 input: my stones, their stones, empty spots
    # output: 19 x 19 x 32: 
    # 0 - 7 liberties 
    # 8 - 15 self capture size
    # 16 - 23 capture size
    # 24 - 31 liberties after move
    # candidates: open 1 liberty slots for opponent stones
    n = positions.shape[0]

    
    visited = positions[:, :, 2] - 1
    output = torch.zeros((n, n, 32))
    candidates = []
    libertyMap = dict()
    stoneMap = dict()
    colorMap = dict()
    
    def getLiberties(x, y, color, placeholder):
        firstLiberty = None
        liberties = 0
        stoneCount = 1
        visited[x][y] = placeholder
        visited2[x][y] = 1
        if x > 0:
            if visited2[x - 1][y] == 0:
                visited2[x - 1][y] = 1
                if positions[x - 1][y][2] == 1:
                    if liberties == 0:
                        firstLiberty = [x - 1, y]
                    liberties += 1
                elif positions[x - 1][y][color] == 1:
                    l, s, fl = getLiberties(x - 1, y, color, placeholder)
                    if liberties == 0:
                        firstLiberty = fl
                    liberties += l
                    stoneCount += s
        if x < n - 1:
            if visited2[x + 1][y] == 0:
                visited2[x + 1][y] = 1
                if positions[x + 1][y][2] == 1:
                    if liberties == 0:
                        firstLiberty = [x + 1, y]
                    liberties += 1
                elif positions[x + 1][y][color] == 1:
                    l, s, fl = getLiberties(x + 1, y, color, placeholder)
                    if liberties == 0:
                        firstLiberty = fl
                    liberties += l
                    stoneCount += s
        if y > 0:
            if visited2[x][y - 1] == 0:
                visited2[x][y - 1] = 1
                if positions[x][y - 1][2] == 1:
                    if liberties == 0:
                        firstLiberty = [x, y - 1]
                    liberties += 1
                elif positions[x][y - 1][color] == 1:
                    l, s, fl = getLiberties(x, y - 1, color, placeholder)
                    if liberties == 0:
                        firstLiberty = fl
                    liberties += l
                    stoneCount += s
        if y < n - 1:
            if visited2[x][y + 1] == 0:
                visited2[x][y + 1] = 1
                if positions[x][y + 1][2] == 1:
                    if liberties == 0:
                        firstLiberty = [x, y + 1]
                    liberties += 1
                elif positions[x][y + 1][color] == 1:
                    l, s, fl = getLiberties(x, y + 1, color, placeholder)
                    if liberties == 0:
                        firstLiberty = fl
                    liberties += l
                    stoneCount += s
        
        return liberties, stoneCount, firstLiberty
    
    counter = 1
    for i in range(n):
        for j in range(n):
            # visited: -1 if stone is there, 0 if no stone there, counter if visited + stone present
            if visited[i][j] == -1: 
                # visited2: for each loop through connected stones, 1 if visited or liberty has been checked
                visited2 = torch.zeros_like(visited) 
                color = int(positions[i][j][1] == 1) # 0 if your stone, 1 if opponent stone
                l, s, fl = getLiberties(i, j, color, counter)
                libertyMap[counter] = l
                stoneMap[counter] = s
                colorMap[counter] = color # 0 = my color, 1 = their color
                if l == 1 and color == 1:
                    candidates.append((fl[0], fl[1])) # simpleKo candidate: may not be legal move for me to capture
                if l == 1:
                    output[fl[0]][fl[1]][min(s, 8) - 1 + 8 * (color + 1)] = 1 # mark self captures / captures
                counter += 1
    for i in range(n):
        for j in range(n):
            x = int(visited[i][j].item())
            if x == 0: # mark liberties after move
                if i > 0:
                    key = int(visited[i - 1][j].item())
                    if key != 0 and colorMap[key] == 1:
                        l = libertyMap[key]
                        output[i][j][min(l, 8) + 23] = 1
                if i < n - 1:
                    key = int(visited[i + 1][j].item())
                    if key != 0 and colorMap[key] == 1:
                        l = libertyMap[key]
                        output[i][j][min(l, 8) + 23] = 1
                if j > 0:
                    key = int(visited[i][j - 1].item())
                    if key != 0 and colorMap[key] == 1:
                        l = libertyMap[key]
                        output[i][j][min(l, 8) + 23] = 1
                if j < n - 1:
                    key = int(visited[i][j + 1].item())
                    if key != 0 and colorMap[key] == 1:
                        l = libertyMap[key]
                        output[i][j][min(l, 8) + 23] = 1
            else:
                l = libertyMap[x]
                output[i][j][min(l, 8) - 1] = 1 # mark liberty counts for each stone location
            
    return output, candidates 

# 55 features
# 0-2: stone positions
# 3-4: all 0's, all 1's
# 5-12: turn history
# 13 - 36: liberties, self captures, captures
# 37 - 44: liberties after move
# 45: simple ko constraint
# 46 - 54: one hot encoding for rank
def gameToFeatures(game):
    swapColor = {'w': 'b', 'b': 'w'}
    root_node = game.get_root()
    b = boards.Board(19)
    
    rankOneHot = None
    for rankInd, rank in enumerate(['5çº§', '4çº§', '3çº§', '2çº§', '1çº§', '1æ®µ', '2æ®µ', '3æ®µ', '4æ®µ']):        
        if root_node.get("BR") == rank and root_node.get("WR") == rank:
            assert rankOneHot == None
            rankOneHot = rankInd
    assert rankOneHot != None
    
    features = []
    labels = []
    counter = 0
    for node in game.get_main_sequence():
        color, move = node.get_move()
        #print(color, move)
        feature = torch.zeros(19, 19, 55)
        if color == None:
            feature[:, :, 2] = 1
            feature[:, :, 4] = 1
            feature[:, :, 46 + rankOneHot] = 1
        else:
            labels.append([move[0], move[1]])
            b.play(move[0], move[1], color)
            for c, p in b.list_occupied_points():
                if c != color:
                    # my color: c / their color: color (last move made)
                    feature[p[0], p[1], 0] = 1
                else:
                    feature[p[0], p[1], 1] = 1
            feature[:, :, 2] = (feature[:, :, 0] + feature[:, :, 1]) == 0
            feature[:, :, 4] = 1
            feature[move[0], move[1], 5] = 1
            # moves 1-7 history from last feature => 2-8 history of current feature
            feature[:, :, 6:13] = features[-1][:, :, 5:12] 
            feature[:, :, 13:45], candidates = buildFeatures(feature[:, :, :3])
            feature[:, :, 45] = checkSimpleKo(oldb, b, candidates, swapColor[color])
            feature[:, :, 46 + rankOneHot] = 1
        features.append(feature)
        counter += 1
        oldb = b.copy()
    return features[:-1], labels

def checkSimpleKo(past, present, candidates, color):
    n = 19
    output = torch.zeros((n, n))
    for x, y in candidates:
        variation = present.copy()
        try: 
            variation.play(x, y, color)
            if variation.list_occupied_points() == past.list_occupied_points():
                output[x][y] = 1
        except Exception:
            pass
    return output

def filterGame(game, rank):
    board_size = game.get_size()
    if board_size != 19:
        return False
    root_node = game.get_root()
    if root_node.get("BR") != rank:
        return False
    if root_node.get("WR") != rank:
        return False
    if root_node.get("RU") != "Japanese":
        return False
    if root_node.get("TM") != 600:
        return False
    if root_node.get("KM") != 0:
        return False
    if dt.strptime(root_node.get("DT"), '%Y-%m-%d').year != 2017:
        return False
    return True 