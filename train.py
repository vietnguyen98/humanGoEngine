import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sc
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from util import GoDataset, getCorrectCount, getPaths
from torch.utils.tensorboard import SummaryWriter
from layers import OutputLayer
from models import convNet

def main():
	# set hyperparameters:
	num_epochs = 5
	lr = .001
	max_grad_norm = 5.0
	eval_every = 50000
	steps_till_eval = eval_every
	bestAcc = -1
	assert len(sys.argv) == 2
	savePath = "logs/" + str(sys.argv[1])
	print("Tensorboard logs will be saved here: ", savePath)
	writer = SummaryWriter(log_dir = savePath)

	if torch.cuda.is_available():
		device = torch.device('cuda:0')
	else:
		device = torch.device('cpu')
	print("using device: ", device)

	modelSavePath = "models/" + str(sys.argv[1])
	print("model will be saved here: ", modelSavePath)


	# model = OutputLayer(55)
	model = convNet(128, batchNorm = True)
	model = model.to(device)
	model.train()
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr = lr)

	print("getting paths")
	trainDPaths, trainLPaths = getPaths("../cleanedGoData/train/")
	valDPaths, valLPaths = getPaths("../cleanedGoData/val/")   
	# trainDPaths = trainDPaths[:50000]
	# trainLPaths = trainLPaths[:50000]
	inds = np.random.choice(len(valDPaths), 10000, replace = False)
	valDPaths = (np.array(valDPaths)[inds]).tolist()
	valLPaths = (np.array(valLPaths)[inds]).tolist()
	for i in range(len(valDPaths)):
		assert valDPaths[i].rsplit('/', 1)[1] == valLPaths[i].rsplit('/', 1)[1]
    
	print("building dataset")
	training_data = GoDataset(trainDPaths, trainLPaths)
	val_data = GoDataset(valDPaths, valLPaths)

	print("building dataloader")
	train_loader = DataLoader(training_data, batch_size = 128, shuffle = True, num_workers = 4)
	val_loader = DataLoader(val_data, batch_size = 128, shuffle = True, num_workers = 4)

	step = 0 
	for t in range(num_epochs):
		print("Epoch ", t + 1, "\n-----------------------------")
		with torch.enable_grad(), \
				tqdm(total=len(train_loader.dataset)) as progress_bar:
			for batch, (X, y) in enumerate(train_loader):
				X = X.to(device)
				batch_size = X.shape[0]
				pred = model(X)
				y = y.to(device)
				loss = loss_fn(pred, y)
				optimizer.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
				optimizer.step()

				# recording
				step += batch_size
				progress_bar.update(batch_size)
				progress_bar.set_postfix(epoch=t + 1, NLL=loss.item())
				writer.add_scalar('train/NLL', loss, step)

				steps_till_eval -= batch_size
				if steps_till_eval <= 0:
					steps_till_eval = eval_every
					print("\nevaluating model...")
					loss_val, accuracy = evaluate(model, val_loader, device, loss_fn)
					writer.add_scalar('val/NLL', loss_val, step)
					writer.add_scalar('val/acc', accuracy, step)
					if accuracy >= bestAcc:
						bestAcc = accuracy
						torch.save(model, modelSavePath + str(step))
						print("new best acc of ", accuracy, "Saving model at: ", modelSavePath + "/"+ str(step) + ".pth")
	return

def evaluate(model, val_loader, device, loss_fn):
	model.eval()
	total=len(val_loader.dataset)
	lossTotal = 0
	accuracyTotal = 0
	currTotal = 0
	'''    
	with torch.no_grad():
		for batch, (X, y) in enumerate(val_loader):
			X = X.to(device)
			batch_size = X.shape[0]
			pred = model(X)
			y = y.to(device)
			loss = loss_fn(pred, y)

			currTotal += batch_size
			lossTotal += loss * batch_size
			accuracyTotal += getCorrectCount(pred, y)
	'''
	with torch.no_grad(), \
			tqdm(total=len(val_loader.dataset)) as progress_bar2:
		for batch, (X, y) in enumerate(val_loader):
			X = X.to(device)
			batch_size = X.shape[0]
			pred = model(X)
			y = y.to(device)
			loss = loss_fn(pred, y)

			currTotal += batch_size
			lossTotal += loss * batch_size
			accuracyTotal += getCorrectCount(pred, y)
			progress_bar2.update(batch_size)
			progress_bar2.set_postfix(NLL = lossTotal.item() / currTotal)
	lossTotal = lossTotal / total
	accuracyTotal = accuracyTotal / total
	return lossTotal, accuracyTotal

if __name__ == '__main__':
    main()
