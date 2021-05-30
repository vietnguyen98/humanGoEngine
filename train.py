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
from util import GoDataset, getCorrectCount
from torch.utils.tensorboard import SummaryWriter

def main():
	# set hyperparameters:
	num_epochs = 10
	lr = 
	steps_till_eval = 50000
	assert len(sys.argv) = 2
	savePath = "logs/" + str(sys.arv[1])
	print("Tensorboard logs will be saved here: ", savePath)
	writer = SummaryWriter(log_dir = savePath)

	if torch.cuda.is_available():
		device = torch.device('cuda:0')
	else:
		device = torch.device('cpu')
	print("using device: ", device)

	model = 
	model = model.to(device)
	model.train()
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr = lr)


	train_loader = DataLoader(training_data, batch_size = 128, shuffle = True)
	val_loader = DataLoader(val_datam, batch_size = 128, shuffle = True)

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
            	optimizer.step()

            	# recording
				step += batch_size
				progress_bar.update(batch_size)
				progress_bar.set_postfix(epoch=epoch, NLL=loss_val)
                writer.add_scalar('train/NLL', loss_val, step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = 50000
                    loss, accuracy = evaluate(model, val_loader, device)
                    writer.add_scalar('val/NLL', loss)
                    writer.add_scalar('val/acc', accuracy)
	return

def evaluate(model, val_loader, device):
	model.eval()
	total=len(val_loader.dataset)
	lossTotal = 0
	accuracyTotal = 0
	currTotal = 0
    with torch.no_grad(), \
            tqdm(total=len(val_loader.dataset)) as progress_bar:
        for batch, (X, y) in enumerate(val_loader):
        	X = X.to(device)
        	batch_size = X.shape[0]
        	pred = model(X)
        	y = y.to(device)
        	loss = loss_fn(pred, y)

        	currTotal += batch_size
        	lossTotal += loss * batch_size
        	accuracyTotal += getCorrectCount(pred, y)
			progress_bar.update(batch_size)
			progress_bar.set_postfix(NLL=lossTotal / currTotal)
    lossTotal = lossTotal / total
    accuracyTotal = accuracyTotal / total
	return lossTotal, accuracyTotal

if __name__ == '__main__':
    main()
