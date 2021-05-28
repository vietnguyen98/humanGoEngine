import sys
import random
import os
import numpy as np
import torch
from tqdm.notebook import tqdm as tq
from datetime import datetime as dt
import glob
from collections import Counter
from sgfmill import sgf
from sgfmill import boards
from sgfmill import ascii_boards
from tqdm import tqdm


def main():
	print(sys.argv)
	for i in tqdm(range(10)):
		print(i)


if __name__ == '__main__':
    main()