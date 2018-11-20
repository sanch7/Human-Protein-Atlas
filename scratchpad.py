import os, sys
import numpy as np
import matplotlib.pyplot as plt

def main():
	imarray = np.random.rand((31072, 256, 256, 4))
	print(imarray.nbytes)

if __name__ == '__main__':
    main()