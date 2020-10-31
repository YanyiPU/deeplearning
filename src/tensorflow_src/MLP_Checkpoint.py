import numpy as np
import tensorflow as tf
import argparse
from zh.model.mnist.mlp import MLP
from zh.model.utils import MNISTLoader


parser = argparse.ArgumentParser(description = "Process some integers.")
parser.add_argument()











if __name__ == "__main__":
    if args.mode == "train":
        train()
    if args.mode == "test":
        test()