from core.train import train
from core.test import test
from core.general import train_general

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True, help='either "train" or "test"')
    parser.add_argument('-n', '--model', type=str, help='path to a trained model')
    parser.add_argument('-d', '--data', type=str, help='path to financial data from Yahoo Finance')
    args = parser.parse_args()

    if args.mode == 'test':
        assert args.model is not None
        assert args.data is not None
        test(args.data, args.model)

    elif args.mode == 'train':
        assert  args.data is not None
        train(args.data)

    elif args.mode == 'general':
        assert args.data is None
        train_general()
