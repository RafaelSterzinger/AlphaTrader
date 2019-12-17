from core.train import train
from core.test import test
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True, help='either "train" or "test"')
    parser.add_argument('-n', '--model', type=str, help='path to a trained model')
    parser.add_argument('-d', '--data', type=str, help='path to financial data from Yahoo Finance', required=True)
    args = parser.parse_args()

    if args.mode == 'test':
        assert args.model is not None
        test(args.data, args.model)

    elif args.mode == 'train':
        train(args.data)
