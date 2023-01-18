import nltk
import argparse

parser = argparse.ArgumentParser(description='CRF')
parser.add_argument('--data', default=None, type=str, help='path to download dataset.')

args = parser.parse_args()

nltk.download('treebank', download_dir=args.data)
