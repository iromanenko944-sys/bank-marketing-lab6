import argparse
from project import download
parser = argparse.ArgumentParser()
parser.add_argument('--variant', default='V4', choices=['V4', '4'])
parser.parse_args()
print(download())
