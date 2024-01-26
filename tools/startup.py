import os
import argparse

def startup(gpu=4,5,6,7):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    from train import main 
    main()

if __name__ = == '__main__':
    startup()
