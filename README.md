# Pytorch-Large-Scale-Curiosity

This is a implementation of Large-Scale-Cruiosity using Pytorch, [here](https://github.com/openai/large-scale-curiosity) is the origin implementation and [here](https://openreview.net/form?id=rJNwDjAqYX) is the paper. I don't use MPI or multiprossing because my laptab can't run with them. I tried but failed.  

I test it on Atari and it works, and I have not run it on Roboschool or Unity3D.Maze. I really need Unity3D.Maze environment but can't find it in the origin implementation.

Another thing, I don't implemente pixel2pixel feature, don't use it.

## Requirement

Python 3.6+  
Pythorch 0.4  

## Usage

Download this repo and run run.py  

    python3 run.py --feat_learning none     # for random feature
    python3 run.py --feat_learning idf      # for IDF
    python3 run.py --feat_learning vaesph   # for VAE
