#!/bin/bash

python3 reddit_train_model.py 1 > ../width_results/output_reddit_1.txt

python3 reddit_train_model.py 2 > ../width_results/output_reddit_2.txt

python3 reddit_train_model.py 4 > ../width_results/output_reddit_4.txt

python3 reddit_train_model.py 8 > ../width_results/output_reddit_8.txt

python3 reddit_train_model.py 16 > ../width_results/output_reddit_16.txt

python3 reddit_train_model.py 32 > ../width_results/output_reddit_32.txt

python3 reddit_train_model.py 64 > ../width_results/output_reddit_64.txt

python3 reddit_train_model.py 128 > ../width_results/output_reddit_128.txt

# python3 reddit_train_model.py 256 > ../width_results/output_reddit_256.txt

python3 reddit_train_model.py 512 > ../width_results/output_reddit_512.txt

python3 reddit_train_model.py 1024 > ../width_results/output_reddit_1024.txt

# python3 reddit_train_model.py 2048 > ../width_results/output_reddit_2048.txt

# python3 reddit_predictions.py 4096 > predictions/output4096.txt

# python3 reddit_predictions.py 8192 > predictions/output8192.txt

# python3 reddit_predictions.py 16384 > predictions/output16384.txt

# python3 reddit_predictions.py 32768 > predictions/output32768.txt

# python3 reddit_predictions.py 55333 > predictions/output55333.txt