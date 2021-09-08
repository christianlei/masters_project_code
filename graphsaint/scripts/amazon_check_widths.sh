#!/bin/bash

python3 graphsaint_data_train_model.py 32 amazon > ../amazon/width_results/output_amazon_32.txt

python3 graphsaint_data_train_model.py 64 amazon > ../amazon/width_results/output_amazon_64.txt

# python3 graphsaint_data_train_model.py 4 amazon > ../amazon/width_results/output_amazon_4.txt

# python3 graphsaint_data_train_model.py 8 amazon > ../amazon/width_results/output_amazon_8.txt

# python3 graphsaint_data_train_model.py 1 amazon > ../amazon/width_results/output_amazon_1.txt

# python3 graphsaint_data_train_model.py 2 amazon > ../amazon/width_results/output_amazon_2.txt

# python3 graphsaint_data_train_model.py 128 amazon > ../amazon/width_results/output_amazon_128.txt

# python3 graphsaint_data_train_model.py 256 amazon > ../amazon/width_results/output_amazon_256.txt

#512 is too big

# python3 yelp_train_model.py 768 > output768.txt

# python3 yelp_train_model.py 1024 > output1024.txt

# python3 yelp_train_model.py 2048 > output2048.txt

# python3 yelp_train_model.py 4096 > output4096.txt