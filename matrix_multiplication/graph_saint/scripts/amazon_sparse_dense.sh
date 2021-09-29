#!/bin/bash

python3 graphsaint_sparse_dense.py 2 amazon > ../amazon/raw/output_amazon_2.txt

python3 graphsaint_sparse_dense.py 4 amazon > ../amazon/raw/output_amazon_4.txt

python3 graphsaint_sparse_dense.py 8 amazon > ../amazon/raw/output_amazon_8.txt

python3 graphsaint_sparse_dense.py 16 amazon > ../amazon/raw/output_amazon_16.txt

python3 graphsaint_sparse_dense.py 32 amazon > ../amazon/raw/output_amazon_32.txt

python3 graphsaint_sparse_dense.py 64 amazon > ../amazon/raw/output_amazon_64.txt

python3 graphsaint_sparse_dense.py 128 amazon > ../amazon/raw/output_amazon_128.txt

python3 graphsaint_sparse_dense.py 256 amazon > ../amazon/raw/output_amazon_256.txt

python3 graphsaint_sparse_dense.py 32768 amazon > ../amazon/raw/output_amazon_32768.txt

python3 graphsaint_sparse_dense.py 65536 amazon > ../amazon/raw/output_amazon_65536.txt

python3 graphsaint_sparse_dense.py 131072 amazon > ../amazon/raw/output_amazon_131072.txt

python3 graphsaint_sparse_dense.py 262144 amazon > ../amazon/raw/output_amazon_262144.txt

python3 graphsaint_sparse_dense.py 524288 amazon > ../amazon/raw/output_amazon_524288.txt

python3 graphsaint_sparse_dense.py 1048576 amazon > ../amazon/raw/output_amazon_1048576.txt

python3 graphsaint_sparse_dense.py 1569959 amazon > ../amazon/raw/output_amazon_1569959.txt
