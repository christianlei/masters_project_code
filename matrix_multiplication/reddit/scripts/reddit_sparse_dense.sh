#!/bin/bash

python3 reddit_sparse_dense.py 2 reddit > ../reddit/raw/output_reddit_2.txt

python3 reddit_sparse_dense.py 4 reddit > ../reddit/raw/output_reddit_4.txt

python3 reddit_sparse_dense.py 8 reddit > ../reddit/raw/output_reddit_8.txt

python3 reddit_sparse_dense.py 16 reddit > ../reddit/raw/output_reddit_16.txt

python3 reddit_sparse_dense.py 32 reddit > ../reddit/raw/output_reddit_32.txt

python3 reddit_sparse_dense.py 64 reddit > ../reddit/raw/output_reddit_64.txt

python3 reddit_sparse_dense.py 128 reddit > ../reddit/raw/output_reddit_128.txt

python3 reddit_sparse_dense.py 256 reddit > ../reddit/raw/output_reddit_256.txt

python3 reddit_sparse_dense.py 32768 reddit > ../reddit/raw/output_reddit_32768.txt

# python3 reddit_sparse_dense.py 65536 reddit > ../reddit/raw/output_reddit_65536.txt

# python3 reddit_sparse_dense.py 131072 reddit > ../reddit/raw/output_reddit_131072.txt

# python3 reddit_sparse_dense.py 262144 reddit > ../reddit/raw/output_reddit_262144.txt

# python3 reddit_sparse_dense.py 524288 reddit > ../reddit/raw/output_reddit_524288.txt

# python3 reddit_sparse_dense.py 1048576 reddit > ../reddit/raw/output_reddit_1048576.txt

# python3 reddit_sparse_dense.py 1569959 reddit > ../reddit/raw/output_reddit_1569959.txt
