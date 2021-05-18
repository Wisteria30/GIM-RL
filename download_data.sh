#!/bin/bash -e
array=(
    "http://www.philippe-fournier-viger.com/spmf/datasets/chess.txt"
    "http://www.philippe-fournier-viger.com/spmf/datasets/mushrooms.txt"
    "http://www.philippe-fournier-viger.com/spmf/datasets/pumsb.txt"
    "http://www.philippe-fournier-viger.com/spmf/datasets/connect.txt"
    "http://www.philippe-fournier-viger.com/spmf/datasets/chess_utility_spmf.txt"
    "http://www.philippe-fournier-viger.com/spmf/datasets/mushroom_utility_SPMF.txt"
    "http://www.philippe-fournier-viger.com/spmf/datasets/accidents_utility_spmf.txt"
    "http://www.philippe-fournier-viger.com/spmf/datasets/connect_utility_spmf.txt"
)

for v in "${array[@]}"
do
  wget -P ./data "$v"
done