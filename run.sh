#!/bin/bash
 
for var in 1 2 3 4 5 6 7 8 9 10 11 12 21 22 23 24 25 26 27 28 29 30
do
  python train_transformer.py --person_index $var
done