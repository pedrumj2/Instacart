#!/bin/bash


git pull origin master

rm -rf output
mkdir output

pass=fafdRE33
ip=127.0.0.1
db=insta
hidden_layers=200
rnn_length=5
ratio=0.01
test_ratio=0.3
train_count=10000
model=1


for train_count in 1000 3000 
do
  for hidden_layers in 400 700 
  do
    for drop_out in 0.5
    do
      python Main.py $pass $ip $db $hidden_layers $rnn_length $ratio $test_ratio $train_count $model > output/${train_count}_${hidden_layers}.txt &
    done
  done
done



wait


