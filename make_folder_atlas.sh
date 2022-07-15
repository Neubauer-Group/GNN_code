#!/bin/#!/usr/bin/env bash

cd training_data
mkdir $1
ln -s /data/gnn_code/atlas_data/charline_uncorrelated/ $1
cd $1
mv charline_uncorrelated raw
mkdir processed

# cp ../atlas_reflected/config.yaml .
cp ../test_atlas/config.yaml .


cd /data/gnn_code
