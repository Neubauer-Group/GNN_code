#!/bin/#!/usr/bin/env bash

cd /raid/projects/atkinsn2/training
mkdir $1

ln -s /raid/projects/atkinsn2/trackml $1
cd $1
mv trackml raw
mkdir processed
cp /raid/projects/atkinsn2/gnn_code/config.yaml .
# vim config.yaml

cd /raid/projects/atkinsn2/gnn_code
