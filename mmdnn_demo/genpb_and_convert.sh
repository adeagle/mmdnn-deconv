#!/bin/bash
python2 network.py
mmconvert -sf tensorflow -iw model.pb --inNodeName input --inputShape 608,608,3 --dstNodeName a b c  -df caffe -om caffe_model
