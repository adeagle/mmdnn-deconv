#!/bin/bash

cd ../MMdnn/
python2 setup.py install
cd -
./genpb_and_convert.sh
python2 forward_tf_caffe.py
