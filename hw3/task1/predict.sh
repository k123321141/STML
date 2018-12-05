#!/bin/bash
mkdir data
#wget https://www.dropbox.com/s/4c536h3nreb09wc/r1-r2.txt?dl=0 -O data/r1-r2.txt
#wget https://www.dropbox.com/s/lpza12vae7u16oz/r3-r4.txt?dl=0 -O data/r3-r4.txt
mkdir src
mkdir src/models
# wget https://www.dropbox.com/s/sgds892rbdwtyl3/decoder-att.pl?dl=0 -O src/models/decoder-att.pl
# wget https://www.dropbox.com/s/121fc18g1vs6r95/encoder-att.pl?dl=0 -O src/models/encoder-att.pl
#wget https://www.dropbox.com/s/ojd6cg1z0q77bkr/decoder-rnn.pl?dl=0 -O src/models/decoder-rnn.pl
#wget https://www.dropbox.com/s/1ga27rrkkmwazbn/encoder-rnn.pl?dl=0 -O src/models/encoder-rnn.pl
input=$1
output=$2
python2 src/predict_rnn.py $1 $2
