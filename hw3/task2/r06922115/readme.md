### predict.sh
會先下載兩個model
以及我處理過的training data，這部分是為了建vocabulary dictionary
而predict時考慮效能，目前只包含without attention的model



### training code
seq2seq-translation.ipynb
seq2seq-translation-rnn.ipynb

### lib 
pytorch==0.4.1
keras==2.1.6
numpy==1.14.0
tensorflow==1.8.0
