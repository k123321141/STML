### training code
task1.ipynb
使用python 2.7

### own generated data
./kaggle/t1-merge.txt
單純合併t1-train.txt 與 t1-test-seen.txt
./t1.emb
利用./lib/node2vec產生的graph embedding
python ../node2vec/src/main.py --walk-length 800 --num-walks 100 --window-size 100  --iter 10 --directed \
    --input ./kaggle/t1-merge.txt \
    --output ./t1.emb

### lib 
pytorch==0.4.1
keras==2.1.6
numpy==1.14.0
tensorflow==1.8.0
