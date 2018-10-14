### ceiba容量限制
請將kaggle資料放到/src/底下
請使用node2vec.sh產生t3.emb
請下載pre-trained word embedding glove.6B.300d.txt，並放在/src/glove/之下
http://nlp.stanford.edu/data/glove.6B.zip


### training code
task3.ipynb
使用python 2.7

### own generated data
./kaggle/t3-fake.txt
使用fake link產生的資料
請使用 task3.ipynb的第一個code block，須先執行第二個block得到時間上的資訊

./t3.emb
利用./lib/node2vec產生的graph embedding

python ../node2vec/src/main.py --walk-length 800 --num-walks 100 --window-size 20  --iter 3 --directed \
    --dimensions 128 \
    --input ./kaggle/t3-fake.txt \
    --output ./t3.emb

### lib 
pytorch==0.4.1
keras==2.1.6
numpy==1.14.0
tensorflow==1.8.0
