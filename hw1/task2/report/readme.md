### ceiba容量限制
請將kaggle資料放到/src/底下
請使用node2vec.sh產生t2.emb
請下載pre-trained word embedding glove.6B.300d.txt，並放在/src/glove/之下
http://nlp.stanford.edu/data/glove.6B.zip


### training code
task2.ipynb
使用python 2.7

### own generated data
./kaggle/t2-fake.txt
使用fake link產生的資料
請使用 task2.ipynb的第一個code block

./t2.emb
利用./lib/node2vec產生的graph embedding

python ../node2vec/src/main.py --walk-length 800 --num-walks 100 --window-size 20  --iter 3 --directed \
    --dimensions 128 \
    --input ./kaggle/t2-fake.txt \
    --output ./t2.emb

### lib 
pytorch==0.4.1
keras==2.1.6
numpy==1.14.0
tensorflow==1.8.0
