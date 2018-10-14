### ceiba容量限制
請將kaggle資料放到/src/底下
請使用node2vec.sh產生t1.emb


### training code
task1.ipynb
使用python 2.7

### own generated data
./kaggle/t1-merge.txt
將t1-train.txt與t1-seen-test.txt合併的資料

./t1.emb
利用./lib/node2vec產生的graph embedding

python ../node2vec/src/main.py --walk-length 800 --num-walks 100 --window-size 20  --iter 3 --directed \
    --dimensions 128 \
    --input ./kaggle/t1-merge.txt \
    --output ./t1.emb

### lib 
pytorch==0.4.1
keras==2.1.6
numpy==1.14.0
tensorflow==1.8.0
