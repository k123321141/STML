python ../lib/node2vec/src/main.py --walk-length 800 --num-walks 100 --window-size 20  --iter 3 --directed \
    --dimensions 128 \
    --input ./kaggle/t2-fake.txt \
    --output ./t2.emb
