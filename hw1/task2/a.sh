python ../node2vec/src/main.py --walk-length 80 --num-walks 10 --window-size 10  --iter 1 --directed \
    --input ./kaggle/t1-fake.txt \
    --output ./t1_weak.emb
