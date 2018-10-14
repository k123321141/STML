python ../node2vec/src/main.py --walk-length 800 --num-walks 100 --window-size 100  --iter 10 --directed \
    --input ./kaggle/t2-fake.txt \
    --output ./t2_fake.emb
