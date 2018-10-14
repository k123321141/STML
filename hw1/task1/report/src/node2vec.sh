cat ./kaggle/t1-train.txt > ./kaggle/t1-merge.txt
cat ./kaggle/t1-test-seen.txt >> ./kaggle/t1-merge.txt
python ../lib/node2vec/src/main.py --walk-length 800 --num-walks 100 --window-size 20  --iter 3 --directed \
    --dimensions 128 \
    --input ./kaggle/t1-merge.txt \
    --output ./t1.emb
