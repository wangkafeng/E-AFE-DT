# E-AFE-DT

https://github.com/wangkafeng/E-AFE

This is E-AFE improvement. Replace RNN-based agent with Decision Transformer-based agent.

Decision Transformer reference :
https://github.com/kzl/decision-transformer 
https://github.com/daniellawson9999/online-decision-transformer

run with Decision Transformer agent:  --controller trajectory
python Main_minhash_tran.py --multiprocessing True --epochs 1 --dataset PimaIndian --minhash False --controller trajectory
