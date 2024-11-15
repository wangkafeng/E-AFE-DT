# E-AFE-DT
E-AFE-DT is an E-AFE improvement version. Replace RNN-based agent with Decision Transformer-based agent.
The environment is PyTorch.

https://github.com/wangkafeng/E-AFE
The environment is Tensorflow.

The paper of E-AFE method information is as follows.
Toward Efficient Automated Feature Engineering
2023 IEEE 39th International Conference on Data Engineering (ICDE)
https://ieeexplore.ieee.org/document/10184603


Decision Transformer reference :

https://github.com/kzl/decision-transformer 

https://github.com/daniellawson9999/online-decision-transformer



E-AFE-DT run with Decision Transformer agent:  --controller dt

# python Main_minhash_tran.py --multiprocessing False --epochs 1 --dataset PimaIndian --minhash True --controller trajectory
python Main_minhash_tran.py --multiprocessing False --epochs 1 --dataset PimaIndian --minhash True --controller dt
