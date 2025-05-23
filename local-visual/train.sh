python main.py --t 8 --dataset shanghai --epoch 300 --model vit_b_32 --lr 3e-4 --feature 512 &&
python main.py --t 8 --dataset shanghai --epoch 300 --model vit_b_16 --lr 3e-4 --feature 512 &&
python main.py --t 8 --dataset shanghai --epoch 300 --model vit_l_14 --lr 3e-4 --feature 768 &&
python main.py --t 8 --dataset shanghai --epoch 300 --model rn50 --lr 3e-4 --feature 1024 &&
python main.py --t 8 --dataset shanghai --epoch 300 --model rn101 --lr 3e-4 --feature 512 &&

python main.py --t 8 --dataset ubnormal --epoch 300 --model vit_b_32 --lr 3e-4 --feature 512 &&
python main.py --t 8 --dataset ubnormal --epoch 300 --model vit_b_16 --lr 3e-4 --feature 512 &&
python main.py --t 8 --dataset ubnormal --epoch 300 --model vit_l_14 --lr 3e-4 --feature 768 &&
python main.py --t 8 --dataset ubnormal --epoch 300 --model rn50 --lr 3e-4 --feature 1024 &&
python main.py --t 8 --dataset ubnormal --epoch 300 --model rn101 --lr 3e-4 --feature 512

python main.py --t 8 --dataset nwpu --epoch 50 --model vit_b_32 --lr 3e-4 --feature 512 &&
python main.py --t 8 --dataset nwpu --epoch 50 --model vit_b_16 --lr 3e-4 --feature 512 &&
python main.py --t 8 --dataset nwpu --epoch 50 --model vit_l_14 --lr 3e-4 --feature 768 &&
python main.py --t 8 --dataset nwpu --epoch 50 --model rn50 --lr 3e-4 --feature 1024 &&
python main.py --t 8 --dataset nwpu --epoch 50 --model rn101 --lr 3e-4 --feature 512
