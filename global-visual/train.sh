python main.py --dataset shanghai --src FFPP --tar MidP --epoch 500 --time 8 --lr 3e-4 --model vit_b_32 &&
python main.py --dataset shanghai --src FFPP --tar MidP --epoch 500 --time 8 --lr 3e-4 --model vit_b_16 &&
python main.py --dataset shanghai --src FFPP --tar MidP --epoch 500 --time 8 --lr 3e-4 --model vit_l_14 &&
python main.py --dataset shanghai --src FFPP --tar MidP --epoch 500 --time 8 --lr 3e-4 --model rn101 &&
python main.py --dataset shanghai --src FFPP --tar MidP --epoch 500 --time 8 --lr 3e-4 --model rn50

python main.py --dataset ubnormal --src FFPP --tar MidP --epoch 300 --time 8 --lr 3e-4 --model vit_b_16 &&
python main.py --dataset ubnormal --src FFPP --tar MidP --epoch 300 --time 8 --lr 3e-4 --model vit_b_32 &&
python main.py --dataset ubnormal --src FFPP --tar MidP --epoch 300 --time 8 --lr 3e-4 --model vit_l_14 &&
python main.py --dataset ubnormal --src FFPP --tar MidP --epoch 300 --time 8 --lr 3e-4 --model rn101 &&
python main.py --dataset ubnormal --src FFPP --tar MidP --epoch 300 --time 8 --lr 3e-4 --model rn50

python main.py --dataset nwpu --src FFPP --tar MidP --epoch 500 --time 8 --lr 3e-4 --model vit_b_16 &&
python main.py --dataset nwpu --src FFPP --tar MidP --epoch 500 --time 8 --lr 3e-4 --model vit_b_32 &&
python main.py --dataset nwpu --src FFPP --tar MidP --epoch 500 --time 8 --lr 3e-4 --model vit_l_14 &&
python main.py --dataset nwpu --src FFPP --tar MidP --epoch 500 --time 8 --lr 3e-4 --model rn101 &&
python main.py --dataset nwpu --src FFPP --tar MidP --epoch 500 --time 8 --lr 3e-4 --model rn50