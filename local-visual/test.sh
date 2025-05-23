python test.py --dataset shanghai --time 8 --model vit_b_32 &&
python test.py --dataset shanghai --time 8 --model vit_b_16 &&
python test.py --dataset shanghai --time 8 --model vit_l_14 &&
python test.py --dataset shanghai --time 8 --model rn101 &&
python test.py --dataset shanghai --time 8 --model rn50 &&

python test.py --dataset ubnormal --time 8 --model vit_b_32 &&
python test.py --dataset ubnormal --time 8 --model vit_b_16 &&
python test.py --dataset ubnormal --time 8 --model vit_l_14 &&
python test.py --dataset ubnormal --time 8 --model rn101 &&
python test.py --dataset ubnormal --time 8 --model rn50 &&

python test.py --dataset nwpu --time 8 --model vit_b_32 &&
python test.py --dataset nwpu --time 8 --model vit_b_16 &&
python test.py --dataset nwpu --time 8 --model vit_l_14 &&
python test.py --dataset nwpu --time 8 --model rn101 &&
python test.py --dataset nwpu --time 8 --model rn50 