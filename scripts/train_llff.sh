scene=$1

# round 0
python train.py -s ./data/nerf_llff_data/$scene --exp_name $scene --eval --pseudo_loop_iters 0 &&
python tools/loop.py -s ./data/nerf_llff_data/$scene -m ./output/$scene -p 0 &&

# round 1
python train.py -s ./data/nerf_llff_data/$scene --exp_name ${scene}_1 --eval --pseudo_loop_iters 1 -sps &&
python tools/loop.py -s ./data/nerf_llff_data/$scene -m ./output/${scene}_1 -p 1 &&

# round 2
python train.py -s ./data/nerf_llff_data/$scene --exp_name ${scene}_2 --eval --pseudo_loop_iters 2 -sps &&
python tools/loop.py -s ./data/nerf_llff_data/$scene -m ./output/${scene}_2 -p 2 &&

# round 3
python train.py -s ./data/nerf_llff_data/$scene --exp_name ${scene}_3 --eval --pseudo_loop_iters 3 -sps