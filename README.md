
 download MosaicKD's pre-trained models from [Dropbox (266 M)](https://www.dropbox.com/sh/w8xehuk7debnka3/AABhoazFReE_5mMeyvb4iUWoa?dl=0) and extract them as "mosaic_core/checkpoints/pretrained/*.pth"

 
* **MosaicKD (This work)**
    ```bash
    python kd_mosaic.py --lr 0.1 --batch-size 256 --teacher wrn40_2 --student wrn16_1 --dataset cifar10 --unlabeled cifar10 --epoch 200 --lr 0.1 --local 1 --align 1 --adv 1 --balance 10 --gpu 0 --pipeline=multi_teacher
     python kd_mosaic.py --gpu 0 --pipeline=multi_teacher --dataset cifar10 --unlabeled cifar10 --epochs 100 --ckpt_path /workspace/DFFK2/mosaic_core/checkpoints/0826_e100_n1_db256 --fp16 --logfile n1db256
    ```