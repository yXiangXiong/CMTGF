set -ex
python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --netG unet_256 --which_direction BtoA --dataset_mode aligned --norm batch
