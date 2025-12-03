train-vae --latent-dim 16 --epochs 3000 --cycles 3 --dataset ccsne --outdir out_ccsne16 --batch-size 128
train-vae --latent-dim 16 --epochs 3000 --cycles 3 --dataset blip --outdir out_blip16 --batch-size 128

train-vae --latent-dim 32 --epochs 3000 --cycles 3 --dataset ccsne --outdir out_ccsne32 --batch-size 128
train-vae --latent-dim 32 --epochs 3000 --cycles 3 --dataset blip --outdir out_blip32 --batch-size 128
