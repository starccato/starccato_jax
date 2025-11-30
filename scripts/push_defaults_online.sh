git clone git@github.com:starccato/data.git
cp out_ccsne/model.h5 data/weights/starcatto_jax/ccsne_vae_z32.h5
cp out_blip/model.h5 data/weights/starcatto_jax/blip_vae_z32.h5
cd data
git add weights/starcatto_jax/*.h5
git commit -m "Update default VAE weights for starccato_jax v$(python -c 'import starccato_jax;print(starccato_jax.__version__)')"
git push
git filter-repo --force --invert-paths --path weights/starcatto_jax/ccsne_vae_z32.h5 --path weights/starcatto_jax/blip_vae_z32.h5
git push --force
cd ..
rm -rf data