import tensorflow_datasets as tfds 

dataset_dir='/home/smedapati/tensorflow_datasets/downloads/manual/'

for ds_name in ['wmt14_translate/de-en:1.0.0', 'wmt17_translate/de-en:1.0.0']:
    tfds.builder(ds_name, data_dir=dataset_dir).download_and_prepare()
