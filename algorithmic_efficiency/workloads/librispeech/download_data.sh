#! /bin/bash

for d in dev test; do
    for s in clean other; do
	echo $d, $s
	wget http://www.openslr.org/resources/12/$d-$s.tar.gz
	tar xzvf $d-$s.tar.gz
    done
done

wget http://www.openslr.org/resources/12/raw-metadata.tar.gz
wget http://www.openslr.org/resources/12/train-clean-100.tar.gz


# Untar files
tar xzvf raw-metadata.tar.gz
tar xzvf train-clean-100.tar.gz

