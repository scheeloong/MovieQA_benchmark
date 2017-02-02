#!/bin/bash
#
# MovieQA
# 29.03.2016

# Burst tarballs if argument is not empty
BURST=$1

### Download text stories
echo "************* MovieQA *************"
echo "Downloading text-answering stories"
echo "***********************************"
read -p 'Press [Enter] to continue...'

mkdir story
wget http://movieqa.cs.toronto.edu/dl_data/7556a595e71c42c1a9eb0640c5f98759/text-plot -O story/plot.tar.gz
wget http://movieqa.cs.toronto.edu/dl_data/7556a595e71c42c1a9eb0640c5f98759/text-splitplot -O story/splitplot.tar.gz
wget http://movieqa.cs.toronto.edu/dl_data/7556a595e71c42c1a9eb0640c5f98759/text-script -O story/script.tar.gz
wget http://movieqa.cs.toronto.edu/dl_data/7556a595e71c42c1a9eb0640c5f98759/text-subtt -O story/subtt.tar.gz

if [ ! -z $BURST ]; then
    # Burst everything and go back
    cd story
    tar -xf plot.tar.gz
    tar -xf splitplot.tar.gz
    tar -xf script.tar.gz
    tar -xf subtt.tar.gz
    cd ..

else
    echo "************* MovieQA *************"
    echo "Please burst the tarballs in 'MOVIEQA_BASE/story/' yourself!"
    echo "***********************************"
    echo
fi


### Ready!
echo "************* MovieQA *************"
echo "Your copy of the story sources is downloaded, good luck!"
echo "***********************************"

