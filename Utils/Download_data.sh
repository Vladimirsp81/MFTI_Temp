FILE=edges2shoes
URL=https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/$FILE.tar.gz
TAR_FILE=./MFTI_Proj/datasets/$FILE.tar.gz
TARGET_DIR=./MFTI_Proj/datasets/$FILE/
wget -N $URL -O $TAR_FILE
mkdir $TARGET_DIR
tar -zxvf $TAR_FILE -C ./MFTI_Proj/datasets/
rm $TAR_FILE