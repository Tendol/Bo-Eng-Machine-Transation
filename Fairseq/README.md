# Bo-Eng-Machine-Transation
Tibetan to English Machine Translation

## Dependencies:

pip install transformers
pip install pytorch
pip install tensorflow
pip install fairseq

## files: 
fairseq.sh - bash script to run the Machine Training pipeline using fairseq 

## Instructions:

cd preProcessing
bash prepare-bo-en.sh
cd ..
bash fairseq.sh 

