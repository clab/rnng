# Recurrent Neural Network Grammars
Code for the Recurrent Neural Network Grammars paper (NAACL 2016), by Chris Dyer, Adhiguna Kuncoro, Miguel Ballesteros, and Noah A. Smith. The code is written in C++.

# Prerequisites
cmake version 2.8+
The latest development version of Eigen

# Build instructions
Assuming the latest development version of Eigen is stored at: /opt/tools/eigen-dev 

    mkdir build
    cd build
    cmake -DEIGEN3_INCLUDE_DIR=/opt/tools/eigen-dev ..
    make -j2

    
# Sample input format: 
sample\_input\_english.txt (English PTB) and sample\_input\_chinese.txt (Chinese CTB)

# Oracles
The oracle converts the bracketed phrase-structure tree into a sequence of actions. 
The script to obtain the oracle also converts singletons in the training set and unknown words in the dev and test set into a fine-grained set 'UNK' symbols 

# Obtaining the oracle for the discriminative model
    python get_oracle.py [training file] [training file] > train.oracle
    python get_oracle.py [training file] [dev file] > dev.oracle
    python get_oracle.py [training file] [test file] > test.oracle

# Obtaining the oracle for the generative model
    python get_oracle_gen.py [training file] [training file] > train_gen.oracle
    python get_oracle_gen.py [training file] [dev file] > dev_gen.oracle
    python get_oracle_gen.py [training file] [test file] > test_gen.oracle


