# Recurrent Neural Network Grammars
Code for the Recurrent Neural Network Grammars paper (NAACL 2016), by Chris Dyer, Adhiguna Kuncoro, Miguel Ballesteros, and Noah A. Smith. The code is written in C++.

# Prerequisites
cmake version 2.8+ \\
The latest development version of Eigen \\
C++ compiler (supporting the C++11 language standard) \\
Boost libraries

# Build instructions
Assuming the latest development version of Eigen is stored at: /opt/tools/eigen-dev 

    mkdir build
    cd build
    cmake -DEIGEN3_INCLUDE_DIR=/opt/tools/eigen-dev ..
    make -j2

    
# Sample input format: 
sample\_input\_english.txt (English PTB) and sample\_input\_chinese.txt (Chinese CTB)

# Oracles
The oracle converts the bracketed phrase-structure tree into a sequence of actions. \\ 
The script to obtain the oracle also converts singletons in the training set and unknown words in the dev and test set into a fine-grained set of 'UNK' symbols 

### Obtaining the oracle for the discriminative model
    python get_oracle.py [training file] [training file] > train.oracle
    python get_oracle.py [training file] [dev file] > dev.oracle
    python get_oracle.py [training file] [test file] > test.oracle

### Obtaining the oracle for the generative model
    python get_oracle_gen.py [training file] [training file] > train_gen.oracle
    python get_oracle_gen.py [training file] [dev file] > dev_gen.oracle
    python get_oracle_gen.py [training file] [test file] > test_gen.oracle

# Discriminative Model
The discriminative variant of the RNNG is used as a proposal distribution for decoding the generative model (although it can also be used for decoding on its own). To save time we recommend training both models in parallel\\
On the English PTB dataset the discriminative model typically converges after about 3 days on a single-core CPU device. 

### Training the discriminative model

    nohup ../build/nt-parser/nt-parser --cnn-mem 1700 -x -T [training_oracle_file] -d [dev_oracle_file] -C [original_dev_file (PTB bracketed format, see sample_input_english.txt)] -P -t --pretrained_dim [dimension of pre-trained word embedding] -w [pre-trained word embedding] --lstm_input_dim 128 --hidden_dim 128 -D 0.2 > log.txt

The training log is printed to log.txt (including the parameter file where the model is saved to)

### Decoding with discriminative model

    ./nt-parser --cnn-mem 1800 -x -T [training_oracle_file] -p [test_oracle_file] -C [original_test_file (PTB bracketed format, see sample_input_english.txt)] -P --pretrained_dim [dimension of pre-trained word embedding] -w [pre-trained word embedding] --lstm_input_dim 128 --hidden_dim 128 -m [parameter file] > output.txt

    Note: the output will be stored in /tmp/parse/parser_test_eval.xxxx.txt and the parser will output F1 score calculated with EVALB with COLLINS.prm option. The parameter file (following the -m in the command above) can be obtained from log.txt. If training was done using pre-trained word embedding (by specifying the -w and --pretrained_dim options), then decoding must alo use the exact same options used for training.



