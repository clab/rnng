# Recurrent Neural Network Grammars
Code for the [Recurrent Neural Network Grammars](https://arxiv.org/abs/1602.07776) paper (NAACL 2016), by Chris Dyer, Adhiguna Kuncoro, Miguel Ballesteros, and Noah A. Smith, after the Corrigendum (last two pages on the ArXiv version of the paper). The code is written in C++.

# Citation
@inproceedings{dyer-rnng:16,
 author = {Chris Dyer and Adhiguna Kuncoro and Miguel Ballesteros and Noah A. Smith},
 title = {Recurrent Neural Network Grammars},
 booktitle = {Proc. of NAACL},
 year = {2016},
} 

# Prerequisites
 * A C++ compiler supporting the [C++11 language standard](https://en.wikipedia.org/wiki/C%2B%2B11) 
 * [Boost](http://www.boost.org/) libraries
 * [Eigen](http://eigen.tuxfamily.org) (latest development release)
 * [CMake](http://www.cmake.org/)
 * [EVALB](http://nlp.cs.nyu.edu/evalb/) (latest version. IMPORTANT: please put the EVALB folder on the same directory as `get_oracle.py` and `sample_input_chinese.txt` to ensure compatibility)

# Build instructions
Assuming the latest development version of Eigen is stored at: /opt/tools/eigen-dev 

    mkdir build
    cd build
    cmake -DEIGEN3_INCLUDE_DIR=/opt/tools/eigen-dev ..
    make -j2

    
# Sample input format: 
`sample_input_english.txt` (English PTB) and `sample_input_chinese.txt` (Chinese CTB)

# Oracles
The oracle converts the bracketed phrase-structure tree into a sequence of actions.     
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
The discriminative variant of the RNNG is used as a proposal distribution for decoding the generative model (although it can also be used for decoding on its own). To save time we recommend training both models in parallel.      
     
On the English PTB dataset the discriminative model typically converges after about 3 days on a single-core CPU device. 

### Training the discriminative model

    nohup build/nt-parser/nt-parser --cnn-mem 1700 -x -T [training_oracle_file] -d [dev_oracle_file] -C [original_dev_file (PTB bracketed format, see sample_input_english.txt)] -P -t --pretrained_dim [dimension of pre-trained word embedding] -w [pre-trained word embedding] --lstm_input_dim 128 --hidden_dim 128 -D 0.2 > log.txt

IMPORTANT: please run the command at the same folder where `remove_dev_unk.py` is located.    

If not using pre-trained word embedding, then remove the `--pretrained_dim` and `-w` flags.    

The training log is printed to `log.txt` (including information on where the parameter file for the model is saved to, which is used for decoding under the -m option below)

### Decoding with discriminative model

    build/nt-parser/nt-parser --cnn-mem 1700 -x -T [training_oracle_file] -p [test_oracle_file] -C [original_test_file (PTB bracketed format, see sample_input_english.txt)] -P --pretrained_dim [dimension of pre-trained word embedding] -w [pre-trained word embedding] --lstm_input_dim 128 --hidden_dim 128 -m [parameter file] > output.txt

Note: the output will be stored in `/tmp/parse/parser_test_eval.xxxx.txt` and the parser will output F1 score calculated with EVALB with COLLINS.prm option. The parameter file (following the -m in the command above) can be obtained from `log.txt`.       

If training was done using pre-trained word embedding (by specifying the -w and --pretrained\_dim options) or POS tags (-P option), then decoding must alo use the exact same options used for training.

# Generative Model
The generative model achieved state of the art results, and decoding is done using sampled trees from the trained discriminative model     
For the best results the generative model takes about 7 days to converge.

### Training the generative model
    nohup build/nt-parser/nt-parser-gen -x -T [training_oracle_generative] -d [dev_oracle_generative] -t --clusters clusters-train-berk.txt --input_dim 256 --lstm_input_dim 256 --hidden_dim 256 -D 0.3 > log_gen.txt

The training log is printed to `log_gen.txt`, including information on where the parameters of the model is saved to, which is used for decoding later.

# Decoding with the generative model
Decoding with the generative model requires sample trees from the trained discriminative model

### Sampling trees from the discriminative model
     build/nt-parser/nt-parser --cnn-mem 1700 -x -T [training_oracle_file] -p [test_oracle_file] -C [original_test_file (PTB bracketed format, see sample_input_english.txt)] -P --pretrained_dim [dimension of pre-trained word embedding] -w [pre-trained word embedding] --lstm_input_dim 128 --hidden_dim 128 -m [parameter file of trained discriminative model] --alpha 0.8 -s 100 > test-samples.props 

important parameters

 * s = # of samples (all reported results used 100)
 * alpha = posterior scaling (since this is a proposal, a higher entropy distribution is probably good, so a value < 1 is sensible. All reported results used 0.8)

### Prepare samples for likelihood evaluation

    utils/cut-corpus.pl 3 test-samples.props > test-samples.trees

### Evaluate joint likelihood under generative model

    build/nt-parser/nt-parser-gen -x -T [training_oracle_generative] --clusters clusters-train-berk.txt --input_dim 256 --lstm_input_dim 256 --hidden_dim 256 -p test-samples.trees -m [parameters file from the trained generative model, see log_gen.txt] > test-samples.likelihoods

### Estimate marginal likelihood (final step to get language modeling ppl)

    utils/is-estimate-marginal-llh.pl 2416 100 test-samples.props test-samples.likelihoods > llh.txt 2> rescored.trees

 * 100 = # of samples
 * 2416 = # of sentences in test set
 * `rescored.trees` will contain the samples reranked by p(x,y)
      
The file `llh.txt` would contain the final language modeling perplexity after marginalization (see the last lines of the file)

### Compute generative model parsing accuracy (final step to get parsing accuracy from the generative model)

    utils/add-fake-preterms-for-eval.pl rescored.trees > rescored.preterm.trees
    utils/replace-unks-in-trees.pl [Discriminative oracle for the test file] rescored.preterm.trees > hyp.trees    
    utils/remove_dev_unk.py [gold trees on the test set (same format as sample_input_english.txt)] hyp.trees > hyp_final.trees
    EVALB/evalb -p COLLINS.prm [gold trees on the test set (same format as sample_input_english.txt)] hyp_final.trees > parsing_result.txt

The file `parsing_result.txt` contains the final parsing accuracy using EVALB

# Contact
If there are any issues, please let us know at adhiguna.kuncoro [ AT SYMBOL ] gmail [ DOT ] com
