# Recurrent Neural Network Grammars
Code for the [What Do Recurrent Neural Network Grammars Learn About Syntax?](https://arxiv.org/abs/1611.05774) paper (EACL 2017), by Adhiguna Kuncoro, Miguel Ballesteros, Lingpeng Kong, Chris Dyer, Graham Neubig, and Noah A. Smith. 

# Release
The code is `nt-parser-gen-attention-gated-stack-only.cc`. The simplest step to run the code is to copy it to the `nt-parser` directory and modify the `nt-parser/CMakeLists.txt` file to include and also build this one, and then follow the similar steps as with running the original RNNG.

# Citation
				@inproceedings{kuncoro-rnng:17,
				 author = {Adhiguna Kuncoro and Miguel Ballesteros and Lingpeng Kong and Chris Dyer and Graham Neubig and Noah A. Smith},
				 title = {What Do Recurrent Neural Network Grammars Learn About Syntax?},
				 booktitle = {Proc. of EACL},
				 year = {2017},
				} 

