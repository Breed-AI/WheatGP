# WheatGP
WheatGP, a genomic prediction method based on CNN and LSTM
WheatGP is designed to improve the phenotype prediction accuracy by modeling both additive genetic effects and epistatic genetic effects. It is primarily composed of a convolutional neural network (CNN) module and a long short-term memory (LSTM) module. The multi-layer CNNs within the CNN module focus on capturing short-range dependencies within the genomic sequence. Meanwhile, the LSTM module, with its unique gating mechanism, is designed to retain long-distance dependencies relationships between gene loci in the features.

You can use the following command to create and activate the same environment as this project with one click:
> ```bash
> conda env create -f environment.yml
> conda activate wheatgp_env
> ```
