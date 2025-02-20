# WheatGP
WheatGP, a genomic prediction method based on CNN and LSTM
WheatGP is designed to improve the phenotype prediction accuracy by modeling both additive genetic effects and epistatic genetic effects. It is primarily composed of a convolutional neural network (CNN) module and a long short-term memory (LSTM) module. The multi-layer CNNs within the CNN module focus on capturing short-range dependencies within the genomic sequence. Meanwhile, the LSTM module, with its unique gating mechanism, is designed to retain long-distance dependencies relationships between gene loci in the features.

![image](https://github.com/user-attachments/assets/6855b5ef-a2ed-4c3b-ae71-bda07a0cfe85)

Multiple modules and blocks are hierarchically stacked as shown in Figure, through which the intricate features inherent in genomic data could be effectively learned by WheatGP at diverse levels of abstraction. In CNN module, the genotypic input in the form of a one-dimensional vector is evenly divided into five slices, each slice is processed through the multi-layer CNN structure. The CNN module is concentrated on learning the local features of the slices. The LSTM module is employed to further extract global features from genomic data. The interactions between non-allelic genes at different gene loci are captured by its long short-term memory mechanism to modeling the additive and epistatic effects of genes. Ultimately, the fully connected layer in the Prediction module maps the extracted distributed feature representations to the sample tag space, enabling phenotype prediction of wheat. The shape adjustment block could perform a series of adaptive linear operations on the features extracted by the previous network layer, feeding them into the subsequent network layer.

You can use the following command to create and activate the same environment as this project with one click:
> ```bash
> conda env create -f environment.yml
> conda activate wheatgp_env
> ```
