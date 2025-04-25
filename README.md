# WheatGP

#### **Overview of WheatGP**

Wheat's polyploid structure involves intricate allelic interactions and pronounced epistatic effects, which require advanced feature extraction capabilities to accurately capture the underlying genetic architecture. WheatGP is designed to improve the phenotype prediction accuracy by modeling both additive genetic effects and epistatic genetic effects. It is primarily composed of a convolutional neural network (CNN) module and a long short-term memory (LSTM) module. 

In CNN module, the genotypic input in the form of a one-dimensional vector is evenly divided into five slices, each slice is processed through the multi-layer CNN structure. The CNN module is concentrated on learning the local features of the slices. The LSTM module is employed to further extract global features from genomic data. The interactions between non-allelic genes at different gene loci are captured by its long short-term memory mechanism more efficiently. Ultimately, the fully connected layer in the Prediction module maps the extracted distributed feature representations to the sample tag space, enabling phenotype prediction of wheat. 

#### Environment Requirements

WheatGP is engineered to leverage either a GPU or a CPU for computations. The selection of hardware significantly influences the model's performance. During the training phase, we strongly recommend running WheatGP on a GPU. As for the CPU, it is more suitable for validation on small - scale datasets and for the inference phase.

| Dependencies | Version |
| ------------ | ------- |
| Python       | 3.10.15 |
| torch        | 1.13.1  |
| numpy        | 1.26.4  |
| pandas       | 2.2.2   |
| tqdm         | 4.66.5  |
| matplotlib   | 3.9.2   |

You can use `conda` to create a new environment named `wheatgp` and install all the required dependencies at once. First, make sure `conda` is installed on your system. If conda is not available on your system, you can alternatively use pip to install the above - mentioned packages.

```
conda env create -f environment.yml
```

#### A selectable GUI for you

We have provided the original scripts for the model training and validation phases. The code of the enhanced WheatGP (including transfer learning and Bayesian - based hyperparameter optimization) can be obtained by contacting the corresponding author of the paper. For your convenience in testing, we have created a localized **GUI**. The specific usage is as follows：

After creating the “wheatgp” environment, you need to install additional dependencies to ensure the proper functioning of the **GUI** application.

```
conda activate wheatgp

pip install ttkbootstrap
```

Navigate to the `*/home/user/WheatGP/GUI`* directory and run the GUIcode.py script. The **GUI** will then start.

```
cd /home/user/WheatGP/GUI

python GUIcode.py
```

****NOTE: ****

Your original data can be in the form of genotype and phenotype in.csv format. Prior to training, this data needs to undergo preprocessing. During the training process, genotype and phenotype data in the.pkl format with dictionary type are required. Before commencing training, you must set the path for saving the model to execute the training operation. In addition, parameters such as the **LSTM dim** also need to be set correctly. If there are any errors in your settings, the information box below will **display the error reasons**. We provide a  model to demonstrate the performance of WheatGP in wheat599 dataset. You only need to import the **Existing Model** and the validation data in `./data_example`. It should be noted that the phenotypes here are the phenotypes after normalization. After the test is completed, you can click the **Save Test Results** button. At this time, a result file containing the true values and predicted values will be generated in the root directory.

If you encounter any problems during the use process or have further needs for discussing the project, such as obtaining the code of the enhanced WheatGP, please contact us. We are glad to provide you with efficient assistance. 
