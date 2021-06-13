# StableNet

Unofficial implement of [Deep Stable Learning for Out-Of-Distribution Generalization](https://arxiv.org/pdf/2104.07876.pdf).

## Example

Use the office31 dataset, webcam as the training set and dslr as the validation set. 
The validation accuracy is about 0.99.  And the saliency maps shows that StableNet does removes irrelevant feature information.

|      Model      |               Saliency maps                |
| :-------------: | :----------------------------------------: |
|    resnet18     |   ![origin](assets/resnet.png "Origin")    |  |
| resnet18_stable | ![recon](assets/resnet_stable.png "Recon") |