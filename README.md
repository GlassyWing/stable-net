# StableNet

Unofficial implement of [Deep Stable Learning for Out-Of-Distribution Generalization](https://arxiv.org/pdf/2104.07876.pdf).

## Example

Use the caltech101 dataset,  And the saliency maps shows that StableNet does removes irrelevant feature information.

|      Model      |                Saliency maps                 |
| :-------------: | :------------------------------------------: |
|    resnet18     |   ![origin](assets/resnet_3.png "Origin")    |
| resnet18_stable | ![recon](assets/resnet_stable_3.png "Recon") |
|    resnet18     |   ![origin](assets/resnet_4.png "Origin")    |
| resnet18_stable | ![recon](assets/resnet_stable_4.png "Recon") |
|    resnet18     |    ![origin](assets/resnet.png "Origin")     |
| resnet18_stable |  ![recon](assets/resnet_stable.png "Recon")  |

