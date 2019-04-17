# Low-Power and High-Speed Deep FPGA Inference Engines for Weed Classification at the Edge

![](https://img.shields.io/badge/license-GPL-blue.svg)
![DOI](https://img.shields.io/badge/DOI-10.1109%2FACCESS.2019.2911709-brightgreen.svg)

Github Repository Detailing the *DeepWeedsX* Dataset and our Specific Code Level Implementations for *'Low-Power and High-Speed Deep FPGA Inference Engines for Weed Classification at the Edge'*, published in IEEE Access.

## Abstract
Deep Neural Networks (DNNs) have recently achieved remarkable performance in a myriad of applications, ranging from image recognition to language processing. Training such networks on Graphics Processing Units (GPUs) currently offers unmatched levels of performance; however, GPUs are subject to large power requirements. With recent advancements in High Level Synthesis (HLS) techniques, new methods for accelerating deep networks using Field Programmable Gate Arrays (FPGAs) are emerging. FPGA-based DNNs present substantial advantages in energy efficiency over conventional CPU- and GPUaccelerated networks. Using the Intel FPGA Software Development Kit (SDK) for OpenCL development environment, networks described using the high-level OpenCL framework can be accelerated targeting heterogeneous platforms including CPUs, GPUs, and FPGAs. These networks, if properly customized on GPUs and FPGAs, can be ideal candidates for learning and inference in resource-constrained portable devices such as robots and the Internet of Things (IoT) edge devices, where power is limited and performance is critical. Here, we introduce GPU- and FPGA-accelerated deterministically binarized DNNs, tailored toward weed species classification for robotic weed control. Our developed networks are trained and benchmarked using a publicly available weed species dataset, named DeepWeeds, which includes close to 18,000 weed images. We demonstrate that our FPGA-accelerated binarized networks significantly outperform their GPU-accelerated counterparts, achieving a >7-fold decrease in power consumption, while performing inference on weed images 2.86 times faster compared to our best performing baseline full-precision GPU implementation. These significant benefits are gained whilst losing only 1.17% of validation accuracy. This is a significant step toward enabling deep inference and learning on IoT edge devices, and smart portable machines such as an agricultural robot, which is the target application in this paper.

## Preface
The *DeepWeedsX* dataset consists of 17,508 unique 256x256 colour images in 9 classes. There are 15,007 training images and 2,501 test images. These images were collected in situ from eight rangeland environments across northern Australia.

Liaison with land care groups and property owners across northern Australia led to the selection of eight target weed species
for the the collection of a large weed species image dataset; Chinee Apple (Ziziphus mauritiana), Lantana, Parkinsonia (Parkinsonia aculeata), Parthenium (Parthenium hysterophorus), Prickly Acacia (Vachellianilotica), Rubber vine (Cryptostegia grandiflora), Siam weed (Chromolaena odorata) and Snakeweed (Stachytarphetaspp).

*DeepWeedsX is a subset of the DeepWeeds dataset, which was originally collected by Alex Olsen, and has previously been made openly accessible. We present a labeled variant with clearly defined training and test datasets. A validation dataset may be constructed for parameter optimization using a subset of the labeled training dataset. All original data collection was funded by the Australian Government Department of Agriculture and Water Resources Control Tools and Technologies for Established Pest Animals and Weeds Programme (Grant No. 4-53KULEI).*

## Dataset Class Distribution

The following is the class distribution of the dataset:

|    Class    |    Species Label     |    Training    |    Test          |    Total    |
|-------------|----------------------|----------------|------------------|-------------|
|    0        |    Chinee Apple      |    964         |    161           |    1,125     |
|    1        |    Lantana           |    912         |    152           |    1,064     |
|    2        |    Parkinsonia       |    884         |    147           |    1,031     |
|    3        |    Parthenium        |    876         |    146           |    1,022     |
|    4        |    Prickly Acacia    |    910         |    152           |    1,062     |
|    5        |    Rubber Vine       |    865         |    144           |    1,009     |
|    6        |    Siam Weed         |    921         |    153           |    1,074     |
|    7        |    Snake Weed        |    871         |    145           |    1,016     |
|    8        |    Other             |    7,804        |    1301          |    9,105     |
|    **Total**    |                      |    **15,007**       |    **2,501**          |    **17,508**    |

## Class Labels & Dataset Avaliability
We provide all class labels and dataset images, to encourage future comparison to our implementations. Alternatively, *DeepWeedsX* is publicly available on Kaggle [here](https://www.kaggle.com/coreylammie/deepweedsx).

### Class Labels
All class label files consist of Comma Seperated Values (CSVs) detailing the label and species, for example: *20161207-111327-0.jpg, 0* denotes that *20161207-111327-0.jpg* belongs to class 0 (Chinee Apple).

[Training Set Labels](Class%20Labels/train_set_labels.csv)

[Test Set Lables](Class%20Labels/train_set_labels.csv)

### Dataset Images
All images are compressed in a single ZIP archive, and are labelled as per the class file labels.

[DeepWeeds Images](https://coreylammie.me/low-power-and-high-speed-deep-fpga-inference-engines-for-weed-classification/DeepWeeds_Images_256.zip)

## Dataset Loaders
Currently, we only provide a dataset loader for the [PyTorch](https://pytorch.org/) library. [Loader.py](Dataset%20Loaders/PyTorch/Loader.py) requires [Preprocessing.py](Dataset%20Loaders/PyTorch/Preprocessing.py) and [Sampler.py](Dataset%20Loaders/PyTorch/Sampler.py) to function. Example usage:


~~~~
from Loader import *
from Preprocessing import *

train_data_loader, test_data_loader = loadDeepWeeds(batch_size=32,
                                                    shuffle=True,
                                                    pre_processing_transform=IPT(),
                                                    use_imbalanced_dataset=True,
                                                    image_directory_path="DeepWeedsImages",
                                                    train_csv_path="train_set_labels.csv",
                                                    test_csv_path="test_set_labels.csv")
~~~~

In the future, we intend to provide dataset loaders for [Keras](https://keras.io/) and [Tensorflow](https://www.tensorflow.org/).

## Baseline Implementations
We provide all code required to reproduce all our baseline implementations using the PyTorch library. All dependancies can be installed using:

~~~~
pip -r install requirements.txt
~~~~

where requirements.txt is avaliable [here](requirements.txt).

<table>
  <tr>
    <th>Network Architecture</th>
    <th>(3, 32, 32)</th>
    <th>(3, 64, 64)</th>
    <th>(3, 224, 224)</th>
  </tr>
  <tr>
    <td colspan="4">IPT</td>
  </tr>
  <tr>
    <td>VGG-16</td>
    <td>86.72</td>
    <td>89.48</td>
    <td>91.08</td>
  </tr>
  <tr>
    <td>DenseNet-128-32</td>
    <td>90.08</td>
    <td>91.52</td>
    <td>89.40</td>
  </tr>
  <tr>
    <td>WRN-28-10</td>
    <td>88.88</td>
    <td>93.36</td>
    <td>94.82</td>
  </tr>
  <tr>
    <td colspan="4">FIPT</td>
  </tr>
  <tr>
    <td>VGG-16</td>
    <td>81.45</td>
    <td>89.12</td>
    <td>93.04</td>
  </tr>
  <tr>
    <td>DenseNet-128-32</td>
    <td>85.89</td>
    <td>86.05</td>
    <td>94.24</td>
  </tr>
  <tr>
    <td>WRN-28-10</td>
    <td>85.97</td>
    <td>90.72</td>
    <td>95.85</td>
  </tr>
</table>

## Citation
To cite the repository/paper, kindly use the following BibTex entry:

```
@ARTICLE{8693488, 
author={C. {Lammie} and A. {Olsen} and T. {Carrick} and M. R. {Azghadi}}, 
journal={IEEE Access}, 
title={Low-Power and High-Speed Deep FPGA Inference Engines for Weed Classification at the Edge}, 
year={2019}, 
volume={}, 
number={}, 
pages={1-1}, 
keywords={Machine Learning (ML);Deep Neural Networks (DNNs);Convolutional Neural Networks (CNNs);Binarized Neural Networks (BNNs);Internet of Things (IoT);Field Programmable Gate Arrays (FPGAs);High-level Synthesis (HLS);Weed Classification}, 
doi={10.1109/ACCESS.2019.2911709}, 
ISSN={2169-3536}, 
month={},}
```

## License
All code is licensed under the GNU General Public License v3.0. Details pertaining to this are avaliable at: https://www.gnu.org/licenses/gpl-3.0.en.html
