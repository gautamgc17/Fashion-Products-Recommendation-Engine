# Fashion Product Recommendation Engine 

## Project Overview
- An end-to-end content-based fashion product recommendation system.
- Key components include human keypoint estimation, fashion article detection and localization, gender detection,
embedding generation followed by item similarity search.
- The model pipeline leverages visual similarity of items for product recommendation.
- Given a query fashion article image worn by a model, the system first checks whether it is a front facing or full pose image. If yes, then it is passed through gender detection model to identify its gender. This step is essential to avoid cross gender recommendations. The query image is further passed through object detection and localization model to extract different clothing (primary and secondary) items from the image so that the extracted items can be used to generate embeddings. Then either cosine similarity measure or approximate K-nearest neighbor search algorithm such as FAISS, Annoy can be employed to get top K similar items for each of the extracted items and the same can be recommended to the user. 


_The overall architecture of this project was inspired by the research paper titled [Buy Me That Look: An Approach for Recommending Similar Fashion Products](https://ieeexplore.ieee.org/abstract/document/9565561). The research paper provided valuable insights and served as a foundation for the development and implementation of this project._

```
A. Ravi, S. Repakula, U. K. Dutta and M. Parmar, "Buy Me That Look: An Approach for Recommending Similar Fashion Products," 2021 IEEE 4th International Conference on Multimedia Information Processing and Retrieval (MIPR), Tokyo, Japan, 2021, pp. 97-103, doi: 10.1109/MIPR51284.2021.00022.
```


![Problem Illustration](https://github.com/gautamgc17/Fashion-Products-Recommendation-Engine/blob/e07ead007a9fe79645e315c15133f6f1de1ed2a1/assets/idea.PNG)

## Business Problem:
The business problem addressed in this research paper is the need for a computer vision-based technique to recommend similar fashion products based on images of models or user-generated content. The goal is to provide an enhanced shopping experience on fashion e-commerce platforms by suggesting similar fashion items to the ones worn by the model in the product display page (PDP) image. Thus, the idea is to recommend all the secondary clothes worn by the model in the query image along with the primary product using Deep learning techniques. 

The significance of this problem lies in promoting cross-sells to boost revenue and improving customer experience and engagement. By detecting all products in an image and retrieving similar images from the database, the proposed method aims to recommend not only the primary article corresponding to the user's query but also similar articles for all fashion items worn by the model. This approach enhances the potential for increased sales and customer satisfaction.


## Overall Architecture

![Project Workflow](https://github.com/gautamgc17/Fashion-Products-Recommendation-Engine/blob/e07ead007a9fe79645e315c15133f6f1de1ed2a1/assets/architecture.png)

## 1. Data Acquisition and Exploratory Data Analysis
To begin building the Fashion Recommendation Engine, we require a diverse catalog of fashion products. By utilizing Selenium, a popular browser automation tool, we can search Myntra's e-commerce website using specific keywords and extract product details. This includes saving the product URLs and its associated metadata. Additionally, this can be used to download the images and build a database which will serve as a foundation for recommending fashion products across different categories using embedding generation technique followed by similarity search measures, during the inference stage of the final pipeline.

## 2. Full-shot Image Detection and Gender Classification
In order to recommend relevant items to the user it is essential to first detect the gender of the person present in the query image. Doing so will allow us to generate and query embeddings only on certain subsets of the catalog instead of the whole database. This will also in reducing irrelevant recommendations to the user based on their gender.

To tackle this problem, firstly a [Pose Estimation](https://github.com/quanhua92/human-pose-estimation-opencv) model was utilized to identify a partial or full-shot look image in the PDP page. A full-shot look image refers to the one that is displaying a person from the head to the toe. Thereafter, transfer learning was utilized to develop a [Gender Classification](https://github.com/e0xextazy/gender-est) model using PyTorch framework, which was fine tuned on a variety of publicly available open source datasets.

## 3. Fashion article Detection and Localisation
The front-facing full-shot look image obtained from the previous step contains multiple fashion articles and accessories worn by the model. Corresponding to each of these articles, we have to recommend a list of similar fashion products. For this subtask of identifying different article types, we must crop, or segment out the individual regions Of interests (ROIs) from the full-shot look image. 

Therefore, [YOLOv5](https://github.com/ultralytics/yolov5) object detection model was trained with a custom dataset containing 240 images (180 images in traing set and 60 images in validation set) for detecting and localizing the fashion product articles into three broad categories, namely - topwear, bottomwear and footwear. Transfer Learning technique was employed to [Train YOLOv5 on Custom Dataset](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/) by loading YOLOv5s (small variant of YOLOv5) weights initially trained on the COCO Dataset to fine-tune the model for our use-case. 

The structure of the yaml file for training on custom datasetlooks like:
![yaml file](https://github.com/gautamgc17/Fashion-Products-Recommendation-Engine/blob/e07ead007a9fe79645e315c15133f6f1de1ed2a1/assets/format.PNG)


## 4. Candidate Retrieval (Embedding Generation)
To retrieve similar fashion products based on a query image, we employ embedding generation to represent images/products. This involves using a [Siamese Network](https://keras.io/examples/vision/siamese_network/), a special type of neural network that takes three inputs: anchor, positive, and negative. The anchor and positive inputs represent similar products, while the anchor and negative inputs represent dissimilar products. Through a CNN backbone, in our case - [Resnet50](https://keras.io/api/applications/resnet/) model, the network generates n-dimensional vectors for the anchor, positive, and negative inputs. A specific loss function, known as the Triplet Loss Function, is used to minimize the distance metric between the anchor and positive inputs while maximizing the distance metric between the anchor and negative inputs. This approach enables effective retrieval of fashion products that are similar to the items in the query image.

![Siamese Network](https://github.com/gautamgc17/Fashion-Products-Recommendation-Engine/blob/e07ead007a9fe79645e315c15133f6f1de1ed2a1/assets/siamese_nets.png)

The Pinterest [Shop The Look](https://dl.acm.org/doi/abs/10.1145/3394486.3403372) dataset was used for this task. "Shop The Look" is a dataset taken from "<a href="https://arxiv.org/pdf/1812.01748.pdf"> Wang-Cheng Kang, Eric Kim, Jure Leskovec, Charles Rosenberg, Julian McAuley (2019). Complete the Look: Scene-based Complementary Product Recommendation</a>". In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR'19). 

The dataset includes pairs of scenes and products. The products are images taken in a professional environment, while the scenes show the same products in a more relaxed or informal setting. The dataset follows a specific format, where the scenes and products are encoded with a signature which can be converted into a URL using a function provided in the [STL-Dataset](https://github.com/kang205/STL-Dataset) GitHub repository. The bounding box is represented by four numbers: ```(left, top, right, bottom)``` and all the numbers are normalized.

```
@article{DBLP:journals/corr/abs-1812-01748,
  author       = {Wang{-}Cheng Kang and
                  Eric Kim and
                  Jure Leskovec and
                  Charles Rosenberg and
                  Julian J. McAuley},
  title        = {Complete the Look: Scene-based Complementary Product Recommendation},
  journal      = {CoRR},
  volume       = {abs/1812.01748},
  year         = {2018},
  url          = {http://arxiv.org/abs/1812.01748},
  eprinttype    = {arXiv},
  eprint       = {1812.01748},
  timestamp    = {Mon, 22 Jul 2019 19:11:00 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1812-01748.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## 5. Image Retrieval
After embedding generating step for the fashion product catalog, we proceed to create a pipeline that integrates all the modules. This pipeline takes a query image as input and generates relevant recommendations as output. In this project, [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss) library, which is an open-source library developed by Facebook's AI Research team, has been utilized to retrieve similar articles. It is based on approximate nearest neighbor search algorithm and designed to efficiently search and retrieve similar vectors or embeddings in large-scale datasets. 

![Project Pipeline](https://github.com/gautamgc17/Fashion-Products-Recommendation-Engine/blob/e07ead007a9fe79645e315c15133f6f1de1ed2a1/assets/pipeline.png)

### References
- [Human Pose Estimation Model OpenCV](https://github.com/quanhua92/human-pose-estimation-opencv)
- [Gender Prediction using MobilenetV2](https://github.com/e0xextazy/gender-est)
- [Shop The Look: Building a Large Scale Visual Shopping System at Pinterest](https://dl.acm.org/doi/abs/10.1145/3394486.3403372)
- [Buy Me That Look: An Approach for Recommending Similar Fashion Products](https://ieeexplore.ieee.org/abstract/document/9565561)
- [Complete the Look: Scene-based Complementary Product Recommendation](https://arxiv.org/pdf/1812.01748.pdf)
- [Mean Average Precision (mAP) metric for Object Detection](https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52)
- [You Only Look Once (YoLo): Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640v1.pdf)
- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [Train YOLOv5 on Custom Dataset (Reference-1)](https://blog.roboflow.com/how-to-train-yolov5-on-a-custom-dataset/)
- [Train YOLOv5 on Custom Dataset (Reference-2)](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/)
- [Pinterest’s Shop The Look Data](https://github.com/kang205/STL-Dataset)
- [Image similarity estimation using a Siamese Network with a triplet loss](https://keras.io/examples/vision/siamese_network/)
- [Understanding FAISS for Similarity Search](https://towardsdatascience.com/understanding-faiss-619bb6db2d1a)



