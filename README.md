
## Introduction
<a href="http://atvs.ii.uam.es/atvs/">
    <img src="./media/BiDA-logo.png" alt="BiDA Lab" title="BiDA Lab" align="right" height="150" width="350"/>
</a>

Welcome to the AI4Food-NutritionDB GitHub repository! 

In this repository, you will find valuable resources that can aid researchers and developers in various ways:

- **AI4Food-NutritionDB Database**: This comprehensive nutrition database incorporates food images and a nutrition taxonomy derived from national and international recommendations. It offers insights into food intake, quality, and categorisation.

- **Food Recognition Systems**: We provide state-of-the-art food recognition systems based on Xception and EfficientNetV2 architectures. These systems are designed to accurately classify food images into different granularity levels, from major categories to subcategories and final products.

- **Experimental Protocol**: For the research community, we offer a benchmark protocol that facilitates experiments with different categorisation levels.

- **Pre-trained Models**: We also provide our pre-trained Deep Learning models, fine-tuned with the AI4Food-NutritionDB dataset.

For further details, please visit [our article](https://doi.org/10.1007/s11042-024-19161-4).

## Getting Started

To get started, visit the respective sections in the repository and follow the provided instructions. For any questions or collaborations, please don't hesitate to contact us.

**Contact**: For inquiries, contact us at sergio.romero@uam.es or ruben.tolosana@uam.es.

## Table of content

- [Overview](#overview)  
    - [AI4Food-NutritionDB Food Image Database](#ai4fooddb)
    - [Food Recognition Systems](#systems)
    - [References](#references)
- [Download AI4Food-NutritionDB database](#download) 
- [Experimental Protocol](#protocol) 
- [How to use the Food Recognition Systems](#models) 
    - [Install & Requirements](#install)
    - [Run the Model](#run)
- [Citation](#cite)
- [Contact](#contact)

## <a name="overview">Overview<a>

Leading a healthy lifestyle has become one of the most challenging goals in today's society due to our sedentary lifestyle and poor eating habits. As a result, national and international organisms have made numerous efforts to promote healthier food diets and physical activity habits. However, these recommendations are sometimes difficult to follow in our daily life and they are also based on a general population. As a consequence, a new area of research, **personalised nutrition**, has been conceived focusing on individual solutions through smart devices and Artificial Intelligence (AI) methods.

This study presents the **AI4Food-NutritionDB database**, the first nutrition database that considers food images and a nutrition taxonomy based on recommendations by national and international organisms. In addition, **four different categorisation levels** are considered following nutrition experts: **6 nutritional levels, 19 main categories (e.g., 'Meat'), 73 subcategories (e.g., 'White Meat'), and 893 final food products (e.g., 'Chicken')**. The AI4Food-NutritionDB opens the doors to new food computing approaches in terms of food intake frequency, quality, and categorisation. Also, in addition to the database, we propose a standard experimental protocol and benchmark including three tasks based on the nutrition taxonomy (i.e., category, subcategory, and final product) to be used for the research community. Finally, we also **release our Deep Learning models** trained with the AI4Food-NutritionDB, which can be used as pre-trained models, achieving accurate recognition results with challenging food image databases. 

<p align="center"><img src="./media/AI4Food-NutritionDB Framework.png" alt="framework" title="AI4Food-NutritionDB Framework"/></p>

### <a name="ai4fooddb">AI4Food-NutritionDB Food Image Database<a>
AI4Food-NutritionDB has been built by combining food images from 7 different databases in the state of the art, considering food products from all over the world. The seven food image databases are described below:
    

+ [UECFood-256](http://foodcam.mobi/dataset256.html) [1]: UECFood-256 contains 256 food categories and more than 30K Japanese food images from different platforms such as Bing Image Search, Flickr or Twitter (using web scraping acquisition protocol). In addition, they used Amazon Mechanical Turk (AMT) in order to select and label food images and their corresponding food product.
    
+ [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) [2]: This database counts more than 100K food images and 101 unique food products from different world regions. All the images were collected from the FoodSpotting application, a social platform where people uploaded and shared food images. 
    
+ [Food-11](https://www.kaggle.com/vermaavi/food11) [3]: Singla et al. analysed the eating behaviour to create a dataset that comprised most of the food groups consumed in the United States. This way they defined 11 general food categories from the  United States Department of Agriculture (USDA): bread, dairy products, dessert, eggs, fried food, meat, noodles/pasta, rice, seafood, soups, and vegetables/fruits. They combined 3 different databases (Food-101, UECFood-100, and UECFood-256) and 2 social platforms (Flickr and Instagram) to finally accumulate more than 16K food images.

+ [FruitVeg-81](https://www.tugraz.at/index.php?id=22808) [4]: Many of the state-of-the-art food image databases haven't fruit or vegetable food products (or it only considers many fruits/vegetables in a photo, not individually). Therefore, this database contains enough food products in order to fulfill some of the underrepresented food groups. FruitVeg-81 database has 81 different fruits and vegetable food products acquired from the self-collected acquisition protocol.

+ [MAFood-121](http://www.ub.edu/cvub/mafood121/) [5]: Considering the 11 most popular cuisines in the world (according to Google Trends), Aguilar et al. released the MAFood-121 database. This dataset contains 121 unique food products and around 21K food images grouped in 10 main categories (bread, eggs, fried food, meat, noodle/pasta, rice, seafood, soup, dumplings, and vegetables), similar to the Food-11 dataset. Also, they utilised the same acquisition protocol (combination), using 3 state-of-the-art public databases (Food-101, UECFood-256, and TurkishFoods-15) and private ones.

+ [ISIA Food-500](http://123.57.42.89/FoodComputing-Dataset/ISIA-Food500.html) [6]: Is a 500-category database released in 2020. In addition, all the food images (around 400K) were captured from Google, Baidu, and Bing search engines, including both Western and Eastern cuisines. Similar to other databases, they grouped all the food products into 11 major categories: meat, cereal, vegetables, seafood, fruits, dairy products, bakery products, fat products, pastries, drinks, and eggs.

+ [VIPER-FoodNet](https://lorenz.ecn.purdue.edu/~vfn/) [7]: Similar to Food-11, VIPER-FoodNet is an 82-category database where the authors considered the most frequent food products consumed in the United States from the What We Eat In America (WWEIA) database. All the images were obtained using the web scraping acquisition protocol, concretely from Google Images.


### <a name="systems">Food Recognition Systems<a>

The proposed food recognition systems are based on two state-of-the-art CNN architectures with outstanding performances in computer vision tasks, **Xception** [8] and [EfficientNetV2](https://github.com/sebastian-sz/efficientnet-v2-keras/tree/main) [9]. First, the Xception approach is inspired by Inception, replacing Inception modules with depthwise separable convolutions. Secondly, the EfficientNetV2 approach is an optimised model within the EfficientNet family of architectures, able to achieve better results with fewer parameters compared to other models in challenging databases like ImageNet [10]. 

In this study, we follow the same training approach considered in [11]: using a pre-trained model with ImageNet, the last fully-connected layers are replaced with the number of classes specific to each experiment. Then, all the weights from the model are fixed up to the fully-connected layers and re-trained for over 10 epochs. Subsequently, the entire network is trained again for 50 more epochs, choosing the best-performing model in terms of validation accuracy. We use the following features for all experiments: Adam optimiser based on binary cross-entropy using a learning rate of $10^-3$, and $\beta_1$ and $\beta_2$ of $0.9$ and $0.999$, respectively. In addition, training and testing are performed with an image size of 224x224. The experimental protocol was executed with the aid of an NVIDIA GeForce RTX 4090 GPU, utilising the Keras library. 

For reproducibility reasons, we adopt the same experimental protocol considered in the collected databases,  dividing them into development and test subsets following each corresponding subdivision. In addition, the development subset is also divided into train and validation subsets. However, three of the collected databases -FruitVeg81, UECFood-256, and Food-101- do not contain this division. In such cases, we employ a similar procedure as presented in [11]: around 80\% of the images comprise the development subset, with the train and validation subsets also distributed around 80\% and 20\% of the development subset, respectively. The remaining images correspond to the test subset (around 20\%). It is important to remark that no images are duplicated across the three subsets (train, validation, and test) in any of the seven databases. Top-1 (Top-1 Acc.) and Top-5 classification accuracy (Top-5 Acc.) are used as evaluation metrics.

**Three different scenarios** are considered for the intra-database evaluation of the AI4Food-NutritionDB database. Each scenario represents a different **level of granularity defined by the number of categories (19), subcategories (73), and final products (893)**. Regarding the whole AI4Food-NutritionDB database, category scenario performances show the best results, obtaining **77.74\% Top-1 Acc. and 97.78\% Top-5 Acc. for Xception, and 82.04\% Top-1 Acc. and 98.45\% Top-5 Acc. for EfficientNetV2**. However, the performance significantly drops as the granularity becomes finer for both architectures. For example, for the EfficientNetV2 architecture, the Top-1 Acc. decreases from 82.04\% to 77.66\% and 66.28\% for the subcategory (73 classes) and product (893 classes) analysis, respectively. This decrease is mainly due to the similarity in appearance among different subcategories (e.g., "White Meat" and "Red Meat"),  final products (e.g., "Pizza Carbonara" and "Pizza Pugliese"), or even the same food cooked in several manners (e.g., "Baked Salmon" and "Cured Salmon"). Regarding each specific dataset, the FruitVeg-81 dataset shows the best results in general for both deep learning architectures, classifying almost perfectly the different fruits and vegetables (over 98\% Top-1 and Top-5 Acc. for all categorisation scenarios). Contrarily, the VIPER-FoodNet dataset obtains the worst results at each categorisation scenario as images sometimes contain food products with mixed ingredients (e.g., different types of beans, meat, and pasta). Finally, in terms of the deep learning architecture, **EfficientNetV2 outperforms Xception in all scenarios (category, subcategory, product) of the AI4Food-NutritionDB for both Top-1 Acc. and Top-5 Acc. metrics**. These results highlight the potential of the state-of-the-art EfficientNetV2 architecture for the nutrition taxonomy proposed in the present article.


### <a name="references">References<a>

**[1]** Y. Kawano and K. Yanai. Automatic Expansion of a Food Image Dataset Leveraging Existing Categories with Domain Adaptation. In *Proc. of ECCV Workshop on Transferring and Adapting Source Knowledge in Computer Vision*, 2014.
    
**[2]** L. Bossard, M. Guillaumin, and L. Van Gool. Food-101 – Mining Discriminative Components with Random Forests. In *Proc. European Conference on Computer Vision*, pages 446–461, 2014.
    
**[3]** A. Singla, L. Yuan, and T. Ebrahimi. Food/Non-Food Image Classification and Food Categorization Using Pre-Trained GoogLeNet Model. In *Proc. of the International Workshop on Multimedia Assisted Dietary Management*, page 3–11, 2016.
    
**[4]** G. Waltner, M. Schwarz, S. Ladstätter, A. Weber, P. Luley, M. Lindschinger, I. Schmid, W. Scheitz, H. Bischof, and L. Paletta. Personalized Dietary Self-Management using Mobile Vision-based Assistance. In *Proc. of ICIAP Workshop on Multimedia Assisted Dietary Management*, 2017.
    
**[5]** E. Aguilar, M. Bolaños, and P. Radeva. Food Recognition using Fusion of Classifiers Based on CNNs. In *Proc. International Conference on Image Analysis and Processing*, pages 213–224, 2017.
    
**[6]** W. Min, L. Liu, Z. Wang, Z. Luo, X. Wei, X. Wei, and S. Jiang. ISIA Food-500: A Dataset for Large-Scale Food Recognition via Stacked Global-Local Attention Network. In *Proc. of the ACM International Conference on Multimedia*, 2020.
    
**[7]** R. Mao, J- He, Z. Shao, S. Kalyan Yarlagadda, and F. Zhu. Visual Aware Hierarchy Based Food Recognition. In *Proc. International Conference on Pattern Recognition*, pages 571–598, 2021.
    
**[8]** F. Chollet, Xception: Deep Learning with Depthwise Separable Convolutions. In *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2017.

**[9]** M. Tan M and Q. Le, Efficientnetv2: Smaller Models and Faster Training. In *Proc. International Conference on Machine Learning*, pp 10096–10106, 2021.

**[10]** A. Krizhevsky, I. Sutskever, and G. E. Hinton, Imagenet Classification with Deep Convolutional Neural Networks. *Advances in Neural Information Processing Systems*, vol. 25, 2012.

**[11]** R. Tolosana, S. Romero-Tapiador, R. Vera-Rodriguez, E. Gonzalez-Sosa, and J. Fierrez, DeepFakes Detection Across Generations: Analysis of Facial Regions, Fusion, and Performance Evaluation. *Engineering Applications of Artificial Intelligence*, vol. 110, p. 104673, 2022.


## <a name="download">Download AI4Food-NutritionDB Food Image Database<a>

In order to generate the AI4Food-NutritionDB food image database, all seven databases must be downloaded and placed in the correct path. For each of them, we provide some guidelines to download it. Note that 5 of them have a link, which will direct you to the corresponding database.
    
+ **UECFood-256** ([Right-click here to copy the URL](http://foodcam.mobi/dataset256.zip)).
+ **Food-101** ([Right-click here to copy the URL](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz)). 
+ **Food-11** ([Direct Download](https://www.kaggle.com/datasets/vermaavi/food11/download)): You must first sign in (or log in) to Kaggle. 
+ **FruitVeg-81** ([Download instructions](https://www.tugraz.at/index.php?id=22808)): follow the download instructions (you need to send an email) before downloading the database.
+ **MAFood-121** ([Direct Download](https://drive.google.com/uc?id=1lr3b7cPl_yoBK1N8thDJMYpPNCxDZFPD&export=download)).
+ **ISIA Food-500** ([Download link](http://123.57.42.89/Dataset_ict/ISIA_Food500_Dir/dataset/)): link to download website. ISIA_Food500.z01~10 and ISIA_Food500.zip files must be downloaded.
+ **VIPER-FoodNet** ([Right-click here to copy the URL](https://lorenz.ecn.purdue.edu/~vfn/vfn_1_0.zip)).

After downloading all 7 databases, place them in the same path as this GitHub repository (previously cloned or downloaded). Then, run [**database_generation.py**](https://github.com/BiDAlab/AI4Food-NutritionDB/blob/main/src/database_generation.py) and wait until the database is properly generated! Once this last step is done, the AI4Food-NutritionDB database will be available. As mentioned in the paper, the database is organised into 19 major categories, 73 subcategories, and 911 final products.
    
``` 
python database_generation.py
``` 

## <a name="protocol">Experimental Protocol<a>
We encourage the research community to use the AI4Food-NutritionDB  database in order to improve state-of-the-art food recognition systems in challenging scenarios. Therefore, we provide the [benchmark protocol](https://github.com/BiDAlab/AI4Food-NutritionDB/tree/main/experimental_protocol) for all the 3 experiments regarding its categorisation level:

| Categorisation Level|# Classes|# Train Images (%)|# Validation Images (%)| # Test Images (%)|  # Total Images |
|:-:|:-:|:-:|:-:|:-:|:-:|
| [Category](https://github.com/BiDAlab/AI4Food-NutritionDB/tree/main/experimental_protocol/category) | 19 | 336,750 (60.58%) | 63,715 (11.46%) | 155,390 (27.96%) | 555,855 |
| [Subcategory](https://github.com/BiDAlab/AI4Food-NutritionDB/tree/main/experimental_protocol/subcategory) | 73 | 337,152 (60.58%) | 63,809 (11.46%) | 155,610 (27.96%) | 556,571 |
| [Product](https://github.com/BiDAlab/AI4Food-NutritionDB/tree/main/experimental_protocol/product) | 840 | 330,061 (60.47%) | 62,528 (11.46%) | 153,271 (28.07%) | 545,860 |

**NOTE:** It is important to remark that the AI4Food-NutritionDB database must be previously downloaded before using the experimental protocol.

## <a name="models">How to use the Food Recognition Systems<a>
Two food recognition systems and a total of six different models are available in this repository. Each one represents a different architecture (Xception and EfficientNetV2) and a granularity defined by the number of categories, subcategories, and products. First, the environment must be installed and activated in order to use our food recognition models.
    
### <a name="install">Install & Requirements<a>
Please, follow the steps to install the environment properly on your computer:
    
1) **Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)**

2) **Update Conda** (from the Conda terminal):

``` 
conda update conda
conda upgrade --all
``` 
    
3) **Create a new environment from .yml file:**
    
``` 
conda env create -f FoodRecognition_env.yml
```     

4) **Activate the created environment:**
        
``` 
conda activate FoodRecognition_env
```

OR


1) **Install Dependencies:** Ensure you have Python installed on your system. You can install any necessary Python packages by running the following command:

```bash
pip install -r requirements.txt
```


### <a name="run">Run the Model<a>
After installing the configuration for the current environment, food recognition models can be executed from the terminal as follows:
    
``` 
python food_recognition_system.py --arch [architecture_name] --model [model_name] --img [img_parameter] --show [show_parameter]
```     
    

Where:

+ **--arch** are the different architectures: **xception** or **efficientnetv2**
+ **--model** are the different available models: **category**, **subcategory** or **product**
+ **--img** to run a single or multiple images: **single** or **multiple**
+ **--show** to show (or not) the image with its corresponding final prediction: **true** or **false**
        
For instance: 
    
``` 
python food_recognition_system.py --arch efficientnetv2 --model category --img multiple --show false
``` 
    
**NOTE**: **test images are located in the media/sample folder**. Place food images inside this folder or change the path of the **test_dir** parameter (in [food_recognition_system.py](https://github.com/BiDAlab/AI4Food-NutritionDB/blob/main/src/food_recognition_system.py) file).

Finally, the model will be prepared to recognise the food class of the various food images. Examples of both correct and wrong predictions are shown below:

<p align="center"><img src="./media/Food Prediction Samples.svg" alt="Food Prediction Samples" title="Food Prediction Samples"/></p>

    
## <a name="cite">Citation<a>
- **[AI4Food-NutritionDB_2024]** S. Romero-Tapiador,  R. Tolosana, A. Morales, J. Fierrez, R. Vera-Rodriguez, I. Espinosa-Salinas, E. Carrillo-de Santa Pau, A. Ramirez-de Molina and J. Ortega-Garcia, [**"Leveraging Automatic Personalised Nutrition: Food Image Recognition Benchmark and Dataset based on Nutrition Taxonomy"**](https://doi.org/10.1007/s11042-024-19161-4), Multimedia Tools and Applications, 2024.

  ```
  @ARTICLE{romerotapiador2024leveraging,
      title={Leveraging Automatic Personalised Nutrition: Food Image Recognition Benchmark and Dataset based on Nutrition Taxonomy}, 
      author={Sergio Romero-Tapiador and Ruben Tolosana and Aythami Morales and Isabel Espinosa-Salinas and Gala Freixer and Julian Fierrez and Ruben Vera-Rodriguez and Enrique Carrillo de Santa Pau and Ana Ramírez de Molina and Javier Ortega-Garcia},
      year={2024},
      doi = {10.1007/s11042-024-19161-4},
      journal={Multimedia Tools and Applications}
    }
  ```

   
All these articles are publicly available in the [publications](http://atvs.ii.uam.es/atvs/listpublications.do) section of the BiDA-Lab group webpage. Please, remember to reference the above articles on any work made public.
    
## <a name="contact">Contact<a>
  
For more information, please contact us via email at sergio.romero@uam.es or ruben.tolosana@uam.es.
