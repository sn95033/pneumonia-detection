
<a href="https://colab.research.google.com/github/sn95033/pneumonia-detection/blob/master/Pneumonia_32_32.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Childhood pneumonia detection 

<div>
<img src= "childrens_pneumonia.png"
           width=400"/>
</div
             
<br> Illustrative Examples of Chest X-Rays in Patients with Pneumonia.
The normal chest X-ray (left panel) depicts clear lungs without any areas of abnormal opacification in the image. Bacterial pneumonia (middle) typically exhibits a focal lobar consolidation, in this case in the right upper lobe (white arrows), whereas viral pneumonia (right) manifests with a more diffuse ‘‘interstitial’’ pattern in both lungs.
http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.

For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.

Acknowledgements
Data: https://data.mendeley.com/datasets/rscbjbr9sj/2

License: CC BY 4.0

Citation: http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5
   

# Analysis Workflow (OSEMN)

1. **Obtain and Pre-process**
    - [x] Import data
    - [x] Inspect the training, test and validation data
    - [x] Check data size, loking for class imbalance
    - [x] Determine if images are unique - Review of filenames or other identifiers.  In this project it turns out that some of the images are of the same child (same person ID),  which suggests it's possible multiple images were taken. This inclusion of images of the same child, probably leads to more weighting towards those images for training the model. On the other hand, it would seem to be a form of image augmentation.
    - [x] Initialize GoogleDrive, and read in the data locally for fastest runtime. Setup drives for writing data / images <br><br>

2. **Data Scoping**
     - [x] Remove "duplicate images"  -- i.e. different images of the same child
     - [x] Determine class imbalance
     - [x] Set up image size experiments
     - [x] Split into train test and validate data 
3.  **Create a Baseline Neural Network Model**
    - [x] Run the models
    - [x] Evaluate the model metrics  <br><br>

4. **Create a Convolutional Neural Network Model**
    - [x] Evaluate the model metrics
    - [x] Choose step sizes to be smaller, allowing for more epochs, adding layers or dropout of data<br><br>
    
5. **Revise data inputs if needed to improve quality indicators**
    - [x] By improving unbalanced datasets through image augmentation
    - [x] By optimizing number of layers and use of Dropout
    - [x] By speeding the model optimization by starting Transfer learning the initial weighting values from another model

    
6. **Write the Report**
    - [X] Explain key findings and recommended next steps


# Clearly Convolution Neural Networks are very powerful for classification of images

While fully dense connected layers already had good accuracy, particular for the Recall of True Positives (those with Pneumonia),  the CNNs dramatically improved the True Negatives (those that don't have Pneumonia).  

It can be hoped that these types of models can serve as "2nd opinions" or a second set of eyes,  to doctors and nurses in resource constrained countries, where human resources are a limiting factor in stopping the preventable disease of childhood pneumonia.



```
