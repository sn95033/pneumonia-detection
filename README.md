
<a href="https://colab.research.google.com/github/sn95033/pneumonia-detection/blob/master/Pneumonia_32_32.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Childhood pneumonia detection 

* Student name: Rebecca Mih
* Student pace: Part Time Online
* Scheduled project review date/time: June 27, 2020 2:00 PM PDT
* Instructor name: James Irving
* Blog post URL: https://github.com/sn95033/pneumonia-detection


* **Data Source:** 
   - References: <br>https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
<br> http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5 <br> <br>

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

3.  **Create a Baseline Neural Network Model**
    - [x] Split into train and test data 
    - [x] Run the model
    - [x] Review Quality indicators of the model <br><br>

4. **Create a Convolutional Neural Network Model**
    - [x] Compare the model quality
    - [x] Choose one or more models for grid searching <br><br>
    
5. **Revise data inputs if needed to improve quality indicators**
    - [x] By improving unbalanced datasets through image augmentation
    - [x] By speeding the model optimization by starting Transfer learning the initial weighting values from another model
    - [x] through use of subject matter knowledge <br><br>

    
6. **Write the Report**
    - [X] Explain key findings and recommended next steps


# Obtain and Pre-Process the Data

1. **Obtain and Pre-process**
    - [x] Import libraries
    - [x] Initialize GoogleDrive, and read in the data locally for fastest runtime. Setup drives for writing data / images 
    - [x] Inspect the training, test and validation data
    - [x] Check data size, review for class imbalance
    - [x] Determine if images are unique - Review of filenames or other identifiers. <br><br>
     In this project it turns out that some of the images are of the same child (same person ID),  i.e. multiple images were taken. This inadvertent inclusion of images of the same child, probably leads to more weighting towards those images for training the model. On the other hand, it's a form of image augmentation, since a child is hardly likely to sit still in exactly the same position, during x-ray . This data issue will be analysed in teh notebook, as to its impact on model accuracy.<br><br>
   

# 1. Importing Libraries



    


<style  type="text/css" >
</style><table id="T_2003a2a8_bf18_11ea_b554_0242ac1c0002" ><caption>Loaded Packages and Handles</caption><thead>    <tr>        <th class="col_heading level0 col0" >Handle</th>        <th class="col_heading level0 col1" >Package</th>        <th class="col_heading level0 col2" >Description</th>    </tr></thead><tbody>
                <tr>
                                <td id="T_2003a2a8_bf18_11ea_b554_0242ac1c0002row0_col0" class="data row0 col0" >dp</td>
                        <td id="T_2003a2a8_bf18_11ea_b554_0242ac1c0002row0_col1" class="data row0 col1" >IPython.display</td>
                        <td id="T_2003a2a8_bf18_11ea_b554_0242ac1c0002row0_col2" class="data row0 col2" >Display modules with helpful display and clearing commands.</td>
            </tr>
            <tr>
                                <td id="T_2003a2a8_bf18_11ea_b554_0242ac1c0002row1_col0" class="data row1 col0" >fs</td>
                        <td id="T_2003a2a8_bf18_11ea_b554_0242ac1c0002row1_col1" class="data row1 col1" >fsds_100719</td>
                        <td id="T_2003a2a8_bf18_11ea_b554_0242ac1c0002row1_col2" class="data row1 col2" >Custom data science bootcamp student package</td>
            </tr>
            <tr>
                                <td id="T_2003a2a8_bf18_11ea_b554_0242ac1c0002row2_col0" class="data row2 col0" >mpl</td>
                        <td id="T_2003a2a8_bf18_11ea_b554_0242ac1c0002row2_col1" class="data row2 col1" >matplotlib</td>
                        <td id="T_2003a2a8_bf18_11ea_b554_0242ac1c0002row2_col2" class="data row2 col2" >Matplotlib's base OOP module with formatting artists</td>
            </tr>
            <tr>
                                <td id="T_2003a2a8_bf18_11ea_b554_0242ac1c0002row3_col0" class="data row3 col0" >plt</td>
                        <td id="T_2003a2a8_bf18_11ea_b554_0242ac1c0002row3_col1" class="data row3 col1" >matplotlib.pyplot</td>
                        <td id="T_2003a2a8_bf18_11ea_b554_0242ac1c0002row3_col2" class="data row3 col2" >Matplotlib's matlab-like plotting module</td>
            </tr>
            <tr>
                                <td id="T_2003a2a8_bf18_11ea_b554_0242ac1c0002row4_col0" class="data row4 col0" >np</td>
                        <td id="T_2003a2a8_bf18_11ea_b554_0242ac1c0002row4_col1" class="data row4 col1" >numpy</td>
                        <td id="T_2003a2a8_bf18_11ea_b554_0242ac1c0002row4_col2" class="data row4 col2" >scientific computing with Python</td>
            </tr>
            <tr>
                                <td id="T_2003a2a8_bf18_11ea_b554_0242ac1c0002row5_col0" class="data row5 col0" >pd</td>
                        <td id="T_2003a2a8_bf18_11ea_b554_0242ac1c0002row5_col1" class="data row5 col1" >pandas</td>
                        <td id="T_2003a2a8_bf18_11ea_b554_0242ac1c0002row5_col2" class="data row5 col2" >High performance data structures and tools</td>
            </tr>
            <tr>
                                <td id="T_2003a2a8_bf18_11ea_b554_0242ac1c0002row6_col0" class="data row6 col0" >sns</td>
                        <td id="T_2003a2a8_bf18_11ea_b554_0242ac1c0002row6_col1" class="data row6 col1" >seaborn</td>
                        <td id="T_2003a2a8_bf18_11ea_b554_0242ac1c0002row6_col2" class="data row6 col2" >High-level data visualization library based on matplotlib</td>
            </tr>
    </tbody></table>



        <script type="text/javascript">
        window.PlotlyConfig = {MathJaxConfig: 'local'};
        if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
        if (typeof require !== 'undefined') {
        require.undef("plotly");
        requirejs.config({
            paths: {
                'plotly': ['https://cdn.plot.ly/plotly-latest.min']
            }
        });
        require(['plotly'], function(Plotly) {
            window._Plotly = Plotly;
        });
        }
        </script>
        


    [i] Pandas .iplot() method activated.
    

# Setup Google Drive and Read Data In Locally



```python

# 

from google.colab import drive
drive.mount('/gdrive',force_remount=True)
%cd /gdrive/My Drive/
%cd ~
%cd ..
```

    Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3aietf%3awg%3aoauth%3a2.0%3aoob&response_type=code&scope=email%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdocs.test%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive.photos.readonly%20https%3a%2f%2fwww.googleapis.com%2fauth%2fpeopleapi.readonly
    
    Enter your authorization code:
    ··········
    Mounted at /gdrive
    /gdrive/My Drive
    /root
    /
    


```python
# Procedure to load data from Google Drive into local memory in Colaboratory
# First set the path where the zipped data resides in Google Drive

# printout the current directory 
print(os.path.abspath(os.curdir))

#Set the path to the folder containing zipped data in Google drive
source_folder = r'/gdrive/My Drive/Datasets/'

#Search for a zip file using *.zip,  recursively search the folder.  
# Make sure there is only 1 zip file in the folder!
file = glob.glob(source_folder+'*.zip',recursive=True)[0]

# Print out the zipped file found.  Make sure it's the one you expected!
file
```

    /
    




    '/gdrive/My Drive/Datasets/17810_23812_bundle_archive.zip'




```python
# Read from the zip file into local memory

zip_path = file
!cp "{zip_path}" . 

!unzip -q 17810_23812_bundle_archive.zip  # Load the data into local memory
!rm 17810_23812_bundle_archive.zip

# Sometimes when you are running the notebook multiple times, the data will still be in memory
# And the text box will ask if you want to over-write the data.
# it's ok to select [N]one   which is the fastest
```


```python
#List the folder in local memory.  The folder name is chest_xray

import os,glob
os.path.abspath(os.curdir)
os.listdir()
```




    ['home',
     'run',
     'bin',
     'sys',
     'dev',
     'tmp',
     'root',
     'mnt',
     'lib',
     'var',
     'opt',
     'etc',
     'sbin',
     'lib64',
     'boot',
     'srv',
     'proc',
     'media',
     'usr',
     'chest_xray',
     'gdrive',
     '.dockerenv',
     'tools',
     'datalab',
     'swift',
     'dlib-19.18.0-cp36-cp36m-linux_x86_64.whl',
     'dlib-19.18.0-cp27-cp27mu-linux_x86_64.whl',
     'tensorflow-1.15.2',
     'content',
     'lib32']




```python
# Set the data directory to the folder chest_xray
import os,glob

data_dir = r'chest_xray/'
os.listdir(data_dir)

# We can see here that the train, test, and val folders are the current locataion,
# That is where the images are stored, according to the Kaggle documentation
```




    ['val', 'chest_xray', '__MACOSX', 'test', 'train']




```python
# Check you are still in the same data directory to the folder chest_xray
import os,glob

data_dir = r'chest_xray/'
os.listdir(data_dir)

# We can see here that the train, test, and val folders are the current locataion,
# That is where the images are stored, according to the Kaggle documentation
```




    ['val', 'chest_xray', '__MACOSX', 'test', 'train']




```python
# Set the directories to access train, validate, and test images

train_dir = data_dir + 'train/'

# Path to validation directory
val_dir = data_dir + 'val/'

# Path to test directory
test_dir = data_dir + 'test/'
```


```python
# First read in the train images
# Get the path to the non-pneumonia and pneumonia sub-directories
nop_train_dir = train_dir + 'NORMAL/'  #No Pnuemonia training directory
p_train_dir = train_dir + 'PNEUMONIA/' # Pneumonia training directory

# Get the list of all the images in the directory
nop_train_list = glob.glob(nop_train_dir+'*.jpeg')
p_train_list = glob.glob(p_train_dir+'*.jpeg')

# An empty list. We will insert the data into this list in (img_path, label) format
train_images = []

# Go through all the normal cases. The label for these cases will be 0
for img in nop_train_list:
    train_images.append((img,0))

# Go through all the pneumonia cases. The label for these cases will be 1
for img in p_train_list:
    train_images.append((img, 1))

# Get a pandas dataframe from the data we have in our list 
train_images = pd.DataFrame(train_images, columns=['image', 'label'],index=None)

# Shuffle the data 
#train_images = train_images.sample(frac=1.).reset_index(drop=True)

#######

# Read the test images in the same way
nop_test_dir = test_dir + 'NORMAL/'
p_test_dir = test_dir + 'PNEUMONIA/'

# Get the list of all the images
nop_test_list = glob.glob(nop_test_dir+'*.jpeg')
p_test_list = glob.glob(p_test_dir+'*.jpeg')

# An empty list. We will insert the data into this list in (img_path, label) format
test_images = []

# Go through all the normal cases. The label for these cases will be 0
for img in nop_test_list:
    test_images.append((img,0))

# Go through all the pneumonia cases. The label for these cases will be 1
for img in p_test_list:
    test_images.append((img,1))

# Get a pandas dataframe from the data we have in our list 
test_images = pd.DataFrame(test_images, columns=['image', 'label'],index=None)

# Look at the dataframe
display(train_images)
display(test_images)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>chest_xray/train/NORMAL/IM-0439-0001.jpeg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>chest_xray/train/NORMAL/NORMAL2-IM-0503-0001.jpeg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>chest_xray/train/NORMAL/IM-0702-0001.jpeg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>chest_xray/train/NORMAL/IM-0215-0001.jpeg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>chest_xray/train/NORMAL/NORMAL2-IM-0507-0001.jpeg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5211</th>
      <td>chest_xray/train/PNEUMONIA/person420_bacteria_...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5212</th>
      <td>chest_xray/train/PNEUMONIA/person546_virus_108...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5213</th>
      <td>chest_xray/train/PNEUMONIA/person1556_bacteria...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5214</th>
      <td>chest_xray/train/PNEUMONIA/person348_virus_720...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5215</th>
      <td>chest_xray/train/PNEUMONIA/person1237_bacteria...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5216 rows × 2 columns</p>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>chest_xray/test/NORMAL/NORMAL2-IM-0354-0001.jpeg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>chest_xray/test/NORMAL/IM-0011-0001.jpeg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>chest_xray/test/NORMAL/NORMAL2-IM-0150-0001.jpeg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>chest_xray/test/NORMAL/NORMAL2-IM-0321-0001.jpeg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>chest_xray/test/NORMAL/NORMAL2-IM-0339-0001.jpeg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>619</th>
      <td>chest_xray/test/PNEUMONIA/person82_bacteria_40...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>620</th>
      <td>chest_xray/test/PNEUMONIA/person161_bacteria_7...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>621</th>
      <td>chest_xray/test/PNEUMONIA/person122_bacteria_5...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>622</th>
      <td>chest_xray/test/PNEUMONIA/person140_bacteria_6...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>623</th>
      <td>chest_xray/test/PNEUMONIA/person88_bacteria_43...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>624 rows × 2 columns</p>
</div>


# Inspect the Image Data for Class Balance and Uniqueness


```python
# Get the # of each class
num_train_cases = train_images['label'].value_counts()
print('Number of Training Cases:', '\n', num_train_cases, '\n')

# Plot the results 

sns.barplot(x=num_train_cases.index, y= num_train_cases.values, palette= 'YlGnBu')
plt.title('Number of Images in Training Set', fontsize=14)
plt.xlabel('Types', fontsize=14)
plt.ylabel('Number of Images', fontsize=14)
plt.xticks(range(len(num_train_cases.index)), ['No Pneumonia (0)', 'Pneumonia (1)'])
#plt.tight_layout()
plt.show();
%cd /gdrive/My Drive/'Output'/
plt.savefig("Training Set Image Data.png");

num_test_cases= test_images['label'].value_counts()
print('\n')
print('Number of Test Cases: ', '\n', num_test_cases, '\n')
sns.barplot(x=num_test_cases.index, y= num_test_cases.values, palette= 'YlGnBu')
plt.title('Number of Images in the Test Set', fontsize=14)
plt.xlabel('Types', fontsize=14)
plt.ylabel('Number of Images', fontsize=14)
plt.xticks(range(len(num_test_cases.index)), ['No Pneumonia (0)', 'Pneumonia (1)'])
plt.show();
plt.savefig("Test Set Image Data.png");
```

    Number of Training Cases: 
     1    3875
    0    1341
    Name: label, dtype: int64 
    
    


![png](output_21_1.png)


    /gdrive/My Drive/Output
    
    
    Number of Test Cases:  
     1    390
    0    234
    Name: label, dtype: int64 
    
    


![png](output_21_3.png)



    <Figure size 432x288 with 0 Axes>


## There is a 3:1 Class imbalance which may need to be addressed through image augmentation


```python
# Reset to the appropriate folder
#source_folder = r'/gdrive/My Drive/Datasets/'
%cd /gdrive/My Drive/chest_xray/
```

    /gdrive/My Drive/chest_xray
    


```python
# Get few samples for both the classes
pneumonia_samples = (train_images[train_images['label']==1]['image'].iloc[:5:1]).tolist()
normal_samples = (train_images[train_images['label']==0]['image'].iloc[:5:1]).tolist()

# Concat the data in a single list and del the above two list
train_samples = pneumonia_samples + normal_samples
del pneumonia_samples, normal_samples

# Plot the data 
f, ax = plt.subplots(2,5, figsize=(30,10))
for i in range(10):
    img = imread(train_samples[i])
    ax[i//5, i%5].imshow(img, cmap='gray')
    if i<5:
        ax[i//5, i%5].set_title("Pneumonia")
    else:
        ax[i//5, i%5].set_title("Normal")
    ax[i//5, i%5].axis('off')
    ax[i//5, i%5].set_aspect('auto')
plt.show();
```


![png](output_24_0.png)





```python
# change directory to the output folder and save the picture above
%cd /gdrive/My Drive/'Output'/
plt.savefig("Typical images of pneumonia and normal cases.png")
```

    /gdrive/My Drive/Output
    


    <Figure size 432x288 with 0 Axes>



```python
%cd /gdrive/My Drive/chest_xray
```

    /gdrive/My Drive/chest_xray
    


```python
#Load the data as numbers by reading the RBG of the file
import cv2,glob,os
import itertools

# create a list of filenames for training and testing
train_img_filenames = [*p_train_list,*nop_train_list]
test_img_filenames = [*p_test_list,*nop_test_list]
print('Number of train filenames = ',len(train_img_filenames))
print('Number of test filenames = ', len(test_img_filenames))
train_img_filenames[:10]
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-7-7aca84ab9b14> in <module>()
          4 
          5 # create a list of filenames for training and testing
    ----> 6 train_img_filenames = [*p_train_list,*nop_train_list]
          7 test_img_filenames = [*p_test_list,*nop_test_list]
          8 print('Number of train filenames = ',len(train_img_filenames))
    

    NameError: name 'p_train_list' is not defined



```python
import cv2
def load_image_cv2(filename, RGB=True):
  """Function to load an image and convert the output to numbers
  Inputs: filename of the image to be read, string
  Pass RGB to the cv2 library to convert the image from BGR to RGB
  elses converts to gray scale """
  import cv2

  IMG = imread(filename)

  if RGB: cmap = cv2.COLOR_BGR2RGB
  else: cmap=cv2.COLOR_BGR2GRAY
  return cv2.cvtColor(IMG,cmap)
```


```python
## Load in and display image.
IMG = load_image_cv2(train_img_filenames[0],RGB=True)

## Even if you import as grayscale, must tell plt to use gray cmap
fig,ax= plt.subplots(ncols=2,figsize=(12,5))
ax[0].imshow(IMG)
ax[1].imshow(IMG,cmap='gray')
print(IMG.shape)
```

    (759, 1098, 3)
    


![png](output_30_1.png)



```python
## Using seaborn color palette with imshow
from matplotlib.colors import ListedColormap
cmap = ListedColormap(sns.color_palette('gray',n_colors=25))

## Remove axes labels https://stackoverflow.com/a/2176591
[(a.get_xaxis().set_visible(False), a.get_yaxis().set_visible(False)) for a in ax]
plt.imshow(IMG,cmap=cmap);
```


![png](output_31_0.png)



```python

```


```python
# Setup a directory to save images
*a,_=test_dir.split('/')
save_dir = '/'.join(a)
save_dir
```




    'chest_xray/test'




```python
cv2.imwrite(save_dir+'sample_original.jpg',IMG)
```




    True




```python
## RESIZING IMAGES
print(IMG.shape)
size_32_32 = cv2.resize(IMG,(32,32))
plt.imshow(size_32_32,cmap=cmap);
```

    (759, 1098, 3)
    


![png](output_35_1.png)



```python
cv2.imwrite(save_dir+'sample_32_32.jpg',size_32_32)
```




    True




```python
## RESIZING IMAGES
print(IMG.shape)
size_64_64 = cv2.resize(IMG,(64,64))
plt.imshow(size_64_64,cmap=cmap);
```

    (759, 1098, 3)
    


![png](output_37_1.png)



```python
cv2.imwrite(save_dir+'sample_64_64.jpg',size_64_64)
```




    True




```python
## RESIZING IMAGES
print(IMG.shape)
size_256_256 = cv2.resize(IMG,(256,256))
plt.imshow(size_256_256,cmap=cmap);
```

    (759, 1098, 3)
    


![png](output_39_1.png)



```python
cv2.imwrite(save_dir+'sample_256_256.jpg',size_256_256)
```




    True



## Load the Data
### Translate the image data into an Array
### Redfine the Train, Test, Validate splits as needed
### Encode the labels as Categorical
### Create X_train, X_test,X_val, y_train, y_test, y_val
### Store values in case internet is slow (using Pickle)



```python
# Translate all images to data, and change image size to reduce computation time
img_size = (32,32,3)
from PIL import Image
from keras.preprocessing import image

from imageio import imread
from skimage.transform import resize
import cv2
from tqdm import tqdm

# defining a function to read images and translate into data array
def read_img(img_path,target_size=(32, 32, 3)):
  '''Image reader which translates images to numbers
  Inputs:  img_path - the folder or directory where images are stored
           target_size = the new image size, usually reduced for faster
           computation time. Default is (64,64,3)   '''
  img = image.load_img(img_path, target_size=target_size)
  img_array = img_to_array(img)
 
  return img_array

def redefine_splits(p_train_list, nop_train_list, p_test_list, nop_test_list, 
                    img_size=(32,32,3), val_size=0.1):
  '''Redefine the splits by reading the original data from Train and test folders
  then running SKLearn train-test-split algorithm to create a validation set
  to be used during training.  Y labels defined as: 0 = no pneumonia,
  1 = pneumonia present

  Inputs: filenames of each imagefile in the separate folders by class for the 
  train and test data

  Outputs: X_train, X_test, X_val, y_train, y_test, y_val   '''

  # reading the images as image data, creating labels based on the directory
  # the data is in
  train_img = []
  train_label = []

  # pneumonia = 1
  for img_path in tqdm(p_train_list):

    train_img.append(read_img(img_path, target_size = img_size))
    train_label.append(1)

  for img_path in tqdm(nop_train_list):
    train_img.append(read_img(img_path, target_size = img_size))
    train_label.append(0)
      
  print('\n',pd.Series(train_label).value_counts())

  test_img = []
  test_label = []

  for img_path in tqdm(p_test_list):
      test_img.append(read_img(img_path, target_size= img_size))
      test_label.append(1)

  for img_path in tqdm(nop_test_list):
      test_img.append(read_img(img_path, target_size= img_size))
      test_label.append(0)

  print('\n',pd.Series(test_label).value_counts())

# Transform the train and test image data into numpy arrays
# Transform the train and test labels into categoricals

  from sklearn.model_selection import train_test_split
  X = np.array(train_img, np.float32)
  y = to_categorical(np.array(train_label))
  
  X_test = np.array(test_img, np.float32)
  y_test = to_categorical(np.array(test_label))

  X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.1)

  print('New Split Definitions:')
  print(f"X_train={len(X_train)}, X_test={len(X_test)}, X_val={len(X_val)}")
  
  return X_train, X_test, X_val, y_train, y_test,y_val 
```


```python
# Now execute the split redefinition
# Create a validation set that has 10% of the data 
X_train,X_test,X_val,y_train,y_test,y_val = redefine_splits(p_train_list, nop_train_list,
                                                                p_test_list, nop_test_list,
                                            val_size=0.1,img_size=(32,32,3))#(64,64,3))

train_test_val_vars = [X_train,X_test,X_val,y_train,y_test,y_val]

```

    100%|██████████| 3875/3875 [33:28<00:00,  1.93it/s]
    100%|██████████| 1341/1341 [12:12<00:00,  1.83it/s]
      0%|          | 0/390 [00:00<?, ?it/s]

    
     1    3875
    0    1341
    dtype: int64
    

    100%|██████████| 390/390 [03:40<00:00,  1.77it/s]
    100%|██████████| 234/234 [02:14<00:00,  1.74it/s]

    
     1    390
    0    234
    dtype: int64
    New Split Definitions:
    X_train=4694, X_test=624, X_val=522
    

    
    


```python
# Check which directory you are in
#import os,glob

data_dir = r'/gdrive/My Drive/Output/'
os.listdir(data_dir)
```




    ['Second Baseline-64-64-64.jpg',
     'Second Baseline-64-64-64-b.jpg',
     'Second Baseline-64-64-32.jpg',
     'Second Baseline-64-64-32-b.jpg',
     'Baseline-64-64-32.jpg',
     'Baseline-64-64-16.jpg',
     'Second Baseline-64-64-16.jpg',
     'Baseline-64-64-16_early.jpg',
     'best_model.h5',
     'Second Baseline-64-64-16-early.jpg',
     'Stored_Values',
     'Training Set Image Data.png',
     'Test Set Image Data.png',
     'Typical images of pneumonia and normal cases.png']




```python
# Because it takes so long to read the values and do the split, pickle the initial datasets
#%cd /gdrive/My Drive/
import pickle

with open('/gdrive/My Drive/Output/Stored_Values/Split_Recovery/orig_data_32_32', 'wb') as f:
  pickle.dump([X_train, X_test, X_val, y_train, y_test, y_val], f)

```


```python
# Test reading the values from pickle
import pickle
testvar = []
with open('/gdrive/My Drive/Output/Stored_Values/Split_Recovery/orig_data_32_32', 'rb') as f:
  testvar = pickle.load(f)
X_train = testvar[0]
X_test = testvar[1]
X_val = testvar[2]
y_train = testvar[3]
y_test = testvar[4]
y_val = testvar[5]
```

# Create an Optional DataGenerator Using Keras built-in Image Augmentation



# Transform the shape of the data into a 2D vector which can be read by Keras


```python
# First understand the shape of the data and re-shape as needed
print(X_train.shape, X_test.shape)
```

    (4694, 32, 32, 3) (624, 32, 32, 3)
    


```python
# X_train is 4694 images, with each image containing xpixels* ypixels*3  pixels of data
# This is in a 4 dimensional tensor
# In order to process the data, we have to re-shape the data to a two dimensional array
total_num_pixels = X_train[0].shape
total_num_pixels    # we need to move all the data into a 2-D array
```




    (32, 32, 3)




```python
num_images = X_train[0]
reshape_value = num_images.shape[0] *num_images.shape[1]*num_images.shape[2]
reshape_value
```




    3072




```python
X_train_img  = X_train.reshape(X_train.shape[0],reshape_value).astype('float32')/255
X_test_img  = X_test.reshape(X_test.shape[0],reshape_value).astype('float32')/255
X_val_img  = X_val.reshape(X_val.shape[0],reshape_value).astype('float32')/255

print(X_train_img.shape, X_test_img.shape, X_val_img.shape)
```

    (4694, 3072) (624, 3072) (522, 3072)
    


```python
# Create a variable to store the shape of the data
SHAPES = total_num_pixels
```


```python
# Check that the arrays created are correct by re-constructing an image
from PIL import Image
from keras.preprocessing import image

# Check one image
i = np.random.choice(range(len(y_train)))
# display(y_train[i])
display(image.array_to_img(X_train_img[i].reshape(total_num_pixels)))
```


![png](output_54_0.png)


# Create Evaluation Metrics



```python
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_keras_history(history,figsize=(10,4),subplot_kws={}):
    
    if hasattr(history,'history'):
        history=history.history
    try:
        acc_keys = list(filter(lambda x: 'acc' in x,history.keys()))
    except:
        print('No acc keys found')
        pass
    try:
        loss_keys = list(filter(lambda x: 'loss' in x,history.keys()))
    except:
        print('No loss keys found')

        pass
    
    plot_me = pd.DataFrame(history)
    
    fig,axes=plt.subplots(ncols=2,figsize=figsize,**subplot_kws)
    axes = axes.flatten()

    y_labels= ['Accuracy','Loss']
    for a, metric in enumerate([acc_keys,loss_keys]):
        for i in range(len(metric)):
            ax = pd.Series(history[metric[i]],
                        name=metric[i]).plot(ax=axes[a],label=metric[i])
    [ax.legend() for ax in axes]
    [ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True)) for ax in axes]
    [ax.set(xlabel='Epochs') for ax in axes]
    plt.suptitle('Model Training Results',y=1.01)
    plt.tight_layout()
    plt.show()
    return plt.gcf()


def plot_confusion_matrix(conf_matrix, classes = None, normalize=True,
                          title='Confusion Matrix', cmap="Blues",
                          print_raw_matrix=False,
                          fig_size=(4,4)):
    """Check if Normalization Option is Set to True. 
    If so, normalize the raw confusion matrix before visualizing
    #Other code should be equivalent to your previous function.
    Note: Taken from bs_ds and modified
    - Can pass a tuple of (y_true,y_pred) instead of conf matrix.
    """
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt
    import sklearn.metrics as metrics
    
    ## make confusion matrix if given tuple of y_true,y_pred
    if isinstance(conf_matrix, tuple):
        y_true = conf_matrix[0].copy()
        y_pred = conf_matrix[1].copy()
        
        if y_true.ndim>1:
            y_true = y_true.argmax(axis=1)
        if y_pred.ndim>1:
            y_pred = y_pred.argmax(axis=1)
        cm = metrics.confusion_matrix(y_true,y_pred)
    else:
        cm = conf_matrix
        
    ## Generate integer labels for classes
    if classes is None:
        classes = list(range(len(cm)))  
        
    ## Normalize data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt='.2f'
    else:
        fmt= 'd'
        
        
    fontDict = {
        'title':{
            'fontsize':16,
            'fontweight':'semibold',
            'ha':'center',
            },
        'xlabel':{
            'fontsize':14,
            'fontweight':'normal',
            },
        'ylabel':{
            'fontsize':14,
            'fontweight':'normal',
            },
        'xtick_labels':{
            'fontsize':10,
            'fontweight':'normal',
    #             'rotation':45,
            'ha':'right',
            },
        'ytick_labels':{
            'fontsize':10,
            'fontweight':'normal',
            'rotation':0,
            'ha':'right',
            },
        'data_labels':{
            'ha':'center',
            'fontweight':'semibold',

        }
    }

    # Create plot
    fig,ax = plt.subplots(figsize=fig_size)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,**fontDict['title'])
    plt.colorbar()

    tick_marks = classes#np.arange(len(classes))


    plt.xticks(tick_marks, classes, **fontDict['xtick_labels'])
    plt.yticks(tick_marks, classes,**fontDict['ytick_labels'])

    # Determine threshold for b/w text
    thresh = cm.max() / 2.

    # fig,ax = plt.subplots()
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 color='darkgray',**fontDict['data_labels']) #color="white" if cm[i, j] > thresh else "black"

    plt.tight_layout()
    plt.ylabel('True label',**fontDict['ylabel'])
    plt.xlabel('Predicted label',**fontDict['xlabel'])

    if print_raw_matrix:
        print_title = 'Raw Confusion Matrix Counts:'
        print('\n',print_title)
        print(conf_matrix);


    fig = plt.gcf()
    return fig


    
```


```python
#from mlxtend.plotting import plot_confusion_matrix
#from sklearn.metrics import confusion_matrix

def evaluate_model(y_true, y_pred,history=None):
    from sklearn import metrics
    if y_true.ndim>1:
        y_true = y_true.argmax(axis=1)
    if y_pred.ndim>1:
        y_pred = y_pred.argmax(axis=1)   
        
    if history is not None:
        plot_keras_history(history)
        plt.show();
        plt.close();

    num_dashes=20
    print('\n')
    print('---'*num_dashes)
    print('\tCLASSIFICATION REPORT:')
    print('---'*num_dashes)

    print(metrics.classification_report(y_true,y_pred))
    
    fig = plot_confusion_matrix((y_true,y_pred))
    plt.show()
    
```

# Start GPUs and RAM



```python

```


```python
#https://colab.research.google.com/notebooks/pro.ipynb
gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Select the Runtime → "Change runtime type" menu to enable a GPU accelerator, ')
  print('and then re-execute this cell.')
else:
  print(gpu_info)
```

    Sun Jul  5 23:33:52 2020       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 450.36.06    Driver Version: 418.67       CUDA Version: 10.1     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla P100-PCIE...  Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   37C    P0    25W / 250W |      0MiB / 16280MiB |      0%      Default |
    |                               |                      |                 ERR! |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+
    


```python
from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

if ram_gb < 20:
  print('To enable a high-RAM runtime, select the Runtime → "Change runtime type"')
  print('menu, and then select High-RAM in the Runtime shape dropdown. Then, ')
  print('re-execute this cell.')
else:
  print('You are using a high-RAM runtime!')
```

    Your runtime has 27.4 gigabytes of available RAM
    
    You are using a high-RAM runtime!
    

# Build Baseline Model


```python
from keras.models import Sequential
from keras.layers import Dense

def make_baseline_model():
    model = Sequential()
    model.add(Dense(32,activation='relu',input_shape=(X_train_img.shape[1],)))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
```


```python
model = make_baseline_model()
model.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 32)                98336     
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 66        
    =================================================================
    Total params: 98,402
    Trainable params: 98,402
    Non-trainable params: 0
    _________________________________________________________________
    


```python


```


```python
EPOCHS = 1000
BATCH_SIZE = 32
PATIENCE=10

CALLBACKS = [EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=1),
             ModelCheckpoint(filepath='best_model_32_32.h5', monitor='val_loss', save_best_only=True, verbose=1)]
             
history = model.fit(X_train_img, y_train, epochs=EPOCHS, callbacks=CALLBACKS, # Early stopping
                    batch_size=BATCH_SIZE, validation_data = (X_val_img, y_val))
```

    Train on 4694 samples, validate on 522 samples
    Epoch 1/1000
    4694/4694 [==============================] - 2s 458us/step - loss: 0.4136 - accuracy: 0.8074 - val_loss: 0.2944 - val_accuracy: 0.8448
    
    Epoch 00001: val_loss improved from inf to 0.29439, saving model to best_model_32_32.h5
    Epoch 2/1000
    4694/4694 [==============================] - 0s 94us/step - loss: 0.2859 - accuracy: 0.8963 - val_loss: 0.2278 - val_accuracy: 0.9234
    
    Epoch 00002: val_loss improved from 0.29439 to 0.22784, saving model to best_model_32_32.h5
    Epoch 3/1000
    4694/4694 [==============================] - 0s 95us/step - loss: 0.2511 - accuracy: 0.9210 - val_loss: 0.2671 - val_accuracy: 0.8352
    
    Epoch 00003: val_loss did not improve from 0.22784
    Epoch 4/1000
    4694/4694 [==============================] - 0s 85us/step - loss: 0.2151 - accuracy: 0.9350 - val_loss: 0.2235 - val_accuracy: 0.8966
    
    Epoch 00004: val_loss improved from 0.22784 to 0.22351, saving model to best_model_32_32.h5
    Epoch 5/1000
    4694/4694 [==============================] - 0s 89us/step - loss: 0.1907 - accuracy: 0.9450 - val_loss: 0.1566 - val_accuracy: 0.9598
    
    Epoch 00005: val_loss improved from 0.22351 to 0.15662, saving model to best_model_32_32.h5
    Epoch 6/1000
    4694/4694 [==============================] - 0s 85us/step - loss: 0.1855 - accuracy: 0.9442 - val_loss: 0.1952 - val_accuracy: 0.9157
    
    Epoch 00006: val_loss did not improve from 0.15662
    Epoch 7/1000
    4694/4694 [==============================] - 0s 90us/step - loss: 0.1727 - accuracy: 0.9467 - val_loss: 0.1347 - val_accuracy: 0.9617
    
    Epoch 00007: val_loss improved from 0.15662 to 0.13466, saving model to best_model_32_32.h5
    Epoch 8/1000
    4694/4694 [==============================] - 0s 95us/step - loss: 0.1740 - accuracy: 0.9435 - val_loss: 0.1527 - val_accuracy: 0.9579
    
    Epoch 00008: val_loss did not improve from 0.13466
    Epoch 9/1000
    4694/4694 [==============================] - 0s 88us/step - loss: 0.1540 - accuracy: 0.9470 - val_loss: 0.1265 - val_accuracy: 0.9655
    
    Epoch 00009: val_loss improved from 0.13466 to 0.12654, saving model to best_model_32_32.h5
    Epoch 10/1000
    4694/4694 [==============================] - 0s 94us/step - loss: 0.1598 - accuracy: 0.9467 - val_loss: 0.1031 - val_accuracy: 0.9693
    
    Epoch 00010: val_loss improved from 0.12654 to 0.10315, saving model to best_model_32_32.h5
    Epoch 11/1000
    4694/4694 [==============================] - 0s 97us/step - loss: 0.1457 - accuracy: 0.9472 - val_loss: 0.1070 - val_accuracy: 0.9674
    
    Epoch 00011: val_loss did not improve from 0.10315
    Epoch 12/1000
    4694/4694 [==============================] - 0s 94us/step - loss: 0.1375 - accuracy: 0.9542 - val_loss: 0.1238 - val_accuracy: 0.9598
    
    Epoch 00012: val_loss did not improve from 0.10315
    Epoch 13/1000
    4694/4694 [==============================] - 0s 89us/step - loss: 0.1321 - accuracy: 0.9553 - val_loss: 0.1236 - val_accuracy: 0.9502
    
    Epoch 00013: val_loss did not improve from 0.10315
    Epoch 14/1000
    4694/4694 [==============================] - 0s 84us/step - loss: 0.1321 - accuracy: 0.9540 - val_loss: 0.1061 - val_accuracy: 0.9636
    
    Epoch 00014: val_loss did not improve from 0.10315
    Epoch 15/1000
    4694/4694 [==============================] - 0s 90us/step - loss: 0.1398 - accuracy: 0.9525 - val_loss: 0.0939 - val_accuracy: 0.9770
    
    Epoch 00015: val_loss improved from 0.10315 to 0.09392, saving model to best_model_32_32.h5
    Epoch 16/1000
    4694/4694 [==============================] - 0s 90us/step - loss: 0.1222 - accuracy: 0.9580 - val_loss: 0.1109 - val_accuracy: 0.9636
    
    Epoch 00016: val_loss did not improve from 0.09392
    Epoch 17/1000
    4694/4694 [==============================] - 0s 90us/step - loss: 0.1258 - accuracy: 0.9561 - val_loss: 0.0902 - val_accuracy: 0.9674
    
    Epoch 00017: val_loss improved from 0.09392 to 0.09021, saving model to best_model_32_32.h5
    Epoch 18/1000
    4694/4694 [==============================] - 0s 86us/step - loss: 0.1300 - accuracy: 0.9527 - val_loss: 0.0926 - val_accuracy: 0.9770
    
    Epoch 00018: val_loss did not improve from 0.09021
    Epoch 19/1000
    4694/4694 [==============================] - 0s 91us/step - loss: 0.1099 - accuracy: 0.9617 - val_loss: 0.1000 - val_accuracy: 0.9693
    
    Epoch 00019: val_loss did not improve from 0.09021
    Epoch 20/1000
    4694/4694 [==============================] - 0s 86us/step - loss: 0.1186 - accuracy: 0.9578 - val_loss: 0.0906 - val_accuracy: 0.9770
    
    Epoch 00020: val_loss did not improve from 0.09021
    Epoch 21/1000
    4694/4694 [==============================] - 0s 89us/step - loss: 0.1119 - accuracy: 0.9602 - val_loss: 0.0802 - val_accuracy: 0.9713
    
    Epoch 00021: val_loss improved from 0.09021 to 0.08020, saving model to best_model_32_32.h5
    Epoch 22/1000
    4694/4694 [==============================] - 0s 89us/step - loss: 0.1210 - accuracy: 0.9580 - val_loss: 0.0816 - val_accuracy: 0.9770
    
    Epoch 00022: val_loss did not improve from 0.08020
    Epoch 23/1000
    4694/4694 [==============================] - 0s 94us/step - loss: 0.1141 - accuracy: 0.9585 - val_loss: 0.0883 - val_accuracy: 0.9617
    
    Epoch 00023: val_loss did not improve from 0.08020
    Epoch 24/1000
    4694/4694 [==============================] - 0s 92us/step - loss: 0.1041 - accuracy: 0.9617 - val_loss: 0.0886 - val_accuracy: 0.9751
    
    Epoch 00024: val_loss did not improve from 0.08020
    Epoch 25/1000
    4694/4694 [==============================] - 0s 93us/step - loss: 0.1003 - accuracy: 0.9644 - val_loss: 0.0757 - val_accuracy: 0.9770
    
    Epoch 00025: val_loss improved from 0.08020 to 0.07574, saving model to best_model_32_32.h5
    Epoch 26/1000
    4694/4694 [==============================] - 0s 84us/step - loss: 0.1021 - accuracy: 0.9631 - val_loss: 0.0718 - val_accuracy: 0.9732
    
    Epoch 00026: val_loss improved from 0.07574 to 0.07179, saving model to best_model_32_32.h5
    Epoch 27/1000
    4694/4694 [==============================] - 0s 87us/step - loss: 0.0992 - accuracy: 0.9663 - val_loss: 0.0757 - val_accuracy: 0.9693
    
    Epoch 00027: val_loss did not improve from 0.07179
    Epoch 28/1000
    4694/4694 [==============================] - 0s 90us/step - loss: 0.0973 - accuracy: 0.9659 - val_loss: 0.0808 - val_accuracy: 0.9789
    
    Epoch 00028: val_loss did not improve from 0.07179
    Epoch 29/1000
    4694/4694 [==============================] - 0s 93us/step - loss: 0.0935 - accuracy: 0.9678 - val_loss: 0.0699 - val_accuracy: 0.9789
    
    Epoch 00029: val_loss improved from 0.07179 to 0.06994, saving model to best_model_32_32.h5
    Epoch 30/1000
    4694/4694 [==============================] - 0s 93us/step - loss: 0.0967 - accuracy: 0.9638 - val_loss: 0.0785 - val_accuracy: 0.9693
    
    Epoch 00030: val_loss did not improve from 0.06994
    Epoch 31/1000
    4694/4694 [==============================] - 0s 88us/step - loss: 0.1115 - accuracy: 0.9593 - val_loss: 0.0999 - val_accuracy: 0.9540
    
    Epoch 00031: val_loss did not improve from 0.06994
    Epoch 32/1000
    4694/4694 [==============================] - 0s 91us/step - loss: 0.1091 - accuracy: 0.9595 - val_loss: 0.0822 - val_accuracy: 0.9617
    
    Epoch 00032: val_loss did not improve from 0.06994
    Epoch 33/1000
    4694/4694 [==============================] - 0s 84us/step - loss: 0.1038 - accuracy: 0.9631 - val_loss: 0.0766 - val_accuracy: 0.9789
    
    Epoch 00033: val_loss did not improve from 0.06994
    Epoch 34/1000
    4694/4694 [==============================] - 0s 86us/step - loss: 0.0982 - accuracy: 0.9642 - val_loss: 0.0717 - val_accuracy: 0.9693
    
    Epoch 00034: val_loss did not improve from 0.06994
    Epoch 35/1000
    4694/4694 [==============================] - 0s 85us/step - loss: 0.0966 - accuracy: 0.9661 - val_loss: 0.0903 - val_accuracy: 0.9713
    
    Epoch 00035: val_loss did not improve from 0.06994
    Epoch 36/1000
    4694/4694 [==============================] - 0s 88us/step - loss: 0.0974 - accuracy: 0.9640 - val_loss: 0.0860 - val_accuracy: 0.9598
    
    Epoch 00036: val_loss did not improve from 0.06994
    Epoch 37/1000
    4694/4694 [==============================] - 0s 85us/step - loss: 0.0889 - accuracy: 0.9683 - val_loss: 0.1145 - val_accuracy: 0.9579
    
    Epoch 00037: val_loss did not improve from 0.06994
    Epoch 38/1000
    4694/4694 [==============================] - 0s 86us/step - loss: 0.0915 - accuracy: 0.9676 - val_loss: 0.0760 - val_accuracy: 0.9674
    
    Epoch 00038: val_loss did not improve from 0.06994
    Epoch 39/1000
    4694/4694 [==============================] - 0s 95us/step - loss: 0.0909 - accuracy: 0.9672 - val_loss: 0.0810 - val_accuracy: 0.9674
    
    Epoch 00039: val_loss did not improve from 0.06994
    Epoch 00039: early stopping
    


```python

```


```python

```


```python
y_hat_test = model.predict(X_test_img)
evaluate_model(y_test, y_hat_test,history)
```


![png](output_69_0.png)



    <Figure size 432x288 with 0 Axes>


    
    
    ------------------------------------------------------------
    	CLASSIFICATION REPORT:
    ------------------------------------------------------------
                  precision    recall  f1-score   support
    
               0       0.96      0.31      0.47       234
               1       0.70      0.99      0.82       390
    
        accuracy                           0.74       624
       macro avg       0.83      0.65      0.65       624
    weighted avg       0.80      0.74      0.69       624
    
    


![png](output_69_3.png)



```python
plt.savefig('Baseline-32-32-32_early.jpg')
```


    <Figure size 432x288 with 0 Axes>



```python

```


```python
def make_2nd_baseline_model():
    model = Sequential()
    model.add(Dense(64,activation='relu',input_shape=(X_train_img.shape[1],)))
    model.add(Dense(7, activation='relu'))
    model.add(Dense(5, activation='relu'))
    #model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model
```


```python
EPOCHS = 1000
BATCH_SIZE = 32
PATIENCE=10

CALLBACKS = [EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=1),
             ModelCheckpoint(filepath='best_model2_32_32.h5', monitor='val_loss', 
                             save_best_only=True, verbose=1)]
             

#timer = Timer()
model = make_2nd_baseline_model()
model.summary()
#timer.start()
history = model.fit(X_train_img, y_train, epochs=EPOCHS, callbacks=CALLBACKS,
                      batch_size=BATCH_SIZE, validation_data=(X_val_img, y_val))
#timer.stop()
```

    Model: "sequential_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_3 (Dense)              (None, 64)                196672    
    _________________________________________________________________
    dense_4 (Dense)              (None, 7)                 455       
    _________________________________________________________________
    dense_5 (Dense)              (None, 5)                 40        
    _________________________________________________________________
    dense_6 (Dense)              (None, 2)                 12        
    =================================================================
    Total params: 197,179
    Trainable params: 197,179
    Non-trainable params: 0
    _________________________________________________________________
    Train on 4694 samples, validate on 522 samples
    Epoch 1/1000
    4694/4694 [==============================] - 0s 105us/step - loss: 0.5371 - accuracy: 0.7409 - val_loss: 0.4232 - val_accuracy: 0.7471
    
    Epoch 00001: val_loss improved from inf to 0.42319, saving model to best_model2_32_32.h5
    Epoch 2/1000
    4694/4694 [==============================] - 0s 81us/step - loss: 0.3719 - accuracy: 0.8223 - val_loss: 0.3774 - val_accuracy: 0.7778
    
    Epoch 00002: val_loss improved from 0.42319 to 0.37741, saving model to best_model2_32_32.h5
    Epoch 3/1000
    4694/4694 [==============================] - 0s 78us/step - loss: 0.3222 - accuracy: 0.8728 - val_loss: 0.2926 - val_accuracy: 0.8870
    
    Epoch 00003: val_loss improved from 0.37741 to 0.29261, saving model to best_model2_32_32.h5
    Epoch 4/1000
    4694/4694 [==============================] - 0s 83us/step - loss: 0.2611 - accuracy: 0.8918 - val_loss: 0.1791 - val_accuracy: 0.9387
    
    Epoch 00004: val_loss improved from 0.29261 to 0.17914, saving model to best_model2_32_32.h5
    Epoch 5/1000
    4694/4694 [==============================] - 0s 84us/step - loss: 0.2211 - accuracy: 0.9124 - val_loss: 0.1296 - val_accuracy: 0.9540
    
    Epoch 00005: val_loss improved from 0.17914 to 0.12963, saving model to best_model2_32_32.h5
    Epoch 6/1000
    4694/4694 [==============================] - 0s 82us/step - loss: 0.1877 - accuracy: 0.9274 - val_loss: 0.2383 - val_accuracy: 0.9004
    
    Epoch 00006: val_loss did not improve from 0.12963
    Epoch 7/1000
    4694/4694 [==============================] - 0s 78us/step - loss: 0.1857 - accuracy: 0.9280 - val_loss: 0.1934 - val_accuracy: 0.9234
    
    Epoch 00007: val_loss did not improve from 0.12963
    Epoch 8/1000
    4694/4694 [==============================] - 0s 78us/step - loss: 0.1834 - accuracy: 0.9274 - val_loss: 0.1124 - val_accuracy: 0.9617
    
    Epoch 00008: val_loss improved from 0.12963 to 0.11235, saving model to best_model2_32_32.h5
    Epoch 9/1000
    4694/4694 [==============================] - 0s 88us/step - loss: 0.1622 - accuracy: 0.9372 - val_loss: 0.1659 - val_accuracy: 0.9272
    
    Epoch 00009: val_loss did not improve from 0.11235
    Epoch 10/1000
    4694/4694 [==============================] - 0s 79us/step - loss: 0.1506 - accuracy: 0.9399 - val_loss: 0.2086 - val_accuracy: 0.9157
    
    Epoch 00010: val_loss did not improve from 0.11235
    Epoch 11/1000
    4694/4694 [==============================] - 0s 78us/step - loss: 0.1520 - accuracy: 0.9389 - val_loss: 0.0931 - val_accuracy: 0.9617
    
    Epoch 00011: val_loss improved from 0.11235 to 0.09312, saving model to best_model2_32_32.h5
    Epoch 12/1000
    4694/4694 [==============================] - 0s 88us/step - loss: 0.1484 - accuracy: 0.9418 - val_loss: 0.1030 - val_accuracy: 0.9559
    
    Epoch 00012: val_loss did not improve from 0.09312
    Epoch 13/1000
    4694/4694 [==============================] - 0s 81us/step - loss: 0.1498 - accuracy: 0.9389 - val_loss: 0.0964 - val_accuracy: 0.9598
    
    Epoch 00013: val_loss did not improve from 0.09312
    Epoch 14/1000
    4694/4694 [==============================] - 0s 82us/step - loss: 0.1424 - accuracy: 0.9440 - val_loss: 0.3474 - val_accuracy: 0.8467
    
    Epoch 00014: val_loss did not improve from 0.09312
    Epoch 15/1000
    4694/4694 [==============================] - 0s 84us/step - loss: 0.1446 - accuracy: 0.9421 - val_loss: 0.0925 - val_accuracy: 0.9598
    
    Epoch 00015: val_loss improved from 0.09312 to 0.09254, saving model to best_model2_32_32.h5
    Epoch 16/1000
    4694/4694 [==============================] - 0s 80us/step - loss: 0.1298 - accuracy: 0.9521 - val_loss: 0.1121 - val_accuracy: 0.9540
    
    Epoch 00016: val_loss did not improve from 0.09254
    Epoch 17/1000
    4694/4694 [==============================] - 0s 85us/step - loss: 0.1321 - accuracy: 0.9521 - val_loss: 0.0785 - val_accuracy: 0.9655
    
    Epoch 00017: val_loss improved from 0.09254 to 0.07849, saving model to best_model2_32_32.h5
    Epoch 18/1000
    4694/4694 [==============================] - 0s 83us/step - loss: 0.1318 - accuracy: 0.9499 - val_loss: 0.0753 - val_accuracy: 0.9655
    
    Epoch 00018: val_loss improved from 0.07849 to 0.07525, saving model to best_model2_32_32.h5
    Epoch 19/1000
    4694/4694 [==============================] - 0s 82us/step - loss: 0.1299 - accuracy: 0.9501 - val_loss: 0.0730 - val_accuracy: 0.9655
    
    Epoch 00019: val_loss improved from 0.07525 to 0.07299, saving model to best_model2_32_32.h5
    Epoch 20/1000
    4694/4694 [==============================] - 0s 79us/step - loss: 0.1234 - accuracy: 0.9514 - val_loss: 0.0694 - val_accuracy: 0.9789
    
    Epoch 00020: val_loss improved from 0.07299 to 0.06935, saving model to best_model2_32_32.h5
    Epoch 21/1000
    4694/4694 [==============================] - 0s 86us/step - loss: 0.1231 - accuracy: 0.9540 - val_loss: 0.1636 - val_accuracy: 0.9349
    
    Epoch 00021: val_loss did not improve from 0.06935
    Epoch 22/1000
    4694/4694 [==============================] - 0s 83us/step - loss: 0.1217 - accuracy: 0.9538 - val_loss: 0.1497 - val_accuracy: 0.9387
    
    Epoch 00022: val_loss did not improve from 0.06935
    Epoch 23/1000
    4694/4694 [==============================] - 0s 85us/step - loss: 0.1179 - accuracy: 0.9565 - val_loss: 0.0686 - val_accuracy: 0.9789
    
    Epoch 00023: val_loss improved from 0.06935 to 0.06858, saving model to best_model2_32_32.h5
    Epoch 24/1000
    4694/4694 [==============================] - 0s 86us/step - loss: 0.1166 - accuracy: 0.9542 - val_loss: 0.1350 - val_accuracy: 0.9425
    
    Epoch 00024: val_loss did not improve from 0.06858
    Epoch 25/1000
    4694/4694 [==============================] - 0s 87us/step - loss: 0.1198 - accuracy: 0.9533 - val_loss: 0.0839 - val_accuracy: 0.9655
    
    Epoch 00025: val_loss did not improve from 0.06858
    Epoch 26/1000
    4694/4694 [==============================] - 0s 80us/step - loss: 0.1188 - accuracy: 0.9516 - val_loss: 0.0686 - val_accuracy: 0.9808
    
    Epoch 00026: val_loss did not improve from 0.06858
    Epoch 27/1000
    4694/4694 [==============================] - 0s 83us/step - loss: 0.1208 - accuracy: 0.9523 - val_loss: 0.0707 - val_accuracy: 0.9789
    
    Epoch 00027: val_loss did not improve from 0.06858
    Epoch 28/1000
    4694/4694 [==============================] - 0s 80us/step - loss: 0.1137 - accuracy: 0.9576 - val_loss: 0.0626 - val_accuracy: 0.9847
    
    Epoch 00028: val_loss improved from 0.06858 to 0.06261, saving model to best_model2_32_32.h5
    Epoch 29/1000
    4694/4694 [==============================] - 0s 81us/step - loss: 0.1249 - accuracy: 0.9521 - val_loss: 0.0724 - val_accuracy: 0.9693
    
    Epoch 00029: val_loss did not improve from 0.06261
    Epoch 30/1000
    4694/4694 [==============================] - 0s 78us/step - loss: 0.1169 - accuracy: 0.9576 - val_loss: 0.0696 - val_accuracy: 0.9674
    
    Epoch 00030: val_loss did not improve from 0.06261
    Epoch 31/1000
    4694/4694 [==============================] - 0s 82us/step - loss: 0.1140 - accuracy: 0.9561 - val_loss: 0.0900 - val_accuracy: 0.9636
    
    Epoch 00031: val_loss did not improve from 0.06261
    Epoch 32/1000
    4694/4694 [==============================] - 0s 84us/step - loss: 0.1116 - accuracy: 0.9557 - val_loss: 0.0709 - val_accuracy: 0.9674
    
    Epoch 00032: val_loss did not improve from 0.06261
    Epoch 33/1000
    4694/4694 [==============================] - 0s 82us/step - loss: 0.1109 - accuracy: 0.9593 - val_loss: 0.0599 - val_accuracy: 0.9885
    
    Epoch 00033: val_loss improved from 0.06261 to 0.05986, saving model to best_model2_32_32.h5
    Epoch 34/1000
    4694/4694 [==============================] - 0s 82us/step - loss: 0.1052 - accuracy: 0.9599 - val_loss: 0.0631 - val_accuracy: 0.9828
    
    Epoch 00034: val_loss did not improve from 0.05986
    Epoch 35/1000
    4694/4694 [==============================] - 0s 82us/step - loss: 0.1036 - accuracy: 0.9617 - val_loss: 0.1268 - val_accuracy: 0.9483
    
    Epoch 00035: val_loss did not improve from 0.05986
    Epoch 36/1000
    4694/4694 [==============================] - 0s 80us/step - loss: 0.1032 - accuracy: 0.9644 - val_loss: 0.0593 - val_accuracy: 0.9828
    
    Epoch 00036: val_loss improved from 0.05986 to 0.05934, saving model to best_model2_32_32.h5
    Epoch 37/1000
    4694/4694 [==============================] - 0s 83us/step - loss: 0.0996 - accuracy: 0.9606 - val_loss: 0.0587 - val_accuracy: 0.9847
    
    Epoch 00037: val_loss improved from 0.05934 to 0.05872, saving model to best_model2_32_32.h5
    Epoch 38/1000
    4694/4694 [==============================] - 0s 86us/step - loss: 0.1015 - accuracy: 0.9623 - val_loss: 0.0564 - val_accuracy: 0.9751
    
    Epoch 00038: val_loss improved from 0.05872 to 0.05645, saving model to best_model2_32_32.h5
    Epoch 39/1000
    4694/4694 [==============================] - 0s 86us/step - loss: 0.1061 - accuracy: 0.9589 - val_loss: 0.0873 - val_accuracy: 0.9636
    
    Epoch 00039: val_loss did not improve from 0.05645
    Epoch 40/1000
    4694/4694 [==============================] - 0s 88us/step - loss: 0.0980 - accuracy: 0.9602 - val_loss: 0.0556 - val_accuracy: 0.9808
    
    Epoch 00040: val_loss improved from 0.05645 to 0.05560, saving model to best_model2_32_32.h5
    Epoch 41/1000
    4694/4694 [==============================] - 0s 87us/step - loss: 0.1031 - accuracy: 0.9646 - val_loss: 0.0584 - val_accuracy: 0.9847
    
    Epoch 00041: val_loss did not improve from 0.05560
    Epoch 42/1000
    4694/4694 [==============================] - 0s 85us/step - loss: 0.0929 - accuracy: 0.9666 - val_loss: 0.0557 - val_accuracy: 0.9847
    
    Epoch 00042: val_loss did not improve from 0.05560
    Epoch 43/1000
    4694/4694 [==============================] - 0s 78us/step - loss: 0.0949 - accuracy: 0.9668 - val_loss: 0.0565 - val_accuracy: 0.9789
    
    Epoch 00043: val_loss did not improve from 0.05560
    Epoch 44/1000
    4694/4694 [==============================] - 0s 86us/step - loss: 0.0958 - accuracy: 0.9657 - val_loss: 0.4299 - val_accuracy: 0.8621
    
    Epoch 00044: val_loss did not improve from 0.05560
    Epoch 45/1000
    4694/4694 [==============================] - 0s 81us/step - loss: 0.0902 - accuracy: 0.9648 - val_loss: 0.1622 - val_accuracy: 0.9253
    
    Epoch 00045: val_loss did not improve from 0.05560
    Epoch 46/1000
    4694/4694 [==============================] - 0s 89us/step - loss: 0.0876 - accuracy: 0.9680 - val_loss: 0.0535 - val_accuracy: 0.9751
    
    Epoch 00046: val_loss improved from 0.05560 to 0.05349, saving model to best_model2_32_32.h5
    Epoch 47/1000
    4694/4694 [==============================] - 0s 83us/step - loss: 0.0918 - accuracy: 0.9678 - val_loss: 0.0529 - val_accuracy: 0.9808
    
    Epoch 00047: val_loss improved from 0.05349 to 0.05287, saving model to best_model2_32_32.h5
    Epoch 48/1000
    4694/4694 [==============================] - 0s 81us/step - loss: 0.0994 - accuracy: 0.9629 - val_loss: 0.0545 - val_accuracy: 0.9828
    
    Epoch 00048: val_loss did not improve from 0.05287
    Epoch 49/1000
    4694/4694 [==============================] - 0s 87us/step - loss: 0.0902 - accuracy: 0.9653 - val_loss: 0.0612 - val_accuracy: 0.9751
    
    Epoch 00049: val_loss did not improve from 0.05287
    Epoch 50/1000
    4694/4694 [==============================] - 0s 79us/step - loss: 0.0819 - accuracy: 0.9706 - val_loss: 0.0622 - val_accuracy: 0.9732
    
    Epoch 00050: val_loss did not improve from 0.05287
    Epoch 51/1000
    4694/4694 [==============================] - 0s 87us/step - loss: 0.0976 - accuracy: 0.9627 - val_loss: 0.0607 - val_accuracy: 0.9770
    
    Epoch 00051: val_loss did not improve from 0.05287
    Epoch 52/1000
    4694/4694 [==============================] - 0s 82us/step - loss: 0.0861 - accuracy: 0.9676 - val_loss: 0.2908 - val_accuracy: 0.8755
    
    Epoch 00052: val_loss did not improve from 0.05287
    Epoch 53/1000
    4694/4694 [==============================] - 0s 78us/step - loss: 0.0856 - accuracy: 0.9695 - val_loss: 0.0818 - val_accuracy: 0.9636
    
    Epoch 00053: val_loss did not improve from 0.05287
    Epoch 54/1000
    4694/4694 [==============================] - 0s 81us/step - loss: 0.0807 - accuracy: 0.9712 - val_loss: 0.1309 - val_accuracy: 0.9349
    
    Epoch 00054: val_loss did not improve from 0.05287
    Epoch 55/1000
    4694/4694 [==============================] - 0s 87us/step - loss: 0.0817 - accuracy: 0.9700 - val_loss: 0.0862 - val_accuracy: 0.9636
    
    Epoch 00055: val_loss did not improve from 0.05287
    Epoch 56/1000
    4694/4694 [==============================] - 0s 82us/step - loss: 0.0847 - accuracy: 0.9695 - val_loss: 0.0498 - val_accuracy: 0.9885
    
    Epoch 00056: val_loss improved from 0.05287 to 0.04976, saving model to best_model2_32_32.h5
    Epoch 57/1000
    4694/4694 [==============================] - 0s 87us/step - loss: 0.0960 - accuracy: 0.9627 - val_loss: 0.0507 - val_accuracy: 0.9847
    
    Epoch 00057: val_loss did not improve from 0.04976
    Epoch 58/1000
    4694/4694 [==============================] - 0s 87us/step - loss: 0.0903 - accuracy: 0.9672 - val_loss: 0.0458 - val_accuracy: 0.9885
    
    Epoch 00058: val_loss improved from 0.04976 to 0.04584, saving model to best_model2_32_32.h5
    Epoch 59/1000
    4694/4694 [==============================] - 0s 88us/step - loss: 0.0756 - accuracy: 0.9751 - val_loss: 0.0519 - val_accuracy: 0.9847
    
    Epoch 00059: val_loss did not improve from 0.04584
    Epoch 60/1000
    4694/4694 [==============================] - 0s 87us/step - loss: 0.0773 - accuracy: 0.9715 - val_loss: 0.0442 - val_accuracy: 0.9866
    
    Epoch 00060: val_loss improved from 0.04584 to 0.04415, saving model to best_model2_32_32.h5
    Epoch 61/1000
    4694/4694 [==============================] - 0s 82us/step - loss: 0.0792 - accuracy: 0.9685 - val_loss: 0.0464 - val_accuracy: 0.9904
    
    Epoch 00061: val_loss did not improve from 0.04415
    Epoch 62/1000
    4694/4694 [==============================] - 0s 83us/step - loss: 0.0735 - accuracy: 0.9725 - val_loss: 0.0472 - val_accuracy: 0.9866
    
    Epoch 00062: val_loss did not improve from 0.04415
    Epoch 63/1000
    4694/4694 [==============================] - 0s 84us/step - loss: 0.0773 - accuracy: 0.9727 - val_loss: 0.1004 - val_accuracy: 0.9483
    
    Epoch 00063: val_loss did not improve from 0.04415
    Epoch 64/1000
    4694/4694 [==============================] - 0s 87us/step - loss: 0.0745 - accuracy: 0.9751 - val_loss: 0.0458 - val_accuracy: 0.9847
    
    Epoch 00064: val_loss did not improve from 0.04415
    Epoch 65/1000
    4694/4694 [==============================] - 0s 82us/step - loss: 0.0718 - accuracy: 0.9740 - val_loss: 0.0639 - val_accuracy: 0.9751
    
    Epoch 00065: val_loss did not improve from 0.04415
    Epoch 66/1000
    4694/4694 [==============================] - 0s 85us/step - loss: 0.0819 - accuracy: 0.9687 - val_loss: 0.0638 - val_accuracy: 0.9732
    
    Epoch 00066: val_loss did not improve from 0.04415
    Epoch 67/1000
    4694/4694 [==============================] - 0s 87us/step - loss: 0.0789 - accuracy: 0.9729 - val_loss: 0.0474 - val_accuracy: 0.9847
    
    Epoch 00067: val_loss did not improve from 0.04415
    Epoch 68/1000
    4694/4694 [==============================] - 0s 81us/step - loss: 0.0775 - accuracy: 0.9717 - val_loss: 0.0430 - val_accuracy: 0.9904
    
    Epoch 00068: val_loss improved from 0.04415 to 0.04300, saving model to best_model2_32_32.h5
    Epoch 69/1000
    4694/4694 [==============================] - 0s 85us/step - loss: 0.0783 - accuracy: 0.9736 - val_loss: 0.0472 - val_accuracy: 0.9904
    
    Epoch 00069: val_loss did not improve from 0.04300
    Epoch 70/1000
    4694/4694 [==============================] - 0s 88us/step - loss: 0.0675 - accuracy: 0.9781 - val_loss: 0.0418 - val_accuracy: 0.9866
    
    Epoch 00070: val_loss improved from 0.04300 to 0.04181, saving model to best_model2_32_32.h5
    Epoch 71/1000
    4694/4694 [==============================] - 0s 89us/step - loss: 0.0721 - accuracy: 0.9734 - val_loss: 0.0504 - val_accuracy: 0.9847
    
    Epoch 00071: val_loss did not improve from 0.04181
    Epoch 72/1000
    4694/4694 [==============================] - 0s 82us/step - loss: 0.0803 - accuracy: 0.9708 - val_loss: 0.0454 - val_accuracy: 0.9885
    
    Epoch 00072: val_loss did not improve from 0.04181
    Epoch 73/1000
    4694/4694 [==============================] - 0s 82us/step - loss: 0.0637 - accuracy: 0.9778 - val_loss: 0.0736 - val_accuracy: 0.9636
    
    Epoch 00073: val_loss did not improve from 0.04181
    Epoch 74/1000
    4694/4694 [==============================] - 0s 83us/step - loss: 0.0719 - accuracy: 0.9749 - val_loss: 0.0723 - val_accuracy: 0.9693
    
    Epoch 00074: val_loss did not improve from 0.04181
    Epoch 75/1000
    4694/4694 [==============================] - 0s 84us/step - loss: 0.0760 - accuracy: 0.9717 - val_loss: 0.0655 - val_accuracy: 0.9732
    
    Epoch 00075: val_loss did not improve from 0.04181
    Epoch 76/1000
    4694/4694 [==============================] - 0s 80us/step - loss: 0.0687 - accuracy: 0.9746 - val_loss: 0.0578 - val_accuracy: 0.9789
    
    Epoch 00076: val_loss did not improve from 0.04181
    Epoch 77/1000
    4694/4694 [==============================] - 0s 82us/step - loss: 0.0681 - accuracy: 0.9744 - val_loss: 0.0632 - val_accuracy: 0.9713
    
    Epoch 00077: val_loss did not improve from 0.04181
    Epoch 78/1000
    4694/4694 [==============================] - 0s 81us/step - loss: 0.0573 - accuracy: 0.9800 - val_loss: 0.0550 - val_accuracy: 0.9789
    
    Epoch 00078: val_loss did not improve from 0.04181
    Epoch 79/1000
    4694/4694 [==============================] - 0s 89us/step - loss: 0.0771 - accuracy: 0.9725 - val_loss: 0.0462 - val_accuracy: 0.9866
    
    Epoch 00079: val_loss did not improve from 0.04181
    Epoch 80/1000
    4694/4694 [==============================] - 0s 80us/step - loss: 0.0680 - accuracy: 0.9753 - val_loss: 0.0538 - val_accuracy: 0.9789
    
    Epoch 00080: val_loss did not improve from 0.04181
    Epoch 00080: early stopping
    


```python
y_hat_test = model.predict(X_test_img)

evaluate_model(y_test,y_hat_test,history)
```


![png](output_74_0.png)



    <Figure size 432x288 with 0 Axes>


    
    
    ------------------------------------------------------------
    	CLASSIFICATION REPORT:
    ------------------------------------------------------------
                  precision    recall  f1-score   support
    
               0       0.95      0.35      0.51       234
               1       0.72      0.99      0.83       390
    
        accuracy                           0.75       624
       macro avg       0.84      0.67      0.67       624
    weighted avg       0.81      0.75      0.71       624
    
    


![png](output_74_3.png)



```python
#%cd /gdrive/My Drive/'Output'/
```


```python
plt.savefig('Second Baseline-32-32-32-early.jpg');
```


    <Figure size 432x288 with 0 Axes>


# Build a CNN Model

## Create an Optional DataGenerator Using Keras built-in Image Augmentation




```python
# Create a DataGenerator using Keras Image Augmentation
# May not be needed


def make_datagenerator(BATCH_SIZE = 32):
  ''' Create training and test data
  '''

  from keras.preprocessing.image import ImageDataGenerator

  train_datagen = ImageDataGenerator(rescale = 1./255,)
                                  #shear_range = 0.2,
                                  #zoom_range = 0.2,
                                  #horizontal_flip = True)

  test_datagen = ImageDataGenerator(rescale = 1./255)
  val_datagen = ImageDataGenerator(rescale = 1./255)

  training_set = train_datagen.flow(X_train,y=y_train,batch_size=BATCH_SIZE)
  test_set = test_datagen.flow(X_test,y=y_test,batch_size=BATCH_SIZE)
  val_set = val_datagen.flow(X_val,y=y_val,batch_size=BATCH_SIZE)

  return training_set,test_set,val_set
    
training_set,test_set,val_set = make_datagenerator(BATCH_SIZE=32)    
```


```python
print(training_set[0][0].shape)
print('\nLabels for Batch')
print(training_set[0][1].shape)
```

    (32, 32, 32, 3)
    
    Labels for Batch
    (32, 2)
    


```python
# Part 1 - Building the CNN
clock = fs.jmi.Clock()
clock.tic('')
EPOCHS = 6
BATCH_SIZE=32

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(BATCH_SIZE, (3, 3),
                             input_shape = (SHAPES[0],
                                            SHAPES[1],
                                            SHAPES[2]),
                             activation = 'relu'))

#classifier.add(Conv2D(SHAPES['Batchsize'], (3, 3),
#                      input_shape = (SHAPES['img_width'], 
#                                     SHAPES['img_height'],
#                                     SHAPES['img_dim']),
#                       activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(BATCH_SIZE, (3, 3),
                      activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = BATCH_SIZE, activation = 'relu'))
classifier.add(Dense(units = 2, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', 
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])
display(classifier.summary())
# Part 2 - Fitting the CNN to the images

classifier.fit_generator(training_set,
                         steps_per_epoch = 1000,
                         epochs = EPOCHS,
                         validation_data = test_set,
                         validation_steps = 250,workers=-1)

clock.toc('')
```

    --- CLOCK STARTED @:    07/05/20 - 11:51:44 PM           Label:            --- 
    Model: "sequential_10"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_11 (Conv2D)           (None, 30, 30, 32)        896       
    _________________________________________________________________
    max_pooling2d_11 (MaxPooling (None, 15, 15, 32)        0         
    _________________________________________________________________
    conv2d_12 (Conv2D)           (None, 13, 13, 32)        9248      
    _________________________________________________________________
    max_pooling2d_12 (MaxPooling (None, 6, 6, 32)          0         
    _________________________________________________________________
    flatten_6 (Flatten)          (None, 1152)              0         
    _________________________________________________________________
    dense_17 (Dense)             (None, 32)                36896     
    _________________________________________________________________
    dense_18 (Dense)             (None, 2)                 66        
    =================================================================
    Total params: 47,106
    Trainable params: 47,106
    Non-trainable params: 0
    _________________________________________________________________
    


    None


    Epoch 1/6
    1000/1000 [==============================] - 5s 5ms/step - loss: 0.1611 - accuracy: 0.9338 - val_loss: 0.8737 - val_accuracy: 0.7281
    Epoch 2/6
    1000/1000 [==============================] - 5s 5ms/step - loss: 0.0635 - accuracy: 0.9761 - val_loss: 0.9613 - val_accuracy: 0.7792
    Epoch 3/6
    1000/1000 [==============================] - 5s 5ms/step - loss: 0.0400 - accuracy: 0.9859 - val_loss: 1.6931 - val_accuracy: 0.7388
    Epoch 4/6
    1000/1000 [==============================] - 5s 5ms/step - loss: 0.0215 - accuracy: 0.9932 - val_loss: 1.1431 - val_accuracy: 0.7672
    Epoch 5/6
    1000/1000 [==============================] - 5s 5ms/step - loss: 0.0065 - accuracy: 0.9985 - val_loss: 1.9524 - val_accuracy: 0.7576
    Epoch 6/6
    1000/1000 [==============================] - 5s 5ms/step - loss: 0.0113 - accuracy: 0.9959 - val_loss: 2.2692 - val_accuracy: 0.7553
    --- TOTAL DURATION   =  0 min, 30.177 sec --- 
    


<style  type="text/css" >
    #T_8d27f4b8_bf1a_11ea_b554_0242ac1c0002 table, th {
          text-align: center;
    }    #T_8d27f4b8_bf1a_11ea_b554_0242ac1c0002row0_col1 {
            width:  140px;
        }    #T_8d27f4b8_bf1a_11ea_b554_0242ac1c0002row0_col2 {
            width:  140px;
        }</style><table id="T_8d27f4b8_bf1a_11ea_b554_0242ac1c0002" ><caption>Summary Table of Clocked Processes</caption><thead>    <tr>        <th class="col_heading level0 col0" >Lap #</th>        <th class="col_heading level0 col1" >Start Time</th>        <th class="col_heading level0 col2" >Duration</th>        <th class="col_heading level0 col3" >Label</th>    </tr></thead><tbody>
                <tr>
                                <td id="T_8d27f4b8_bf1a_11ea_b554_0242ac1c0002row0_col0" class="data row0 col0" >TOTAL</td>
                        <td id="T_8d27f4b8_bf1a_11ea_b554_0242ac1c0002row0_col1" class="data row0 col1" >07/05/20 - 11:51:44 PM</td>
                        <td id="T_8d27f4b8_bf1a_11ea_b554_0242ac1c0002row0_col2" class="data row0 col2" >0 min, 30.177 sec</td>
                        <td id="T_8d27f4b8_bf1a_11ea_b554_0242ac1c0002row0_col3" class="data row0 col3" ></td>
            </tr>
    </tbody></table>



```python

```


```python
# y_hat_test = classifier.predict_classes(X_test).flatten()
# pd.Series(y_hat_test).value_counts()
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

y_hat_val = classifier.predict_classes(X_val).flatten()
print(pd.Series(y_hat_val))
evaluate_model(y_val,y_hat_val,history)

```

    0      0
    1      0
    2      1
    3      0
    4      1
          ..
    517    1
    518    0
    519    0
    520    1
    521    1
    Length: 522, dtype: int64
    


![png](output_83_1.png)



    <Figure size 432x288 with 0 Axes>


    
    
    ------------------------------------------------------------
    	CLASSIFICATION REPORT:
    ------------------------------------------------------------
                  precision    recall  f1-score   support
    
               0       0.98      0.98      0.98       132
               1       0.99      0.99      0.99       390
    
        accuracy                           0.99       522
       macro avg       0.98      0.98      0.98       522
    weighted avg       0.99      0.99      0.99       522
    
    


![png](output_83_4.png)



```python
plt.savefig('CNN1-32-32-32-early.jpg');
```


    <Figure size 432x288 with 0 Axes>



```python
# Build a CNN with Dropout layers
EPOCHS=6
BATCH_SIZE=32

clock = fs.jmi.Clock()
clock.tic('')
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Dropout

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(BATCH_SIZE, (3, 3),
                             input_shape = (SHAPES[0],
                                            SHAPES[1],
                                            SHAPES[2]),
                             activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.2))

# Adding a second convolutional layer
classifier.add(Conv2D(BATCH_SIZE, (3, 3),
                      activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.2))
# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = BATCH_SIZE, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 2, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', 
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])
display(classifier.summary())
# Part 2 - Fitting the CNN to the images

classifier.fit_generator(training_set,
                         steps_per_epoch = 1000,
                         epochs = EPOCHS,
                         validation_data = test_set,
                         validation_steps = 250,workers=-1)

clock.toc('')
```

    --- CLOCK STARTED @:    07/05/20 - 11:56:37 PM           Label:            --- 
    Model: "sequential_13"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_17 (Conv2D)           (None, 30, 30, 32)        896       
    _________________________________________________________________
    max_pooling2d_17 (MaxPooling (None, 15, 15, 32)        0         
    _________________________________________________________________
    dropout_13 (Dropout)         (None, 15, 15, 32)        0         
    _________________________________________________________________
    conv2d_18 (Conv2D)           (None, 13, 13, 32)        9248      
    _________________________________________________________________
    max_pooling2d_18 (MaxPooling (None, 6, 6, 32)          0         
    _________________________________________________________________
    dropout_14 (Dropout)         (None, 6, 6, 32)          0         
    _________________________________________________________________
    flatten_9 (Flatten)          (None, 1152)              0         
    _________________________________________________________________
    dense_23 (Dense)             (None, 32)                36896     
    _________________________________________________________________
    dropout_15 (Dropout)         (None, 32)                0         
    _________________________________________________________________
    dense_24 (Dense)             (None, 2)                 66        
    =================================================================
    Total params: 47,106
    Trainable params: 47,106
    Non-trainable params: 0
    _________________________________________________________________
    


    None


    Epoch 1/6
    1000/1000 [==============================] - 6s 6ms/step - loss: 0.2347 - accuracy: 0.9030 - val_loss: 0.6211 - val_accuracy: 0.7794
    Epoch 2/6
    1000/1000 [==============================] - 6s 6ms/step - loss: 0.1173 - accuracy: 0.9607 - val_loss: 0.3294 - val_accuracy: 0.7703
    Epoch 3/6
    1000/1000 [==============================] - 6s 6ms/step - loss: 0.0888 - accuracy: 0.9719 - val_loss: 0.9112 - val_accuracy: 0.7926
    Epoch 4/6
    1000/1000 [==============================] - 6s 6ms/step - loss: 0.0721 - accuracy: 0.9766 - val_loss: 1.4996 - val_accuracy: 0.7803
    Epoch 5/6
    1000/1000 [==============================] - 6s 6ms/step - loss: 0.0593 - accuracy: 0.9814 - val_loss: 1.4914 - val_accuracy: 0.7553
    Epoch 6/6
    1000/1000 [==============================] - 6s 6ms/step - loss: 0.0464 - accuracy: 0.9844 - val_loss: 1.3102 - val_accuracy: 0.8106
    --- TOTAL DURATION   =  0 min, 34.195 sec --- 
    


<style  type="text/css" >
    #T_3e765f16_bf1b_11ea_b554_0242ac1c0002 table, th {
          text-align: center;
    }    #T_3e765f16_bf1b_11ea_b554_0242ac1c0002row0_col1 {
            width:  140px;
        }    #T_3e765f16_bf1b_11ea_b554_0242ac1c0002row0_col2 {
            width:  140px;
        }</style><table id="T_3e765f16_bf1b_11ea_b554_0242ac1c0002" ><caption>Summary Table of Clocked Processes</caption><thead>    <tr>        <th class="col_heading level0 col0" >Lap #</th>        <th class="col_heading level0 col1" >Start Time</th>        <th class="col_heading level0 col2" >Duration</th>        <th class="col_heading level0 col3" >Label</th>    </tr></thead><tbody>
                <tr>
                                <td id="T_3e765f16_bf1b_11ea_b554_0242ac1c0002row0_col0" class="data row0 col0" >TOTAL</td>
                        <td id="T_3e765f16_bf1b_11ea_b554_0242ac1c0002row0_col1" class="data row0 col1" >07/05/20 - 11:56:37 PM</td>
                        <td id="T_3e765f16_bf1b_11ea_b554_0242ac1c0002row0_col2" class="data row0 col2" >0 min, 34.195 sec</td>
                        <td id="T_3e765f16_bf1b_11ea_b554_0242ac1c0002row0_col3" class="data row0 col3" ></td>
            </tr>
    </tbody></table>



```python

```


```python
# y_hat_test = classifier.predict_classes(X_test).flatten()
# pd.Series(y_hat_test).value_counts()
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

y_hat_val = classifier.predict_classes(X_val).flatten()
print(pd.Series(y_hat_val))
evaluate_model(y_val,y_hat_val, history)

```

    0      0
    1      0
    2      1
    3      0
    4      1
          ..
    517    1
    518    0
    519    0
    520    1
    521    1
    Length: 522, dtype: int64
    


![png](output_87_1.png)



    <Figure size 432x288 with 0 Axes>


    
    
    ------------------------------------------------------------
    	CLASSIFICATION REPORT:
    ------------------------------------------------------------
                  precision    recall  f1-score   support
    
               0       0.92      0.98      0.95       132
               1       0.99      0.97      0.98       390
    
        accuracy                           0.97       522
       macro avg       0.96      0.97      0.97       522
    weighted avg       0.97      0.97      0.97       522
    
    


![png](output_87_4.png)



```python

```


```python
plt.savefig('CNN1-dropout-32-32-32-early.jpg');
```


    <Figure size 432x288 with 0 Axes>


# Clearly Convolution Neural Networks are very powerful for classification of images

While fully dense connected layers already had good accuracy, particular for the Recall of True Positives (those with Pneumonia),  the CNNs dramatically improved the True Negatives (those that don't have Pneumonia).  

It can be hoped that these types of models can serve as "2nd opinions" or a second set of eyes,  to doctors and nurses in resource constrained countries, where human resources are a limiting factor in stopping the preventable disease of childhood pneumonia.


```python

```
