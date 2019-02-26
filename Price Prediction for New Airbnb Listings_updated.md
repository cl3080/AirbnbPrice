
# Airbnb Price Prediction for New Listings

## I. Introduction

Airbnb is a great platform that provides people online marketplace and service to arrange or offer lodging. The revenue of Airbnb in 2017 has exceeded $2.5 billion. As a travel enthusiast, Airbnb is always my first choice when I am planning a trip. For potential hosts, it might be difficult for them to decide the price of a totally new listing. As far as I know, there is no such a public model on Airbnb website for suggesting prices for new listings. So, in order to help potential hosts to get a descent idea on how much to charge for new listings, this project will build price prediction models for new listings on Airbnb. 

Fortunately, [Inside Airbnb](http://insideairbnb.com/get-the-data.html) has already aggregated all the publicly available informations from Airbnb site for public discussion. So, the dataset obtained from this website directly should be a good starting point for my machine learning model. In particular, I will the dataset collected in New York city compiled on 06 December, 2018. When selecting features for machine learning model, besides the variables provided in the datasets, the photo on the listing's website and the description of listing can be crucial for increasing the value of the listings. So, I will also analyze featured photos and text mining on the descriptions and take these two new features into considerations to see if performance of the prediction models can be improved. 

The project will be described as follows:
    1. Exploratory data analysis.
    2. Feature engineering (Image assessment on web photos and sentiment analysis on descriptions).
    3. Machine learning models and refinement.
    4. Future work.
    5. Conclusion.


```python
import os
import numpy as np
import pandas as pd
import re
import math
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
```


```python
df = pd.read_csv('listings.csv')
print ('There are {} rows and {} columns in the dataset'.format(*df.shape))
df.head(3)
```

    There are 49056 rows and 96 columns in the dataset





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
      <th>id</th>
      <th>listing_url</th>
      <th>scrape_id</th>
      <th>last_scraped</th>
      <th>name</th>
      <th>summary</th>
      <th>space</th>
      <th>description</th>
      <th>experiences_offered</th>
      <th>neighborhood_overview</th>
      <th>...</th>
      <th>requires_license</th>
      <th>license</th>
      <th>jurisdiction_names</th>
      <th>instant_bookable</th>
      <th>is_business_travel_ready</th>
      <th>cancellation_policy</th>
      <th>require_guest_profile_picture</th>
      <th>require_guest_phone_verification</th>
      <th>calculated_host_listings_count</th>
      <th>reviews_per_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2515</td>
      <td>https://www.airbnb.com/rooms/2515</td>
      <td>20181206022948</td>
      <td>2018-12-06</td>
      <td>Stay at Chez Chic budget room #1</td>
      <td>Step into our artistic spacious apartment and ...</td>
      <td>-PLEASE BOOK DIRECTLY. NO NEED TO SEND A REQUE...</td>
      <td>Step into our artistic spacious apartment and ...</td>
      <td>none</td>
      <td>NaN</td>
      <td>...</td>
      <td>f</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>f</td>
      <td>f</td>
      <td>strict_14_with_grace_period</td>
      <td>f</td>
      <td>f</td>
      <td>3</td>
      <td>1.42</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21456</td>
      <td>https://www.airbnb.com/rooms/21456</td>
      <td>20181206022948</td>
      <td>2018-12-06</td>
      <td>Light-filled classic Central Park</td>
      <td>An adorable, classic, clean, light-filled one-...</td>
      <td>An adorable, classic, clean, light-filled one-...</td>
      <td>An adorable, classic, clean, light-filled one-...</td>
      <td>none</td>
      <td>Diverse. Great coffee shops and restaurants, n...</td>
      <td>...</td>
      <td>f</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>f</td>
      <td>f</td>
      <td>moderate</td>
      <td>t</td>
      <td>t</td>
      <td>1</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2539</td>
      <td>https://www.airbnb.com/rooms/2539</td>
      <td>20181206022948</td>
      <td>2018-12-06</td>
      <td>Clean &amp; quiet apt home by the park</td>
      <td>Renovated apt home in elevator building.</td>
      <td>Spacious, renovated, and clean apt home, one b...</td>
      <td>Renovated apt home in elevator building. Spaci...</td>
      <td>none</td>
      <td>Close to Prospect Park and Historic Ditmas Park</td>
      <td>...</td>
      <td>f</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>f</td>
      <td>f</td>
      <td>moderate</td>
      <td>f</td>
      <td>f</td>
      <td>8</td>
      <td>0.25</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 96 columns</p>
</div>




```python
# where to save figures and results
ROOT_DIR = os.path.dirname(os.path.realpath('__file__'))
Image_Path = os.path.join(ROOT_DIR,'Images')

if not os.path.exists('Images'):
    os.makedirs('Images')
Image_path = os.path.join('Images')

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(Image_path,fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)
```


```python
df.columns
```




    Index(['id', 'listing_url', 'scrape_id', 'last_scraped', 'name', 'summary',
           'space', 'description', 'experiences_offered', 'neighborhood_overview',
           'notes', 'transit', 'access', 'interaction', 'house_rules',
           'thumbnail_url', 'medium_url', 'picture_url', 'xl_picture_url',
           'host_id', 'host_url', 'host_name', 'host_since', 'host_location',
           'host_about', 'host_response_time', 'host_response_rate',
           'host_acceptance_rate', 'host_is_superhost', 'host_thumbnail_url',
           'host_picture_url', 'host_neighbourhood', 'host_listings_count',
           'host_total_listings_count', 'host_verifications',
           'host_has_profile_pic', 'host_identity_verified', 'street',
           'neighbourhood', 'neighbourhood_cleansed',
           'neighbourhood_group_cleansed', 'city', 'state', 'zipcode', 'market',
           'smart_location', 'country_code', 'country', 'latitude', 'longitude',
           'is_location_exact', 'property_type', 'room_type', 'accommodates',
           'bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities', 'square_feet',
           'price', 'weekly_price', 'monthly_price', 'security_deposit',
           'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights',
           'maximum_nights', 'calendar_updated', 'has_availability',
           'availability_30', 'availability_60', 'availability_90',
           'availability_365', 'calendar_last_scraped', 'number_of_reviews',
           'first_review', 'last_review', 'review_scores_rating',
           'review_scores_accuracy', 'review_scores_cleanliness',
           'review_scores_checkin', 'review_scores_communication',
           'review_scores_location', 'review_scores_value', 'requires_license',
           'license', 'jurisdiction_names', 'instant_bookable',
           'is_business_travel_ready', 'cancellation_policy',
           'require_guest_profile_picture', 'require_guest_phone_verification',
           'calculated_host_listings_count', 'reviews_per_month'],
          dtype='object')



## II. Exploratory data  analysis and data preprocessing

There are 49056 observations and 96 columns in the dataset. However, not all the columns are needed for the model. Especially, for a new house, there won't be any information about reviews. So columns containing informations about reviews should be dropped. These features are "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin", "review_scores_communication", "review_scores_location", "review_scores_value", "reviews_per_month". After carefully considering each features, these features are kept for further data analysis: 
> - **listing_url:** from the url, photos of the listings can be scraped. Needless to say, an attractive featured photo of the listing can help to increase the value of the listings.<br>
- **description:** In description, host can emphasize the advantages of the listings. For example, phases like "1 min walk to subway", "beautiful view of Hudson river" can help tourists to make the decision.<br>
- **latitude, longitude, zipcode, neighbourhood_group_cleansed:** these four columns provide the information about the location.<br>
- **property_type, room_type, bathrooms, bedrooms, bed_type, square_feet:** these columns describe the properties of the listings, such as how large is the aparment, how many bathrooms or bedrooms it has.<br>
- **guests_included, cleaning_fee, extra_people, minimum_nights, maximum_nights, availability_365, cancellation_policy, security_deposit, host_is_superhost:** these columns provide informations about the policy of booking a room. The house with more flexible policy may be more prefered for some tourists who are not so sure about their schedules. <br>
- **id:** this id is kept for later image scraping.

The data cleaning process will be performed as follows:
1. Drop all the unnecessary columns.
2. "cleaning_fee","extra_people","price" have the dollar sign before the number. Need to remove the "\\$" and change the datetype from string to numerical values.
3. Categorical variables including "property_type","bed_type","room_type","cancellation_policy" contain types with only a few observations, so those categories can be combined into one category and name it "Other".
4. Handle missing values. First, columns including "bathrooms","bedrooms" and "price" have NULL values. They can be filled in with the median. Column "square_feet","cleaning_fee" and "security_deposi" have large proportions of missing values, these three variables will be dropped for further analysis.
5. Check the distribution of numerical variables. Some listings are extremely large (many bathrooms, bedrooms and can accommodate much more people), they need to be removed as outliers.


```python
# drop all the unnecessary columns
feature_to_keep = ['listing_url','id','description','latitude','longitude','property_type','room_type','accommodates','bathrooms',
                  'bedrooms','bed_type','price','square_feet','guests_included','cleaning_fee','extra_people','minimum_nights',
                  'maximum_nights','availability_365','cancellation_policy','neighbourhood_group_cleansed','security_deposit','host_is_superhost']

new_df = df[feature_to_keep]

# remove the dollar sign before "cleaning_fee", "extra_people", "price" and change the datatype to numerical variables
feature_to_remove_dollar = ['cleaning_fee','extra_people','price']
new_df[feature_to_remove_dollar] = new_df[feature_to_remove_dollar].replace('\$','',regex = True)
new_df[feature_to_remove_dollar] = new_df[feature_to_remove_dollar].apply(pd.to_numeric,errors = "coerce")
```


```python
# check the missing values
new_df.isna().sum()
```




    listing_url                         0
    id                                  0
    description                       534
    latitude                            0
    longitude                           0
    property_type                       0
    room_type                           0
    accommodates                        0
    bathrooms                          76
    bedrooms                           49
    bed_type                            0
    price                             279
    square_feet                     48590
    guests_included                     0
    cleaning_fee                    11079
    extra_people                        0
    minimum_nights                      0
    maximum_nights                      0
    availability_365                    0
    cancellation_policy                 0
    neighbourhood_group_cleansed        0
    security_deposit                18087
    host_is_superhost                   7
    dtype: int64




```python
# drop the column "square_feet" and "cleaning_fee"
new_df = new_df.drop(['square_feet','cleaning_fee','security_deposit'], axis = 1)

# drop 534 rows with missing descriptions
new_df['description'] = new_df['description'].dropna()

# fill NaN with median value for 'bathrooms', 'bedrooms','price'
new_df['bathrooms'] = new_df['bathrooms'].fillna(new_df['bathrooms'].median())
new_df['bedrooms'] = new_df['bedrooms'].fillna(new_df['bedrooms'].median())
new_df['price'] = new_df['price'].fillna(new_df['price'].median())
```


```python
new_df['property_type'].value_counts()
```




    Apartment                 39301
    House                      3527
    Townhouse                  1652
    Loft                       1535
    Condominium                1359
    Serviced apartment          691
    Guest suite                 258
    Other                       129
    Boutique hotel              112
    Bed and breakfast           111
    Resort                       89
    Hotel                        75
    Guesthouse                   56
    Hostel                       48
    Bungalow                     29
    Villa                        25
    Tiny house                   13
    Aparthotel                   12
    Boat                          8
    Cottage                       5
    Camper/RV                     4
    Tent                          3
    Earth house                   3
    Cabin                         3
    Casa particular (Cuba)        2
    Island                        1
    Cave                          1
    Houseboat                     1
    Bus                           1
    Castle                        1
    Nature lodge                  1
    Name: property_type, dtype: int64




```python
# merge small catergories in property_type into one category "Other"
Other = ['Serviced apartment','Guest suite','Other','Boutique hotel','Bed and breakfast','Resort','Hotel','Guesthouse',
        'Hostel','Bungalow','Villa','Tiny house','Aparthotel','Boat','Cottage','Camper/RV','Tent','Cabin','Earth house',
        'Casa particular (Cuba)','Cave','Bus','Castle','Island','Nature lodge','Houseboat']
new_df['property_type'].loc[new_df['property_type'].isin(Other)] = "Other"
```


```python
new_df['room_type'].value_counts()
```




    Entire home/apt    26059
    Private room       21934
    Shared room         1063
    Name: room_type, dtype: int64




```python
new_df['bed_type'].value_counts()
```




    Real Bed         48156
    Futon              338
    Pull-out Sofa      285
    Airbed             195
    Couch               82
    Name: bed_type, dtype: int64




```python
# merge small catergories in bed_type into one category "No Bed"
Other = ['Futon','Pull-out Sofa','Airbed','Couch']
new_df['bed_type'].loc[new_df['bed_type'].isin(Other)] = "No Bed"
```


```python
new_df['cancellation_policy'].value_counts()
```




    strict_14_with_grace_period    22875
    flexible                       14626
    moderate                       11381
    super_strict_60                  127
    super_strict_30                   44
    strict                             2
    long_term                          1
    Name: cancellation_policy, dtype: int64




```python
# merge small catergories in cancellation_policy into one category "Other"
Other = ['super_strict_60','super_strict_30','strict','long_term']
new_df['cancellation_policy'].loc[new_df['cancellation_policy'].isin(Other)] = "Other"
```


```python
# check the distribution of Number of bedroom and Number of bathroom
%matplotlib inline
fig,axs = plt.subplots(ncols = 2, nrows = 3, figsize = (16,8))
plt.subplots_adjust(left=0, bottom=0, right=1, top=0.9,hspace=0.5,wspace=0.3)
sns.set(style = "white",font_scale=1.5)

sns.distplot(pd.Series(new_df['bathrooms'],name = "Number of bathrooms (Before cleaning)"),color = 'blue', ax = axs[0,0])
sns.distplot(pd.Series(new_df['bedrooms'], name = "Number of bedrooms (Before cleaning)"), color = "orange",ax = axs[1,0])
sns.distplot(pd.Series(new_df['accommodates'],name = "Number of accommodates (Before cleaning)"),color = 'red', ax = axs[2,0])

new_df = new_df[new_df['bathrooms']<3]
new_df = new_df[new_df['bedrooms']<5]
new_df = new_df[new_df['accommodates']<8]

sns.distplot(pd.Series(new_df['bathrooms'],name = "Number of bathrooms (After cleaning)"),color = 'blue', ax = axs[0,1])
sns.distplot(pd.Series(new_df['bedrooms'], name = "Number of bedrooms (After cleaning)"), color = "orange",ax = axs[1,1])
sns.distplot(pd.Series(new_df['accommodates'],name = "Number of bathrooms (After cleaning)"),color = 'red', ax = axs[2,1])

save_fig("Distribution_of_variables")

print ("Dataset has {} rows and {} columns.".format(*new_df.shape))

new_df.to_csv('cleaned_df.csv')
```

    Saving figure Distribution_of_variables
    Dataset has 47199 rows and 20 columns.



![png](output_19_1.png)


### Location analysis


```python
plt.figure(figsize = (12, 10))
img = scipy.misc.imread('map.png')
plt.imshow(img);
```


![png](output_21_0.png)



```python
df_location = new_df.groupby('neighbourhood_group_cleansed',as_index = False).aggregate({
    'price':['mean','median'],
    'accommodates':'mean'
})
df_location['count'] = new_df['neighbourhood_group_cleansed'].value_counts().tolist()
df_location
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>neighbourhood_group_cleansed</th>
      <th colspan="2" halign="left">price</th>
      <th>accommodates</th>
      <th>count</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bronx</td>
      <td>77.976163</td>
      <td>65.0</td>
      <td>2.474461</td>
      <td>21841</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Brooklyn</td>
      <td>109.634747</td>
      <td>90.0</td>
      <td>2.631191</td>
      <td>19121</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Manhattan</td>
      <td>170.643469</td>
      <td>145.0</td>
      <td>2.726157</td>
      <td>5063</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Queens</td>
      <td>88.036737</td>
      <td>72.0</td>
      <td>2.552834</td>
      <td>881</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Staten Island</td>
      <td>78.000000</td>
      <td>66.0</td>
      <td>2.788396</td>
      <td>293</td>
    </tr>
  </tbody>
</table>
</div>



#### Comparison among different neighborhoods


```python
%matplotlib inline
fig, ax = plt.subplots(figsize = (10,5))
index = np.arange(5)
bar_width = 0.35
opacity = 0.8
 
rects1 = plt.bar(df_location['neighbourhood_group_cleansed'], df_location['price']['mean'], bar_width,
alpha=opacity,
color='purple',
label='Mean')
 
rects2 = plt.bar(index + bar_width,df_location['price']['median'], bar_width,
alpha=opacity,
color='g',
label='Median')
 
plt.xlabel('Neighbourhood')
plt.ylabel('Price ($)')
plt.title('Price Comparison among different neighbourhoods')
plt.xticks(index + 0.5*bar_width, df_location['neighbourhood_group_cleansed'])
plt.legend()
 
plt.tight_layout() 
save_fig("Price comparison among different neighborhoods")

plt.show()
```

    Saving figure Price comparison among different neighborhoods



![png](output_24_1.png)



```python
%matplotlib inline
fig,axs = plt.subplots(ncols = 2,figsize = (16,6))
bar_width = 0.40
opacity = 0.8

plt.subplot(1, 2, 1)
plt.bar(df_location['neighbourhood_group_cleansed'], df_location['count'], bar_width,
alpha=opacity,
color='blue')

plt.xlabel('Neighbourhood')
plt.ylabel('Number of listings')
plt.title('Number of Listings in different neighbourhoods')
 
plt.subplot(1, 2, 2)
plt.bar(df_location['neighbourhood_group_cleansed'], df_location['accommodates']['mean'], bar_width,
alpha=opacity,
color='green')

plt.xlabel('Neighbourhood')
plt.ylabel('Number of bedrooms')
plt.title('Averaged number of bedrooms in different neighbourhoods')

plt.tight_layout()
save_fig("Comparison among different neighborhoods")
plt.show()
```

    Saving figure Comparison among different neighborhoods



![png](output_25_1.png)


## III. Feature Engineering

When I am looking for a place to stay on Airbnb, I always first check the photos of the listing: do they look attractive to me? Then I usually take a look of the description of the listings: Is the text showing me a comfortable room to live? So, based on my personal experience, an attractive photos and a pleasant description can be helpful to increase the value of the listings. In this part, I will focus on engineering two feaures: rate the attractiveness of each photos and topic modeling on description of the listings.

### Photo analysis

In most cases, hosts on Airbnb will upload some photos of their houses. These photos, especially the featured photo on the website, are extremely important to attract more viewers. An ideal photo should have desirable resolution and also be aesthetically attractive. Here I will use __[NIMA: Neural Image Assessment](https://ai.googleblog.com/2017/12/introducing-nima-neural-image-assessment.html)__ to score image quality. In NIMA, a deep convolutional neural network (CNN) is trained to predict whether the image will rated by a viewer as looking good (technically) and attractive (aesthetically). 

To assess both resolution and perceptual quality,  the model first initialize weights from object recognition networks, such as ImageNet, to understand general classification of objects. Then the perceptual quality assessment is achieved by fine-tuning on annotated data. This NIMA model gives a distribution of ratings for a given image on scale of 1 to 10 and also assign the probabilities. NIMA has been tested on Aesthetic Visual Analysis (AVA) datasets, and the rank given by NIMA matches closely the mean scores given by human raters.  

Here, I will use the pre-trained the NIMA model [Github](https://github.com/titu1994/neural-image-assessment) to predict the image score for each featured photo on the website and this score will be incorporated as a new feature for machine learning model. The workflow will be as follows:

1. Use beautiful soup to scrape images from the url link of the listed houses.
2. Predict the image score use NIMA model.



```python
import requests
from bs4 import BeautifulSoup
import urllib.request
import scipy.misc
```


```python
new_df = new_df.reset_index()
```


```python
# extract the url for the web photo from 'listings_url'
listings = new_df['listing_url']
image_link = {}
for i in range(len(listings)):
    file_url = listings[i]
    page = requests.get(file_url)    
    soup = BeautifulSoup(page.text,"html.parser")
    img_tags = soup.find_all('img')
    img_urls = [img['src'] for img in img_tags]
    for url in img_urls:
        if not url.startswith("https://a0.muscache.com/im/pictures/"):
            continue
        image_link_1[file_url] = url
      # print (len(image_link))
      # np.save('imagelink_dic.npy',image_link)
        break
```


```python
# save the link and the dataframe
new_df['image_link'] = new_df['listing_url'].map(image_link)
new_df.to_csv('new_df_imagelink.csv')
```


```python
# some listings are no longer available and there is no photo for them
df_imagelink = df_imagelink.dropna(subset=['image_link'])
```




    12588




```python
df_imagelink.to_csv('webphoto_link.csv')
```


```python
# set up the path for the photos output
ROOT_DIR = os.path.dirname(os.path.realpath('__file__'))
Photo_Path = os.path.join(ROOT_DIR,'Photos')

if not os.path.exists('Photos'):
    os.makedirs('Photos')
Photo_path = os.path.join('Photos')

# scraping images from the link
df_imagelink = df_imagelink[['id','image_link']].reset_index()

for i in range(len(df_imagelink)):
   # link = df_image['image_link'][i]
    link = df_imagelink['image_link'][i]
   # print (url_link)
    photo_id = df_imagelink['id'][i]
    image_name = os.path.join(Photo_Path,str(photo_id)+str('.jpg'))

    if not os.path.isfile(image_name):
        f = open(image_name,'wb')
        f.write(requests.get(link).content)
        f.close()
```


```python
# take random samples and check if their NIMA scores make sense
sample = df_imagelink['image_link'][25]
photo_id = df_imagelink['id'][25]
image_name = os.path.join(Photo_path, str(photo_id)+str('.jpg'))
img = scipy.misc.imread(image_name)
plt.imshow(img)
```




    <matplotlib.image.AxesImage at 0x11b51d710>




![png](output_37_1.png)



```python
# use NIMA model to score images
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from utils import mean_score, std_score

NIMA_dic = {}
image_name = os.path.join(Photo_path, str(photo_id)+str('.jpg'))

with tf.device('/CPU:0'):
    base_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    model = Model(base_model.input, x)
    model.load_weights('weights/mobilenet_weights.h5')
        
    for i in range(len(df_imagelink)): 
        try:
            photo_id = df_imagelink['id'][i]
            image_name = os.path.join(Photo_path, str(photo_id)+str('.jpg'))
        

            img = load_img(image_name)
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            scores = model.predict(x, batch_size=1, verbose=0)[0]

            mean = mean_score(scores)
            std = std_score(scores)
            NIMA_dic[photo_id] = mean
            #print("NIMA Score : %0.3f +- (%0.3f)" % (mean, std))
        except:
            pass
```


```python
# add NIMA_score to new_df
new_df['NIMA_score'] = new_df['id'].map(NIMA_dic)

# fill Null values with median
new_df['NIMA_score'] = new_df['NIMA_score'].dropna()
new_df = new_df.dropna(subset = ['NIMA_score'])

# save file into a csv
new_df.to_csv('new_df_withNIMA.csv')
```


```python
# pick some random samples to check if there score make sense
import random
samples = random.sample(range(10000),3)

from PIL import Image
for sample in samples:
    photo_id = new_df['id'][sample]
    image_name = os.path.join(Photo_path, str(photo_id)+str('.jpg'))
    display(Image.open(image_name))
    score = new_df['NIMA_score'][sample]
    print (score)
```


![png](output_40_0.png)


    5.1056564404862



![png](output_40_2.png)


    4.512407927075401



![png](output_40_4.png)


    4.801112962886691


### Sentiment analysis on description

Description of the houses also has a great impact on guest's decision. An appropriate description can not only provide viewers with more details of the room but also leave them good impressions of the living environment using phrases such as "comfortable", "lovely bedroom", "bright and sunny room". So this part will focus on extraccting useful features from description. __Nature language processing (NLP)__ and __topic modeling__ will be carried out to analyze the text in 'description'. 

Topic model is a widely used text-mining tools to discover the abstract "topics" hidden in a collection of documents. Here, __Latent Dirichlet Allocation (LDA)__ will be used to discover topics in each description. In LDA model, a generative Bayesian inference model is used to assign each document with a probability distribution over topics, where topics are probability over words. 

Before topic modeling, the number of corpus in each description needs to be reduced. Non-english words, stop words and non-alphanumeric strings will be removed. The remaining corpus will also be lemmatised so that only important  and meaningful words will be kept later sentiment analysis. The corpus then needs to be converted into a __Document-term-matrix__, where each row corresponding to the documents and column corresponding to the terms. 


The pipeline of topic modeling on text of description will be as follows:
1. Tokenize words, remove non-english words, stop words and non-alphanumeric strings, convert all letters to lower case, and lemmatize words.
2. Convert the remaining corpus into Document Term Matrix.
3. Apply LDA model to model topics.
4. Use pyLDAvis.gensim to visualize topics.
5. Assign each observation with the topics with highest probability.


```python
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.models.ldamodel import LdaModel

import json
import pyLDAvis.gensim
pyLDAvis.enable_notebook()
```


```python
def preprocess_text(corpus):
    processed_corpus = []
    english_words = set(nltk.corpus.words.words())
    english_stopwords = set(stopwords.words('english'))
    wordnet_lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'[A-Za-z|!]+')
    for row in corpus:
        sentences = []
        word_tokens = tokenizer.tokenize(row)
        word_tokens_lower = [t.lower() for t in word_tokens]
        word_tokens_lower_english = [t for t in word_tokens_lower if t in english_words or not t.isalpha()]
        word_tokens_no_stops = [t for t in word_tokens_lower_english if not t in english_stopwords]
        word_tokens_no_stops_lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in word_tokens_no_stops]
        for word in word_tokens_no_stops_lemmatized:
            if len(word) >= 2:
                sentences.append(word)
        processed_corpus.append(sentences)
    return processed_corpus

def pipline(processed_corpus):
    dictionary = Dictionary(processed_corpus)
    doc_term_matrix = [dictionary.doc2bow(listing) for listing in processed_corpus]
    return dictionary, doc_term_matrix

def lda_topic_model(doc_term_matrix,dictionary,num_topics = 3, passes = 2):
    LDA = LdaModel
    ldamodel = LDA(doc_term_matrix,num_topics = num_topics, id2word = dictionary, passes = passes)
    return ldamodel

def topic_feature(ldamodel,doc_term_matrix,df,new_col,num_topics):
    docTopicProbMat = ldamodel[doc_term_matrix]
    docTopicProbDf = pd.DataFrame(index = df.index, columns = range(0,num_topics))
    for i,doc in enumerate(docTopicProbMat):
        for topic in doc:
            docTopicProbDf.iloc[i,topic[0]] = topic[1]
    docTopicProbDf = docTopicProbDf.fillna(0)
    docTopicProbDf[new_col] = docTopicProbDf.idxmax(axis=1)
    df_topics = docTopicProbDf[new_col]
    df_new = pd.concat([df,df_topics],axis = 1)
    return df_new
```


```python
corpus_description = new_df['description'].astype(str)

# use nlp package to process the text in description
processed_corpus_description = preprocess_text(corpus_description)

# generate the doc_term_matrix for lda model
dictionary_description, doc_term_matrix_description = pipline(processed_corpus_description)

# lda model for topic modeling
ldamodel_description = lda_topic_model(doc_term_matrix_description,dictionary_description)

# add the topic feature to the dataframe
final_df = topic_feature(ldamodel_description,doc_term_matrix_description,new_df,new_col = 'description_topic', num_topics =3)

# visualization of the lda model and save it as html page
p_description = pyLDAvis.gensim.prepare(ldamodel_description, doc_term_matrix_description, dictionary_description)
pyLDAvis.save_html(p_description,'lda_description.html')
```

pyLDAvis package is a great package to visualize the LDA model. The area of the circles means the prevalence of each topic. Here I chose the cluster the corpus into three topics. The red bar represents the estimated term frequency within selected topic and the blue bar represents the overall term frequency. In topic 1, the prevalent term is about the transit of the listings, for example, there are words "walk", "train", "away". Topic 2 is about the neighborhood because it has words "city","neighborhood","york","park". Topic 3 is correlated with utility of the house shown by words "floor", "large","space","private".


```python
p_description
```





<link rel="stylesheet" type="text/css" href="https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.css">


<div id="ldavis_el2695355642553529668179362"></div>
<script type="text/javascript">

var ldavis_el2695355642553529668179362_data = {"mdsDat": {"Freq": [34.19152069091797, 33.53374481201172, 32.274742126464844], "cluster": [1, 1, 1], "topics": [1, 2, 3], "x": [-0.04034420018059402, -0.07654819976101787, 0.11689239994161192], "y": [0.07194467972690471, -0.058479641060509534, -0.013465038666395202]}, "tinfo": {"Category": ["Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3"], "Freq": [11373.0, 16009.0, 39387.0, 8635.0, 18104.0, 19212.0, 8981.0, 16001.0, 7245.0, 16717.0, 27883.0, 4304.0, 6573.0, 7516.0, 16815.0, 10728.0, 9029.0, 7185.0, 11361.0, 4357.0, 5994.0, 17485.0, 14884.0, 5296.0, 27555.0, 2168.0, 3875.0, 5954.0, 3577.0, 2251.0, 929.4793701171875, 406.7648010253906, 222.2169647216797, 335.953857421875, 158.70225524902344, 519.9014892578125, 431.66571044921875, 118.95157623291016, 172.20657348632812, 183.94573974609375, 75.50041198730469, 2141.546630859375, 96.40876770019531, 56.78325271606445, 68.5478286743164, 81.61344909667969, 44.077877044677734, 64.29070281982422, 57.19279861450195, 41.403221130371094, 750.1597900390625, 52.08663558959961, 180.24183654785156, 51.78423309326172, 37.568878173828125, 31.75267219543457, 32.323917388916016, 29.126150131225586, 96.9127197265625, 86.35930633544922, 959.2006225585938, 804.939453125, 463.54681396484375, 263.7723083496094, 206.36898803710938, 100.30462646484375, 3880.19921875, 199.2811279296875, 7553.2421875, 706.4822998046875, 415.7947692871094, 473.0264587402344, 139.9507293701172, 2396.845947265625, 5275.49853515625, 778.9005126953125, 268.524169921875, 1013.1809692382812, 5740.19287109375, 11063.09375, 2246.991943359375, 11919.0888671875, 23206.140625, 3639.98486328125, 1424.4273681640625, 9930.193359375, 4223.85546875, 3139.369384765625, 9681.5751953125, 7742.55712890625, 8845.041015625, 6855.11328125, 4621.9658203125, 4705.71484375, 7421.2861328125, 11574.7470703125, 11613.8349609375, 3781.36865234375, 8246.166015625, 4958.99609375, 15392.8271484375, 5285.43212890625, 5542.67236328125, 6735.41357421875, 5376.28173828125, 6332.595703125, 6695.705078125, 4902.9306640625, 4715.91357421875, 4900.79931640625, 4948.498046875, 4374.2802734375, 174.71897888183594, 94.66676330566406, 90.16781616210938, 67.69003295898438, 66.50647735595703, 66.17338562011719, 308.16632080078125, 51.82996368408203, 65.00626373291016, 46.855247497558594, 50.70886993408203, 61.767478942871094, 42.98408889770508, 39.284305572509766, 43.365055084228516, 132.6004180908203, 39.047035217285156, 37.63555145263672, 39.776424407958984, 52.76180648803711, 45.358211517333984, 41.86042785644531, 37.91434860229492, 57.65824508666992, 48.640785217285156, 41.30146026611328, 31.897640228271484, 34.16583251953125, 28.14333152770996, 31.716421127319336, 62.36116027832031, 48.68559265136719, 376.8292541503906, 10060.4609375, 1878.2403564453125, 241.69813537597656, 3879.42138671875, 189.3050079345703, 204.519775390625, 95.12608337402344, 78.02507781982422, 347.8553466796875, 96.58506774902344, 70.44810485839844, 5633.37939453125, 568.6766967773438, 348.8977966308594, 137.6120147705078, 98.75125122070312, 1150.5863037109375, 3078.636962890625, 2008.23095703125, 487.3035888671875, 4444.10107421875, 250.02597045898438, 1080.29833984375, 1752.6109619140625, 2165.5947265625, 471.3336181640625, 7817.35595703125, 1648.9085693359375, 1058.177978515625, 488.53521728515625, 708.3407592773438, 662.138427734375, 498.2178955078125, 3486.61181640625, 7447.0537109375, 8889.98828125, 2966.816162109375, 3562.41357421875, 4009.6025390625, 7310.05859375, 1024.357177734375, 2944.184326171875, 5914.361328125, 2474.43017578125, 3410.994873046875, 5861.451171875, 3044.59130859375, 2989.113037109375, 2335.006591796875, 4105.91455078125, 3224.016357421875, 4734.591796875, 14614.935546875, 6714.44677734375, 5613.990234375, 5414.50830078125, 3375.608642578125, 5211.45458984375, 4201.63818359375, 3805.146728515625, 4475.517578125, 4710.001953125, 4430.53369140625, 5334.82568359375, 3963.288330078125, 4429.00927734375, 3568.494140625, 3693.4208984375, 1197.217529296875, 412.071533203125, 1106.645751953125, 547.92041015625, 264.5167541503906, 203.0034637451172, 150.9110107421875, 197.59085083007812, 129.2556610107422, 134.64686584472656, 120.83464050292969, 158.592041015625, 121.92082977294922, 419.96478271484375, 254.45272827148438, 171.76846313476562, 635.323974609375, 79.05984497070312, 158.1742706298828, 136.01698303222656, 715.0942993164062, 412.4376525878906, 87.66559600830078, 140.03555297851562, 79.1833267211914, 153.3829803466797, 60.73288345336914, 179.68727111816406, 54.93720626831055, 55.01005172729492, 1731.7183837890625, 390.7892150878906, 1963.1717529296875, 262.2866516113281, 491.7025146484375, 197.03945922851562, 2137.9111328125, 741.6022338867188, 1100.077880859375, 1697.253662109375, 475.4803771972656, 334.5870666503906, 1823.7933349609375, 824.789794921875, 1305.0439453125, 804.1922607421875, 856.9813232421875, 876.2633056640625, 643.4825439453125, 1959.801025390625, 1538.021240234375, 1115.6285400390625, 1409.11669921875, 2923.234130859375, 1031.0565185546875, 3121.84423828125, 5427.7265625, 4070.31640625, 4528.3173828125, 6297.1748046875, 2719.635009765625, 2441.83447265625, 7142.3740234375, 3609.408447265625, 2795.76025390625, 1538.255126953125, 1785.091552734375, 2674.99853515625, 10676.646484375, 3659.944580078125, 2428.10791015625, 2651.288330078125, 5459.8779296875, 8093.46484375, 7584.9013671875, 12576.5830078125, 6509.21240234375, 11551.6962890625, 16086.7177734375, 8030.548828125, 10846.7412109375, 6030.17529296875, 5976.26171875, 4150.888671875, 5728.37255859375, 5264.07763671875, 5080.3759765625, 3855.069091796875, 3945.146484375, 3187.816162109375, 3587.784423828125, 3823.645751953125, 3159.3779296875, 3283.76171875], "Term": ["place", "train", "room", "min", "walk", "bed", "queen", "new", "york", "bathroom", "bedroom", "bus", "station", "minute", "away", "large", "size", "fully", "city", "love", "high", "private", "full", "modern", "kitchen", "airport", "table", "best", "sofa", "dishwasher", "de", "el", "para", "un", "casino", "flushing", "medical", "driveway", "field", "se", "lo", "airport", "kosher", "do", "downstate", "lockable", "hay", "skate", "luna", "meadow", "drive", "expressway", "transfer", "amusement", "depot", "games!", "concourse", "cuadra", "hostel", "chase", "la", "lock", "hospital", "con", "tennis", "county", "bus", "herald", "min", "mall", "stadium", "female", "rite", "parking", "station", "car", "code", "supermarket", "minute", "train", "ride", "walk", "room", "guest", "express", "bathroom", "use", "safe", "private", "subway", "away", "available", "walking", "clean", "access", "kitchen", "bedroom", "distance", "living", "time", "apartment", "also", "two", "park", "street", "one", "bed", "close", "area", "neighborhood", "full", "quiet", "stock", "furry", "touristy", "recommendations!", "humble", "million", "coziness", "whilst", "constant", "cause", "grange", "world!", "bull", "output", "wherever", "place!", "obviously", "attitude", "fix", "population", "somewhat", "suggestions!", "tattoo", "chilled", "pace", "creativity", "worker", "hungry", "swan", "greeting", "tavern", "shy", "whether", "place", "solo", "outpost", "love", "seaport", "beer", "miss", "somewhere", "hustle", "minded", "jungle", "york", "culture", "bustle", "opportunity", "activity", "fun", "good", "want", "you!", "best", "got", "explore", "looking", "business", "city!", "city", "experience", "music", "artist", "spot", "exploring", "spend", "like", "great", "new", "feel", "perfect", "location", "neighborhood", "really", "get", "home", "village", "enjoy", "close", "right", "everything", "make", "need", "around", "stay", "apartment", "away", "space", "park", "cozy", "one", "street", "time", "subway", "train", "walk", "room", "access", "kitchen", "also", "bedroom", "stainless", "flooring", "steel", "granite", "housekeeping", "decorative", "oak", "burning", "cabinetry", "parquet", "reception", "tile", "mounted", "speaker", "oversized", "valet", "custom", "burner", "playroom", "soaking", "marble", "counter", "polished", "rain", "limestone", "backed", "velvet", "package", "configuration", "mahogany", "oven", "blender", "screen", "plush", "fireplace", "leather", "dishwasher", "concierge", "toaster", "hardwood", "sleeper", "rise", "flat", "gas", "maker", "memory", "foam", "fitness", "luxurious", "luxury", "stove", "tub", "lounge", "sofa", "wood", "table", "fully", "modern", "high", "queen", "unit", "microwave", "large", "furnished", "washer", "speed", "doorman", "mattress", "bed", "dining", "brand", "sized", "size", "full", "building", "bedroom", "floor", "kitchen", "apartment", "living", "room", "new", "space", "spacious", "private", "bathroom", "one", "beautiful", "two", "comfortable", "area", "park", "coffee", "access"], "Total": [11373.0, 16009.0, 39387.0, 8635.0, 18104.0, 19212.0, 8981.0, 16001.0, 7245.0, 16717.0, 27883.0, 4304.0, 6573.0, 7516.0, 16815.0, 10728.0, 9029.0, 7185.0, 11361.0, 4357.0, 5994.0, 17485.0, 14884.0, 5296.0, 27555.0, 2168.0, 3875.0, 5954.0, 3577.0, 2251.0, 930.3026733398438, 407.8857116699219, 222.93807983398438, 337.3912048339844, 159.47532653808594, 523.1298828125, 434.95684814453125, 119.93067932128906, 173.72557067871094, 185.6739044189453, 76.4495849609375, 2168.87841796875, 97.64007568359375, 57.60292053222656, 69.55667877197266, 82.90901947021484, 44.796199798583984, 65.40007781982422, 58.21120071411133, 42.1994514465332, 765.9310302734375, 53.191261291503906, 184.55914306640625, 53.02962112426758, 38.488304138183594, 32.53932571411133, 33.13912582397461, 29.87771224975586, 99.43990325927734, 88.61469268798828, 985.63525390625, 835.6905517578125, 478.8744201660156, 273.2679443359375, 213.4259490966797, 103.06150817871094, 4304.5087890625, 208.58035278320312, 8635.333984375, 760.5232543945312, 442.0738220214844, 509.5862121582031, 145.45252990722656, 2855.538330078125, 6573.9609375, 879.2304077148438, 289.12542724609375, 1185.416015625, 7516.02099609375, 16009.5791015625, 2871.04638671875, 18104.78125, 39387.70703125, 5237.4609375, 1863.340576171875, 16717.7578125, 6354.0986328125, 4675.79638671875, 17485.177734375, 14050.3720703125, 16815.39453125, 12516.708984375, 7880.5888671875, 8102.15234375, 14668.3359375, 27555.453125, 27883.83984375, 6676.2939453125, 19018.55078125, 9739.9423828125, 46094.48046875, 11058.6201171875, 11877.111328125, 15973.5673828125, 12134.8486328125, 16624.42578125, 19212.2421875, 11701.05859375, 11456.099609375, 14111.390625, 14884.5166015625, 9713.4541015625, 175.94354248046875, 95.51969146728516, 91.02051544189453, 68.66260528564453, 67.47074890136719, 67.15011596679688, 312.7469787597656, 52.607635498046875, 66.0335464477539, 47.61328887939453, 51.53756332397461, 62.81607437133789, 43.753021240234375, 40.011165618896484, 44.17666244506836, 135.23236083984375, 39.85691833496094, 38.424232482910156, 40.62696075439453, 53.93783187866211, 46.38296127319336, 42.84806823730469, 38.81800079345703, 59.0513916015625, 49.819942474365234, 42.30380630493164, 32.69151306152344, 35.01781463623047, 28.85399627685547, 32.523860931396484, 63.9570426940918, 50.028717041015625, 395.3313903808594, 11373.71875, 2053.660400390625, 253.85357666015625, 4357.5595703125, 199.9931640625, 216.63050842285156, 99.56255340576172, 81.22412872314453, 380.1169738769531, 101.7083511352539, 73.23741149902344, 7245.970703125, 649.140380859375, 391.0633850097656, 147.82211303710938, 104.32279205322266, 1383.8380126953125, 3991.98681640625, 2538.546875, 566.7376708984375, 5954.125, 279.1091003417969, 1331.2913818359375, 2247.001220703125, 2827.72607421875, 550.1160888671875, 11361.9775390625, 2155.834716796875, 1340.145263671875, 579.0900268554688, 879.0554809570312, 820.1925048828125, 599.6211547851562, 5290.25732421875, 12738.779296875, 16001.494140625, 4596.74365234375, 5690.70068359375, 6581.47119140625, 14111.390625, 1394.282958984375, 4906.76904296875, 11449.58203125, 4026.189697265625, 5991.4208984375, 11701.05859375, 5329.7509765625, 5230.9755859375, 3862.40966796875, 7867.84375, 5833.244140625, 9620.12109375, 46094.48046875, 16815.39453125, 15912.552734375, 15973.5673828125, 7111.6728515625, 16624.42578125, 12134.8486328125, 9739.9423828125, 14050.3720703125, 16009.5791015625, 18104.78125, 39387.70703125, 14668.3359375, 27555.453125, 11058.6201171875, 27883.83984375, 1199.4168701171875, 412.9025573730469, 1109.5823974609375, 549.3824462890625, 265.350341796875, 203.90426635742188, 151.67420959472656, 198.65223693847656, 129.95333862304688, 135.4066162109375, 121.58180236816406, 159.5946044921875, 122.73568725585938, 422.9256591796875, 256.29473876953125, 173.1279296875, 641.0120849609375, 79.79806518554688, 159.71209716796875, 137.34747314453125, 722.5007934570312, 416.8692321777344, 88.66400909423828, 141.69810485839844, 80.13406372070312, 155.2818145751953, 61.4886474609375, 181.947021484375, 55.62945556640625, 55.710723876953125, 1769.830322265625, 395.94464111328125, 2024.9033203125, 265.82232666015625, 503.1634521484375, 200.0363311767578, 2251.1826171875, 764.657958984375, 1148.850830078125, 1814.4378662109375, 493.9892578125, 344.13134765625, 2010.33837890625, 888.1825561523438, 1441.6741943359375, 869.8560180664062, 930.2045288085938, 953.8729858398438, 692.0087280273438, 2241.002197265625, 1747.0760498046875, 1243.7960205078125, 1614.700927734375, 3577.9384765625, 1154.7457275390625, 3875.05517578125, 7185.71875, 5296.166015625, 5994.45556640625, 8981.3291015625, 3475.266845703125, 3110.330810546875, 10728.5849609375, 4953.775390625, 3696.542724609375, 1840.6650390625, 2220.86279296875, 3613.9443359375, 19212.2421875, 5317.47900390625, 3235.47998046875, 3643.912841796875, 9029.380859375, 14884.5166015625, 14290.9130859375, 27883.83984375, 11883.748046875, 27555.453125, 46094.48046875, 19018.55078125, 39387.70703125, 16001.494140625, 15912.552734375, 8538.890625, 17485.177734375, 16717.7578125, 16624.42578125, 9264.8779296875, 11877.111328125, 7135.78076171875, 11456.099609375, 15973.5673828125, 7267.580078125, 14668.3359375], "loglift": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 1.0722999572753906, 1.0703999996185303, 1.0700000524520874, 1.0688999891281128, 1.0683000087738037, 1.0670000314712524, 1.065600037574768, 1.065000057220459, 1.0643999576568604, 1.0637999773025513, 1.0607000589370728, 1.0605000257492065, 1.0605000257492065, 1.058899998664856, 1.0585999488830566, 1.0573999881744385, 1.0570000410079956, 1.0561000108718872, 1.0555000305175781, 1.0541000366210938, 1.05239999294281, 1.0521999597549438, 1.0494999885559082, 1.049399971961975, 1.0490000247955322, 1.048699975013733, 1.04830002784729, 1.047700047492981, 1.0473999977111816, 1.0473999977111816, 1.0460000038146973, 1.0356999635696411, 1.0406999588012695, 1.0377999544143677, 1.0396000146865845, 1.0461000204086304, 0.9693999886512756, 1.0276000499725342, 0.939300000667572, 0.9994999766349792, 1.0118999481201172, 0.9987000226974487, 1.034600019454956, 0.8981000185012817, 0.8531000018119812, 0.9520000219345093, 0.9993000030517578, 0.9161999821662903, 0.803600013256073, 0.7035999894142151, 0.8281000256538391, 0.6552000045776367, 0.5442000031471252, 0.7092999815940857, 0.8046000003814697, 0.552299976348877, 0.6647999882698059, 0.6747999787330627, 0.4821000099182129, 0.4772999882698059, 0.4307999908924103, 0.47110000252723694, 0.5396000146865845, 0.5297999978065491, 0.3919000029563904, 0.20579999685287476, 0.1973000019788742, 0.5047000050544739, 0.23749999701976776, 0.39820000529289246, -0.023600000888109207, 0.33489999175071716, 0.3111000061035156, 0.20960000157356262, 0.2590999901294708, 0.1080000028014183, 0.019099999219179153, 0.20329999923706055, 0.18559999763965607, 0.015599999576807022, -0.02800000086426735, 0.2754000127315521, 1.0856000185012817, 1.0836000442504883, 1.0831999778747559, 1.0784000158309937, 1.0781999826431274, 1.0779999494552612, 1.0779000520706177, 1.0777000188827515, 1.0769000053405762, 1.0765999555587769, 1.0764000415802002, 1.0757999420166016, 1.0749000310897827, 1.0743000507354736, 1.0741000175476074, 1.0729999542236328, 1.072100043296814, 1.0719000101089478, 1.0714999437332153, 1.0706000328063965, 1.0702999830245972, 1.0693000555038452, 1.069100022315979, 1.0686999559402466, 1.0686999559402466, 1.068600058555603, 1.0679999589920044, 1.0679999589920044, 1.0677000284194946, 1.0674999952316284, 1.0672999620437622, 1.0654000043869019, 1.044700026512146, 0.9699000120162964, 1.0032999515533447, 1.0435999631881714, 0.9764000177383423, 1.0377000570297241, 1.035099983215332, 1.0470000505447388, 1.05239999294281, 1.0039000511169434, 1.0408999919891357, 1.0537999868392944, 0.8409000039100647, 0.9603000283241272, 0.9785000085830688, 1.0210000276565552, 1.0377000570297241, 0.9079999923706055, 0.8327999711036682, 0.858299970626831, 0.9416000247001648, 0.8001000285148621, 0.9825999736785889, 0.8837000131607056, 0.8440999984741211, 0.8258000016212463, 0.9380999803543091, 0.7186999917030334, 0.8245999813079834, 0.8564000129699707, 0.9225999712944031, 0.8766999840736389, 0.878600001335144, 0.9074000120162964, 0.6757000088691711, 0.5558000206947327, 0.5048999786376953, 0.6547999978065491, 0.6241999864578247, 0.597100019454956, 0.4348999857902527, 0.7843000292778015, 0.5817999839782715, 0.4320000112056732, 0.6057999730110168, 0.5292999744415283, 0.40130001306533813, 0.5327000021934509, 0.5329999923706055, 0.5892999768257141, 0.4422999918460846, 0.49970000982284546, 0.38370001316070557, -0.0560000017285347, 0.1746000051498413, 0.05079999938607216, 0.01080000028014183, 0.3474999964237213, -0.0674000009894371, 0.03200000151991844, 0.1527000069618225, -0.05139999836683273, -0.13089999556541443, -0.3149999976158142, -0.9065999984741211, -0.2160000056028366, -0.7354000210762024, -0.03840000182390213, -0.9289000034332275, 1.128999948501587, 1.1289000511169434, 1.1282000541687012, 1.1282000541687012, 1.1276999711990356, 1.1265000104904175, 1.1258000135421753, 1.125499963760376, 1.125499963760376, 1.1253000497817993, 1.1246999502182007, 1.1246000528335571, 1.1241999864578247, 1.123900055885315, 1.1237000226974487, 1.1230000257492065, 1.121999979019165, 1.1216000318527222, 1.1211999654769897, 1.1211999654769897, 1.1205999851226807, 1.1202000379562378, 1.1196000576019287, 1.1190999746322632, 1.11899995803833, 1.1186000108718872, 1.118499994277954, 1.118399977684021, 1.118399977684021, 1.1181999444961548, 1.1090999841690063, 1.117799997329712, 1.0999000072479248, 1.1174999475479126, 1.107800006866455, 1.1158000230789185, 1.079300045967102, 1.1002999544143677, 1.087499976158142, 1.0641000270843506, 1.0927000045776367, 1.1028000116348267, 1.0334999561309814, 1.0568000078201294, 1.0312999486923218, 1.05239999294281, 1.0489000082015991, 1.0460000038146973, 1.0582000017166138, 0.9968000054359436, 1.0033999681472778, 1.0220999717712402, 0.994700014591217, 0.9287999868392944, 1.0176000595092773, 0.9146999716758728, 0.8503000140190125, 0.8676000237464905, 0.8503999710083008, 0.7757999897003174, 0.885699987411499, 0.8888999819755554, 0.7239999771118164, 0.814300000667572, 0.8515999913215637, 0.9513999819755554, 0.9125000238418579, 0.8299999833106995, 0.54339998960495, 0.7573000192642212, 0.8438000082969666, 0.8129000067710876, 0.6277999877929688, 0.5216000080108643, 0.4973999857902527, 0.33469998836517334, 0.5289000272750854, 0.2615000009536743, 0.07819999754428864, 0.2687000036239624, -0.15870000422000885, 0.1550000011920929, 0.15160000324249268, 0.40959998965263367, 0.014999999664723873, -0.024700000882148743, -0.05460000038146973, 0.2540000081062317, 0.02879999950528145, 0.32510000467300415, -0.03009999915957451, -0.298799991607666, 0.2978000044822693, -0.36579999327659607], "logprob": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, -6.621699810028076, -7.4481000900268555, -8.05270004272461, -7.6392998695373535, -8.389300346374512, -7.202700138092041, -7.388700008392334, -8.677599906921387, -8.307600021362305, -8.241700172424316, -9.132200241088867, -5.7870001792907715, -8.887700080871582, -9.417099952697754, -9.228799819946289, -9.054300308227539, -9.67039966583252, -9.292900085449219, -9.409899711608887, -9.732999801635742, -6.835999965667725, -9.503399848937988, -8.26200008392334, -9.509200096130371, -9.830100059509277, -9.998299598693848, -9.980500221252441, -10.084699630737305, -8.882499694824219, -8.99779987335205, -6.590199947357178, -6.765600204467773, -7.317399978637695, -7.881199836730957, -8.126700401306152, -8.848099708557129, -5.192699909210205, -8.161600112915039, -4.526599884033203, -6.895999908447266, -7.42609977722168, -7.2972002029418945, -8.515000343322754, -5.6743998527526855, -4.885499954223633, -6.798399925231934, -7.863399982452393, -6.5355000495910645, -4.80109977722168, -4.144899845123291, -5.738999843597412, -4.070400238037109, -3.404099941253662, -5.2565999031066895, -6.194799900054932, -4.252999782562256, -5.107800006866455, -5.4045000076293945, -4.278299808502197, -4.501800060272217, -4.36870002746582, -4.623600006103516, -5.0177001953125, -4.999800205230713, -4.5441999435424805, -4.099699974060059, -4.096399784088135, -5.218500137329102, -4.438799858093262, -4.947400093078613, -3.81469988822937, -4.883600234985352, -4.836100101470947, -4.641200065612793, -4.866600036621094, -4.702899932861328, -4.64709997177124, -4.958700180053711, -4.997600078582764, -4.959199905395508, -4.94950008392334, -5.072800159454346, -8.273699760437012, -8.886500358581543, -8.935199737548828, -9.222000122070312, -9.23960018157959, -9.244600296020508, -7.706299781799316, -9.488900184631348, -9.262399673461914, -9.589799880981445, -9.5108003616333, -9.31350040435791, -9.67609977722168, -9.76609992980957, -9.667200088500977, -8.549599647521973, -9.772100448608398, -9.808899879455566, -9.753600120544434, -9.471099853515625, -9.622300148010254, -9.70259952545166, -9.801600456237793, -9.382399559020996, -9.552399635314941, -9.715999603271484, -9.97439956665039, -9.905699729919434, -10.099599838256836, -9.98009967803955, -9.303999900817871, -9.55150032043457, -7.505099773406982, -4.2204999923706055, -5.898799896240234, -7.94920015335083, -5.173500061035156, -8.193499565124512, -8.11620044708252, -8.881699562072754, -9.079899787902832, -7.585100173950195, -8.866499900817871, -9.182000160217285, -4.8003997802734375, -7.093599796295166, -7.582099914550781, -8.512499809265137, -8.844300270080566, -6.388899803161621, -5.404699802398682, -5.831900119781494, -7.248000144958496, -5.037600040435791, -7.915299892425537, -6.451900005340576, -5.9679999351501465, -5.756400108337402, -7.281300067901611, -4.472799777984619, -6.0289998054504395, -6.472599983215332, -7.245500087738037, -6.874000072479248, -6.941400051116943, -7.225900173187256, -5.280200004577637, -4.521299839019775, -4.344200134277344, -5.4415998458862305, -5.258699893951416, -5.140399932861328, -4.539899826049805, -6.505099773406982, -5.4492998123168945, -4.751800060272217, -5.6230998039245605, -5.30210018157959, -4.760700225830078, -5.415800094604492, -5.434199810028076, -5.681099891662598, -5.116700172424316, -5.358500003814697, -4.9741997718811035, -3.847100019454956, -4.624899864196777, -4.803899765014648, -4.840099811553955, -5.312600135803223, -4.878300189971924, -5.093699932098389, -5.192800045013428, -5.0304999351501465, -4.979400157928467, -5.040599822998047, -4.854899883270264, -5.152100086212158, -5.040999889373779, -5.256999969482422, -5.222599983215332, -6.3109002113342285, -7.377399921417236, -6.389500141143799, -7.09250020980835, -7.820700168609619, -8.085399627685547, -8.3818998336792, -8.11240005493164, -8.536800384521484, -8.496000289916992, -8.60420036315918, -8.332300186157227, -8.59529972076416, -7.358500003814697, -7.859499931335449, -8.2524995803833, -6.944499969482422, -9.028400421142578, -8.33489990234375, -8.485799789428711, -6.826200008392334, -7.376500129699707, -8.925100326538086, -8.456700325012207, -9.026900291442871, -8.365699768066406, -9.292099952697754, -8.20740032196045, -9.392399787902832, -9.39109992980957, -5.941800117492676, -7.430500030517578, -5.816299915313721, -7.82919979095459, -7.200799942016602, -8.11520004272461, -5.730999946594238, -6.78980016708374, -6.395500183105469, -5.961900234222412, -7.234300136566162, -7.585700035095215, -5.889999866485596, -6.683499813079834, -6.224599838256836, -6.708799839019775, -6.645199775695801, -6.623000144958496, -6.931700229644775, -5.817999839782715, -6.060400009155273, -6.381499767303467, -6.147900104522705, -5.4182000160217285, -6.460299968719482, -5.352399826049805, -4.799300193786621, -5.087100028991699, -4.980500221252441, -4.6508002281188965, -5.4903998374938965, -5.598100185394287, -4.524799823760986, -5.207300186157227, -5.462800025939941, -6.060200214385986, -5.911399841308594, -5.506899833679199, -4.122799873352051, -5.193399906158447, -5.603799819946289, -5.5157999992370605, -4.793399810791016, -4.399799823760986, -4.464700222015381, -3.9590001106262207, -4.617700099945068, -4.044000148773193, -3.712899923324585, -4.407599925994873, -4.10699987411499, -4.6940999031066895, -4.703100204467773, -5.067500114440918, -4.7453999519348145, -4.829999923706055, -4.865499973297119, -5.141499996185303, -5.1184000968933105, -5.331500053405762, -5.2133002281188965, -5.149700164794922, -5.3404998779296875, -5.3018999099731445]}, "token.table": {"Topic": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 1, 2, 3, 1, 2, 3, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 3, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 3, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 3, 1, 2, 3, 1, 3, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 3, 1, 1, 2, 3, 2, 3, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 2, 1, 2, 3, 1, 2, 3, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 1, 2, 3, 1, 2, 3, 2, 1, 2, 3, 1, 2, 3, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 3, 2, 1, 2, 3, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 3, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 3, 2, 1, 2, 3, 1, 2, 3, 2, 3, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 2, 3, 1, 2, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 3, 1, 2, 3, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 3, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 1, 2, 3, 2, 3, 1, 2, 3, 1, 2, 3, 2, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3], "Freq": [0.5059196949005127, 0.2701737880706787, 0.22388361394405365, 0.03834253177046776, 0.9489776492118835, 0.00958563294261694, 0.9876072406768799, 0.012448830530047417, 0.0004610678006429225, 0.4779077172279358, 0.3226442337036133, 0.19939196109771729, 0.9805840253829956, 0.018857385963201523, 0.018857385963201523, 0.33394452929496765, 0.31706616282463074, 0.34900057315826416, 0.4116584360599518, 0.2751372754573822, 0.3131955862045288, 0.4090348184108734, 0.5526941418647766, 0.03840058669447899, 0.022449014708399773, 0.844428300857544, 0.13469408452510834, 0.9889592528343201, 0.547667920589447, 0.22921361029148102, 0.22314171493053436, 0.5260061025619507, 0.3992769718170166, 0.074693463742733, 0.012879808433353901, 0.9853053092956543, 0.5939791798591614, 0.09110073000192642, 0.3148747682571411, 0.2360527664422989, 0.3477649688720703, 0.4160875082015991, 0.3485277593135834, 0.09577226638793945, 0.5557394027709961, 0.4165136516094208, 0.13244231045246124, 0.4510497748851776, 0.032313086092472076, 0.9463117718696594, 0.023080775514245033, 0.105473093688488, 0.7463732957839966, 0.1481325924396515, 0.007576816715300083, 0.00505121098831296, 0.9875118136405945, 0.059342045336961746, 0.19038906693458557, 0.750429630279541, 0.27003172039985657, 0.19921749830245972, 0.5307568311691284, 0.9827892780303955, 0.9899989366531372, 0.0050339228473603725, 0.9967166781425476, 0.9013804197311401, 0.09362275898456573, 0.005110919941216707, 0.08452020585536957, 0.7659865021705627, 0.1495901644229889, 0.023014172911643982, 0.8924384713172913, 0.08438529819250107, 0.9926639795303345, 0.8860021233558655, 0.1125984713435173, 0.001137358252890408, 0.9970194101333618, 0.9871193766593933, 0.9704936742782593, 0.011284810490906239, 0.011284810490906239, 0.016934402287006378, 0.9821953177452087, 0.016934402287006378, 0.13650792837142944, 0.6879964470863342, 0.1754976212978363, 0.06544072926044464, 0.8561829328536987, 0.07634752243757248, 0.5808333158493042, 0.2528957724571228, 0.1663755476474762, 0.4190219044685364, 0.5008948445320129, 0.08007822185754776, 0.9303920269012451, 0.048421889543533325, 0.024210944771766663, 0.2529039978981018, 0.31234607100486755, 0.4346701204776764, 0.28518253564834595, 0.2680855989456177, 0.4467625916004181, 0.9660847783088684, 0.032934706658124924, 0.0013077742187306285, 0.028771033510565758, 0.970368504524231, 0.9656259417533875, 0.9886848330497742, 0.9843481779098511, 0.015143818221986294, 0.002398833865299821, 0.007196501363068819, 0.9883195161819458, 0.9702943563461304, 0.01940588653087616, 0.00970294326543808, 0.003197473008185625, 0.9848216772079468, 0.0127898920327425, 0.3959687352180481, 0.4747124910354614, 0.1293647736310959, 0.9691799283027649, 0.023638535290956497, 0.9706231951713562, 0.04159346967935562, 0.8765438199043274, 0.08318693935871124, 0.007800165098160505, 0.9906209707260132, 0.9985997080802917, 0.004904262255877256, 0.9955652356147766, 0.987312912940979, 0.025981919839978218, 0.23356932401657104, 0.07804450392723083, 0.6882960796356201, 0.034648455679416656, 0.015547383576631546, 0.9497230648994446, 0.5663321614265442, 0.323682576417923, 0.10994123667478561, 0.9895331859588623, 0.15129254758358002, 0.04502754658460617, 0.8037416934967041, 0.9919967651367188, 0.014376764185726643, 0.9792004227638245, 0.019584009423851967, 0.0013056006282567978, 0.9922398328781128, 0.008338149636983871, 0.9978285431861877, 0.002451667096465826, 0.10214605182409286, 0.5693140625953674, 0.3284696638584137, 0.2028302401304245, 0.5714039206504822, 0.225770503282547, 0.012988008558750153, 0.7649009227752686, 0.22218771278858185, 0.06084318086504936, 0.811242401599884, 0.12769556045532227, 0.057303618639707565, 0.8071275949478149, 0.1353340893983841, 0.7642188668251038, 0.20608149468898773, 0.029516879469156265, 0.9776042103767395, 0.018800079822540283, 0.19231005012989044, 0.6454569101333618, 0.16228879988193512, 0.9282040596008301, 0.06868317723274231, 0.0019623765256255865, 0.9900672435760498, 0.0057562049478292465, 0.0039748516865074635, 0.01987425796687603, 0.9778134822845459, 0.03249908611178398, 0.0492728091776371, 0.918361246585846, 0.9845678806304932, 0.05024029687047005, 0.0427788682281971, 0.9073099493980408, 0.3256548345088959, 0.12655939161777496, 0.5477228164672852, 0.9978141188621521, 0.9940170049667358, 0.005734713282436132, 0.051601555198431015, 0.02687581069767475, 0.9213027358055115, 0.332425981760025, 0.12381993979215622, 0.5437193512916565, 0.14097407460212708, 0.10367786884307861, 0.7553871870040894, 0.09827739745378494, 0.8317447304725647, 0.07009490579366684, 0.19237852096557617, 0.07892969995737076, 0.7285352349281311, 0.9945593476295471, 0.9834254384040833, 0.04503578692674637, 0.025895576924085617, 0.9288631081581116, 0.3694895803928375, 0.5999875068664551, 0.030570013448596, 0.19338741898536682, 0.7712951302528381, 0.03557125851511955, 0.014331313781440258, 0.8957070708274841, 0.08957070857286453, 0.9895694851875305, 0.0018202256178483367, 0.0018202256178483367, 0.9974836111068726, 0.2366003841161728, 0.584592878818512, 0.17874553799629211, 0.9838930368423462, 0.694993257522583, 0.13307975232601166, 0.17183898389339447, 0.015982910990715027, 0.04849987104535103, 0.9352759122848511, 0.9822261929512024, 0.9540687799453735, 0.03356020897626877, 0.009588630869984627, 0.04871167987585068, 0.19584764540195465, 0.7553646564483643, 0.2432403266429901, 0.516525387763977, 0.24018344283103943, 0.9689387679100037, 0.029235221445560455, 0.0020882301032543182, 0.9754635691642761, 0.020112650468945503, 0.9986796975135803, 0.9930229187011719, 0.01482123788446188, 0.9709343910217285, 0.021046152338385582, 0.9155076742172241, 0.0631384626030922, 0.9557956457138062, 0.027308447286486626, 0.42006203532218933, 0.16073043644428253, 0.41922736167907715, 0.9832028150558472, 0.010241696611046791, 0.010241696611046791, 0.972976565361023, 0.026378925889730453, 0.2268705517053604, 0.10737669467926025, 0.665698230266571, 0.004999091848731041, 0.009998183697462082, 0.984821081161499, 0.15708120167255402, 0.659136176109314, 0.18392300605773926, 0.012479087337851524, 0.9858478903770447, 0.43357667326927185, 0.14417502284049988, 0.4222719371318817, 0.9941191673278809, 0.013080515898764133, 0.22897616028785706, 0.6092862486839294, 0.16181792318820953, 0.9632751941680908, 0.013162766583263874, 0.023932304233312607, 0.989035964012146, 0.012061414308845997, 0.1246105208992958, 0.7801508903503418, 0.09523804485797882, 0.06193097308278084, 0.06564683467149734, 0.872607409954071, 0.051863893866539, 0.8901771306991577, 0.058060020208358765, 0.9791929721832275, 0.017178824171423912, 0.023121096193790436, 0.04768725857138634, 0.9291790127754211, 0.052208784967660904, 0.07318154722452164, 0.8746086955070496, 0.987242579460144, 0.17838604748249054, 0.6045448780059814, 0.2172219157218933, 0.0908665806055069, 0.003468190087005496, 0.9051975607872009, 0.9283082485198975, 0.0710037425160408, 0.0013840815518051386, 0.009688571095466614, 0.9896183013916016, 0.19369418919086456, 0.06613273173570633, 0.740188479423523, 0.9715766310691833, 0.9932019710540771, 0.006897235754877329, 0.0022990787401795387, 0.04713423550128937, 0.02759077399969101, 0.9242908954620361, 0.20865947008132935, 0.006430184002965689, 0.7851254940032959, 0.982872486114502, 0.01489200722426176, 0.8746621608734131, 0.11823514848947525, 0.007064000237733126, 0.039328135550022125, 0.9537073373794556, 0.009832033887505531, 0.7637019753456116, 0.19465087354183197, 0.0416443757712841, 0.010043936781585217, 0.9541739821434021, 0.030131811276078224, 0.08458949625492096, 0.14689871668815613, 0.7684804201126099, 0.9940059185028076, 0.04775601625442505, 0.7894666790962219, 0.16266892850399017, 0.3052932918071747, 0.5218710899353027, 0.17285549640655518, 0.34730806946754456, 0.5180212259292603, 0.13471387326717377, 0.06755619496107101, 0.5555731058120728, 0.376839816570282, 0.9955548644065857, 0.9785001277923584, 0.38094547390937805, 0.31345444917678833, 0.3055744767189026, 0.00676488783210516, 0.9335545301437378, 0.06764888018369675, 0.9533054828643799, 0.04727134481072426, 0.9747279286384583, 0.01751580275595188, 0.00395518122240901, 0.9786248803138733, 0.0039017577655613422, 0.0039017577655613422, 0.9910464882850647, 0.02007228322327137, 0.9835419058799744, 0.02007228322327137, 0.005496105179190636, 0.005496105179190636, 0.989298939704895, 0.9957922101020813, 0.4216340482234955, 0.3389975428581238, 0.23939548432826996, 0.8394213914871216, 0.10225742310285568, 0.05848284438252449, 0.996997058391571, 0.1212504431605339, 0.6259334683418274, 0.25269296765327454, 0.10023106634616852, 0.8844952583312988, 0.015298426151275635, 0.007394679822027683, 0.9834924340248108, 0.014789359644055367, 0.006261266302317381, 0.989280104637146, 0.007523822598159313, 0.007523822598159313, 0.9856207370758057, 0.01127853337675333, 0.992510974407196, 0.01853986270725727, 0.9826127290725708, 0.5537261366844177, 0.11867193877696991, 0.32759174704551697, 0.2644374668598175, 0.034404706209897995, 0.7011211514472961, 0.45030325651168823, 0.3397349715232849, 0.20991502702236176, 0.00705725746229291, 0.00705725746229291, 0.9880160093307495, 0.21086107194423676, 0.7344276905059814, 0.05379109084606171, 0.9952147006988525, 0.014563968405127525, 0.9903498291969299, 0.7826414704322815, 0.19748897850513458, 0.019853388890624046, 0.3002016544342041, 0.5713212490081787, 0.12871144711971283, 0.00871760118752718, 0.020341070368885994, 0.9734655022621155, 0.9625133275985718, 0.03437547758221626, 0.5891686081886292, 0.13544835150241852, 0.27539050579071045, 0.6713294982910156, 0.24295327067375183, 0.08576079457998276, 0.017284775152802467, 0.013333969749510288, 0.9694290161132812, 0.9909847378730774, 0.005385786294937134, 0.005385786294937134, 0.04500153660774231, 0.9450322985649109, 0.010000341571867466, 0.019988520070910454, 0.9794374704360962, 0.36159732937812805, 0.03366786614060402, 0.6046926379203796, 0.22009308636188507, 0.05214175209403038, 0.7275146842002869, 0.9785920977592468, 0.015290501527488232, 0.028340697288513184, 0.010121678002178669, 0.9615593552589417, 0.0072808037512004375, 0.9901893138885498, 0.15344031155109406, 0.029625998809933662, 0.8169508576393127, 0.0769357979297638, 0.9144647121429443, 0.008277902379631996, 0.021559640765190125, 0.9701838493347168, 0.012311612255871296, 0.9603057503700256, 0.03693483769893646, 0.27160945534706116, 0.35280323028564453, 0.37555256485939026, 0.2823551893234253, 0.23152890801429749, 0.48612871766090393, 0.0023644817993044853, 0.004728963598608971, 0.9930823445320129, 0.12278170883655548, 0.041832707822322845, 0.8355675339698792, 0.14008845388889313, 0.8305243849754333, 0.03001895360648632, 0.1535739302635193, 0.8054099082946777, 0.03981546312570572, 0.9410192966461182, 0.058813706040382385, 0.0008337384788319468, 0.0008337384788319468, 0.9979849457740784, 0.8024081587791443, 0.15972106158733368, 0.03772459179162979, 0.24843762814998627, 0.49219754338264465, 0.25945618748664856, 0.0009012399823404849, 0.0018024799646809697, 0.9976726174354553, 0.005683641415089369, 0.9946372509002686, 0.005683641415089369, 0.09673305600881577, 0.02289539761841297, 0.8803279995918274, 0.44302159547805786, 0.3462754487991333, 0.21071544289588928, 0.5510886311531067, 0.31856808066368103, 0.13038800656795502, 0.023338275030255318, 0.9802075624465942, 0.8545523285865784, 0.1349737048149109, 0.010123028419911861, 0.9704028367996216, 0.16102996468544006, 0.03328984975814819, 0.8056659698486328, 0.9789272546768188, 0.015635494142770767, 0.9694006443023682, 0.9652059674263, 0.009370931424200535, 0.0234273299574852, 0.00626587588340044, 0.9962742924690247, 0.5091406106948853, 0.3906593918800354, 0.1002059280872345, 0.040910445153713226, 0.0017408700659871101, 0.9574785232543945, 0.9887880682945251, 0.6910237669944763, 0.2941988706588745, 0.014741174876689911, 0.9752970933914185, 0.02167326956987381, 0.08763495087623596, 0.015275816433131695, 0.8972532153129578, 0.46669596433639526, 0.20114319026470184, 0.33215147256851196, 0.9958766102790833, 0.0029639184940606356, 0.0029639184940606356, 0.17092213034629822, 0.0466151237487793, 0.782673716545105, 0.6647678017616272, 0.2041202187538147, 0.1310964822769165, 0.005776075646281242, 0.005776075646281242, 0.9934849739074707, 0.9920530319213867, 0.13536371290683746, 0.61447674036026, 0.2501124143600464, 0.6583343744277954, 0.24474197626113892, 0.09693571925163269, 0.5865043997764587, 0.3408374786376953, 0.0727103054523468, 0.1512676477432251, 0.7910037040710449, 0.057513218373060226, 0.1817914843559265, 0.061679255217313766, 0.7563824653625488, 0.9733645915985107, 0.010118093341588974, 0.9536303281784058, 0.03794284909963608, 0.9884496927261353, 0.017319830134510994, 0.09006311744451523, 0.8928372263908386, 0.978847324848175, 0.01591949164867401, 0.9870085120201111, 0.01591949164867401, 0.039332203567028046, 0.7773975729942322, 0.18313626945018768, 0.10234011709690094, 0.8593040704727173, 0.03705418109893799], "Term": ["access", "access", "access", "activity", "activity", "activity", "airport", "airport", "airport", "also", "also", "also", "amusement", "amusement", "amusement", "apartment", "apartment", "apartment", "area", "area", "area", "around", "around", "around", "artist", "artist", "artist", "attitude", "available", "available", "available", "away", "away", "away", "backed", "backed", "bathroom", "bathroom", "bathroom", "beautiful", "beautiful", "beautiful", "bed", "bed", "bed", "bedroom", "bedroom", "bedroom", "beer", "beer", "beer", "best", "best", "best", "blender", "blender", "blender", "brand", "brand", "brand", "building", "building", "building", "bull", "burner", "burning", "burning", "bus", "bus", "bus", "business", "business", "business", "bustle", "bustle", "bustle", "cabinetry", "car", "car", "car", "casino", "cause", "chase", "chase", "chase", "chilled", "chilled", "chilled", "city", "city", "city", "city!", "city!", "city!", "clean", "clean", "clean", "close", "close", "close", "code", "code", "code", "coffee", "coffee", "coffee", "comfortable", "comfortable", "comfortable", "con", "con", "concierge", "concierge", "concierge", "concourse", "configuration", "constant", "constant", "counter", "counter", "counter", "county", "county", "county", "coziness", "coziness", "coziness", "cozy", "cozy", "cozy", "creativity", "creativity", "cuadra", "culture", "culture", "culture", "custom", "custom", "de", "decorative", "decorative", "depot", "depot", "dining", "dining", "dining", "dishwasher", "dishwasher", "dishwasher", "distance", "distance", "distance", "do", "doorman", "doorman", "doorman", "downstate", "downstate", "drive", "drive", "drive", "driveway", "driveway", "el", "el", "enjoy", "enjoy", "enjoy", "everything", "everything", "everything", "experience", "experience", "experience", "explore", "explore", "explore", "exploring", "exploring", "exploring", "express", "express", "express", "expressway", "expressway", "feel", "feel", "feel", "female", "female", "female", "field", "field", "fireplace", "fireplace", "fireplace", "fitness", "fitness", "fitness", "fix", "flat", "flat", "flat", "floor", "floor", "floor", "flooring", "flushing", "flushing", "foam", "foam", "foam", "full", "full", "full", "fully", "fully", "fully", "fun", "fun", "fun", "furnished", "furnished", "furnished", "furry", "games!", "gas", "gas", "gas", "get", "get", "get", "good", "good", "good", "got", "got", "got", "grange", "granite", "granite", "granite", "great", "great", "great", "greeting", "guest", "guest", "guest", "hardwood", "hardwood", "hardwood", "hay", "herald", "herald", "herald", "high", "high", "high", "home", "home", "home", "hospital", "hospital", "hospital", "hostel", "hostel", "housekeeping", "humble", "humble", "hungry", "hustle", "hustle", "hustle", "jungle", "jungle", "kitchen", "kitchen", "kitchen", "kosher", "kosher", "kosher", "la", "la", "large", "large", "large", "leather", "leather", "leather", "like", "like", "like", "limestone", "limestone", "living", "living", "living", "lo", "lo", "location", "location", "location", "lock", "lock", "lock", "lockable", "lockable", "looking", "looking", "looking", "lounge", "lounge", "lounge", "love", "love", "love", "luna", "luna", "luxurious", "luxurious", "luxurious", "luxury", "luxury", "luxury", "mahogany", "make", "make", "make", "maker", "maker", "maker", "mall", "mall", "marble", "marble", "marble", "mattress", "mattress", "mattress", "meadow", "medical", "medical", "medical", "memory", "memory", "memory", "microwave", "microwave", "microwave", "million", "million", "min", "min", "min", "minded", "minded", "minded", "minute", "minute", "minute", "miss", "miss", "miss", "modern", "modern", "modern", "mounted", "music", "music", "music", "need", "need", "need", "neighborhood", "neighborhood", "neighborhood", "new", "new", "new", "oak", "obviously", "one", "one", "one", "opportunity", "opportunity", "opportunity", "outpost", "outpost", "output", "oven", "oven", "oven", "oversized", "oversized", "oversized", "pace", "pace", "pace", "package", "package", "package", "para", "park", "park", "park", "parking", "parking", "parking", "parquet", "perfect", "perfect", "perfect", "place", "place", "place", "place!", "place!", "place!", "playroom", "playroom", "plush", "plush", "plush", "polished", "polished", "population", "population", "private", "private", "private", "queen", "queen", "queen", "quiet", "quiet", "quiet", "rain", "rain", "rain", "really", "really", "really", "reception", "recommendations!", "recommendations!", "ride", "ride", "ride", "right", "right", "right", "rise", "rise", "rise", "rite", "rite", "room", "room", "room", "safe", "safe", "safe", "screen", "screen", "screen", "se", "se", "se", "seaport", "seaport", "seaport", "shy", "shy", "size", "size", "size", "sized", "sized", "sized", "skate", "skate", "sleeper", "sleeper", "sleeper", "soaking", "soaking", "sofa", "sofa", "sofa", "solo", "solo", "solo", "somewhat", "somewhat", "somewhere", "somewhere", "somewhere", "space", "space", "space", "spacious", "spacious", "spacious", "speaker", "speaker", "speaker", "speed", "speed", "speed", "spend", "spend", "spend", "spot", "spot", "spot", "stadium", "stadium", "stainless", "stainless", "stainless", "station", "station", "station", "stay", "stay", "stay", "steel", "steel", "steel", "stock", "stock", "stock", "stove", "stove", "stove", "street", "street", "street", "subway", "subway", "subway", "suggestions!", "suggestions!", "supermarket", "supermarket", "supermarket", "swan", "table", "table", "table", "tattoo", "tavern", "tavern", "tennis", "tennis", "tennis", "tile", "tile", "time", "time", "time", "toaster", "toaster", "toaster", "touristy", "train", "train", "train", "transfer", "transfer", "tub", "tub", "tub", "two", "two", "two", "un", "un", "un", "unit", "unit", "unit", "use", "use", "use", "valet", "valet", "valet", "velvet", "village", "village", "village", "walk", "walk", "walk", "walking", "walking", "walking", "want", "want", "want", "washer", "washer", "washer", "wherever", "whether", "whether", "whether", "whilst", "wood", "wood", "wood", "worker", "world!", "world!", "world!", "york", "york", "york", "you!", "you!", "you!"]}, "R": 30, "lambda.step": 0.01, "plot.opts": {"xlab": "PC1", "ylab": "PC2"}, "topic.order": [2, 1, 3]};

function LDAvis_load_lib(url, callback){
  var s = document.createElement('script');
  s.src = url;
  s.async = true;
  s.onreadystatechange = s.onload = callback;
  s.onerror = function(){console.warn("failed to load library " + url);};
  document.getElementsByTagName("head")[0].appendChild(s);
}

if(typeof(LDAvis) !== "undefined"){
   // already loaded: just create the visualization
   !function(LDAvis){
       new LDAvis("#" + "ldavis_el2695355642553529668179362", ldavis_el2695355642553529668179362_data);
   }(LDAvis);
}else if(typeof define === "function" && define.amd){
   // require.js is available: use it to load d3/LDAvis
   require.config({paths: {d3: "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min"}});
   require(["d3"], function(d3){
      window.d3 = d3;
      LDAvis_load_lib("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js", function(){
        new LDAvis("#" + "ldavis_el2695355642553529668179362", ldavis_el2695355642553529668179362_data);
      });
    });
}else{
    // require.js not available: dynamically load d3 & LDAvis
    LDAvis_load_lib("https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js", function(){
         LDAvis_load_lib("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js", function(){
                 new LDAvis("#" + "ldavis_el2695355642553529668179362", ldavis_el2695355642553529668179362_data);
            })
         });
}
</script>




```python
final_df.to_csv('final_df.csv')
```

## IV. Prediction model

In this part, __linear regression__ and __random forest algorithm__ will be applied to the clean dataset and predict the price. Linear regression is a simple mode with good generalization while random forest model is more complex and can capture the nonlinear relationships in the dataset. From previous exploratory analysis, the distribution of price might be right-skewed. To double check how those outliers can influence the performance, models will be applied on the whole dataset and dataset without extremely high price separately. 

To measure the accuracy of the model, RMSE (root mean squared error) is used as evaluation metrics. The smaller RMSE, the better accuracy. The target for prediction is "price". Catergorical features also need to be converted to numerical features so that they can be fed into machine learning algorithms. To split the whole dataset into a training set and a testing set, the dataset will be randomly shuffled first and 25% will be used as the splitting ratio.

In order to evaluate how the two new features captured from photo and description of the listings can improve the performance of the model, both models with and without these two new features will be built.

The to-do list in this part is:

1. Clean-up the dataset: separate the "price" from the dataset, drop "zipcode","neighbourhood_group_cleansed" and convert catergorical variables into numerical features. Other columns including "level_0", "id", "listing_url", "description","image_link" can be dropped as well since they are not needed any more.
2. Split the dataset into a training set and a test set using 75:25 ratio and train the model.
3. Select the model with lowest RMSE value for further refinement.


```python
final_df.head(5)
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
      <th>level_0</th>
      <th>index</th>
      <th>listing_url</th>
      <th>id</th>
      <th>description</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>property_type</th>
      <th>room_type</th>
      <th>accommodates</th>
      <th>...</th>
      <th>price</th>
      <th>guests_included</th>
      <th>extra_people</th>
      <th>minimum_nights</th>
      <th>maximum_nights</th>
      <th>availability_365</th>
      <th>cancellation_policy</th>
      <th>NIMA_score</th>
      <th>description_topic</th>
      <th>host_is_superhost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>https://www.airbnb.com/rooms/21456</td>
      <td>21456</td>
      <td>An adorable, classic, clean, light-filled one-...</td>
      <td>40.797642</td>
      <td>-73.961775</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>...</td>
      <td>140.0</td>
      <td>2</td>
      <td>28.0</td>
      <td>5</td>
      <td>365</td>
      <td>248</td>
      <td>moderate</td>
      <td>4.519021</td>
      <td>2</td>
      <td>f</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
      <td>https://www.airbnb.com/rooms/2539</td>
      <td>2539</td>
      <td>Renovated apt home in elevator building. Spaci...</td>
      <td>40.647486</td>
      <td>-73.972370</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>4</td>
      <td>...</td>
      <td>149.0</td>
      <td>1</td>
      <td>25.0</td>
      <td>1</td>
      <td>730</td>
      <td>365</td>
      <td>moderate</td>
      <td>5.003754</td>
      <td>1</td>
      <td>t</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3</td>
      <td>https://www.airbnb.com/rooms/2595</td>
      <td>2595</td>
      <td>Find your romantic getaway to this beautiful, ...</td>
      <td>40.753621</td>
      <td>-73.983774</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>...</td>
      <td>225.0</td>
      <td>2</td>
      <td>0.0</td>
      <td>1</td>
      <td>1125</td>
      <td>350</td>
      <td>strict_14_with_grace_period</td>
      <td>4.826283</td>
      <td>2</td>
      <td>f</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>4</td>
      <td>https://www.airbnb.com/rooms/21644</td>
      <td>21644</td>
      <td>A great space in a beautiful neighborhood- min...</td>
      <td>40.828028</td>
      <td>-73.947308</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>1</td>
      <td>...</td>
      <td>89.0</td>
      <td>1</td>
      <td>55.0</td>
      <td>1</td>
      <td>60</td>
      <td>365</td>
      <td>strict_14_with_grace_period</td>
      <td>4.466677</td>
      <td>0</td>
      <td>f</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>6</td>
      <td>https://www.airbnb.com/rooms/21794</td>
      <td>21794</td>
      <td>It's comfy &amp; has a loft bed &amp; a chaise lounge,...</td>
      <td>40.740085</td>
      <td>-74.002706</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>2</td>
      <td>...</td>
      <td>111.0</td>
      <td>1</td>
      <td>50.0</td>
      <td>30</td>
      <td>1124</td>
      <td>359</td>
      <td>strict_14_with_grace_period</td>
      <td>5.036104</td>
      <td>2</td>
      <td>t</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
# features to keep
cols_to_keep = ['latitude','longitude','accommodates','bathrooms','bedrooms','guests_included','extra_people','maximum_nights','minimum_nights','property_type',
                'bed_type','room_type','cancellation_policy','NIMA_score','description_topic','host_is_superhost','price']
model_df = final_df[cols_to_keep]

# convert strings to dummies
categorical_feats = ['property_type','room_type','bed_type','cancellation_policy','host_is_superhost','description_topic']
model_df = pd.get_dummies(model_df,columns = categorical_feats,drop_first = False)

# separate the target variable "yield" from the dataset
target = model_df['price']
X_df = model_df.drop(['price'],axis = 1)
```

### Remove luxury listings

Before building the model, I checked the distribution of the target value. The distribution of listings's price is skewed towards the lower price. Almost all the prices are lower than \\$500, only a few listings have prices higher than \\$500. These listings may be outliers and influence the performance the model. To make sure this point, models on all the prices and on listings with prices lower than \\$500 will be trained separately. 


```python
sns.distplot(pd.Series(target))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x15f7f0c88>




![png](output_55_1.png)


#### Linear regression on all the dataset


```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

seed = 42
X_train,X_test,y_train,y_test = train_test_split(X_df,target,random_state=seed)
linreg = LinearRegression().fit(X_train, y_train)
y_pred_linreg = linreg.predict(X_test)
print("Root Mean squared error: %.3f" %np.sqrt(mean_squared_error(y_test,y_pred_linreg)))
rmse_lr = np.sqrt(mean_squared_error(y_test,y_pred_linreg))
```

    Root Mean squared error: 73.636


#### Random forest on all the dataset


```python
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor().fit(X_train,y_train)
y_pred_rf = rf_reg.predict(X_test)
print("Root Mean squared error: %.3f" %np.sqrt(mean_squared_error(y_test,y_pred_rf)))
rmse_rf = np.sqrt(mean_squared_error(y_test,y_pred_linreg))
```

    Root Mean squared error: 65.968


#### Linear regression on houses with price <500


```python
# drop those luxury houses
model_df_500 = model_df[model_df['price']<500]
# separate the target variable "yield" from the dataset
target_500 = model_df_500['price']
X_df_500 = model_df_500.drop(['price'], axis = 1)

print ("Final dataset has {} rows, {} columns.".format(*model_df_500.shape))
```

    Final dataset has 33407 rows, 31 columns.



```python
X_train,X_test,y_train,y_test = train_test_split(X_df_500,target_500,random_state=seed)
linreg = LinearRegression().fit(X_train, y_train)
y_pred_linreg = linreg.predict(X_test)
print("Root Mean squared error: %.3f" %np.sqrt(mean_squared_error(y_test,y_pred_linreg)))
rmse_lr_500 = np.sqrt(mean_squared_error(y_test,y_pred_linreg))
```

    Root Mean squared error: 56.622


#### Random forest on houses with price<500


```python
rf_reg = RandomForestRegressor().fit(X_train,y_train)
y_pred_rf = rf_reg.predict(X_test)
print("Root Mean squared error: %.3f" %np.sqrt(mean_squared_error(y_test,y_pred_rf)))
```

    Root Mean squared error: 51.240


#### Comparsion of performance on two datasets


```python
%matplotlib inline
models = ['linear regression','random forest']
rmse_whole = [73.636,65.968]
rmse = [56.622,51.349]


fig,axs = plt.subplots(figsize = (7,6))
bar_width = 0.40
opacity = 0.8
 
rects1 = plt.bar(index, rmse_whole, bar_width,
alpha=opacity,
color='b',
label='Using the whole dataset')
 
rects2 = plt.bar(index + bar_width, rmse, bar_width,
alpha=opacity,
color='g',
label='After removing luxury listings')
 
plt.xlabel('Models')
plt.ylabel('RMSE')
plt.title('Model performance')
plt.xticks(index + 0.5*bar_width, ('linear regression','random forest'))
plt.legend()
 
plt.tight_layout()
save_fig('Comparison of model performance on different datasets')
plt.show()
```

    Saving figure Comparison of model performance on different datasets



![png](output_66_1.png)


After removing luxury listings, the performance of both linear regression and random forest are improved. So, the dataset without prices higher than \\$500 will be used. Also, from this quick test, random forest gave better RMSE than linear regression. Hence, random forest is chosen for later fine-tuning.

### Fine tuning the random forest model


```python
from sklearn.model_selection import GridSearchCV

param_grid = {"n_estimators" :[150,175,200,225,250,300],
             "criterion": ['mse'],
             "max_features": ['auto'],
             "max_depth": [3,5,7,9,11,15,20],
             "min_samples_split":[4,6,8,10,12],
             "bootstrap":[True]}

rf_fine = RandomForestRegressor(random_state = seed)
rf_cv = GridSearchCV(rf_fine,param_grid,cv=5).fit(X_train,y_train)
y_rf_cv_pred = rf_cv.predict(X_test)
print("Root Mean squared error: %.3f" % np.sqrt(mean_squared_error(y_test, y_rf_cv_pred)))
print('Variance score: %.3f' % r2_score(y_test, y_rf_cv_pred))
print("Best Parameters: {}".format(rf_cv.best_params_))
```

    Root Mean squared error: 48.840
    Variance score: 0.630
    Best Parameters: {'bootstrap': True, 'criterion': 'mse', 'max_depth': 15, 'max_features': 'auto', 'min_samples_split': 8, 'n_estimators': 300}



```python
# save the model
from sklearn.externals import joblib 
rf_reg = rf_cv.best_estimator_
joblib.dump(rf_cv, "model.pkl")
```




    ['model.pkl']



### Model sensitivity

To test the robustness of the model, the model will run on different splittings of training and test sets 100 times. The RMSE value gave a normal distribution and a 95% confident interval is XXX.


```python
import random

random_seed = random.sample(range(1000), 500)
rmse = []
for seed in random_seed:
    X_train,X_test,y_train,y_test = train_test_split(X_df_500,target_500,random_state=seed)
    rf_reg = rf_reg.fit(X_train,y_train)
    y_rf_pred = rf_reg.predict(X_test) 
    rmse_temp = np.sqrt(mean_squared_error(y_test, y_rf_pred))
    rmse.append(rmse_temp)
```


```python
import pickle
with open("rmse.txt", "wb") as fp:   #Pickling
    pickle.dump(rmse, fp) 
```


```python
sns.distplot(rmse);
```


![png](output_75_0.png)



```python
mean_rmse = np.mean(rmse)
std_rmse = np.std(rmse)
higher = mean_rmse + 1.96*std_rmse
lower = mean_rmse - 1.96*std_rmse
```


```python
print ('95% confidence interval is: {:.3f} - {:.3f}'.format(lower,higher))
```

    95% confidence interval is: 46.731 - 49.207


### Feature importance


```python
feature_import = rf_reg.feature_importances_*100
feature_import = pd.DataFrame(list(zip(feature_import,X_train.columns.values)))
feature_import = feature_import.sort_values(by=0,axis=0,ascending=False)
feature_import.columns = ['importance %','feature']
print(feature_import[:20])
```

        importance %                                          feature
    16     33.234713                        room_type_Entire home/apt
    1      15.785836                                        longitude
    0      13.153560                                         latitude
    9       6.431818                                       NIMA_score
    3       5.085877                                        bathrooms
    4       4.813686                                         bedrooms
    2       4.215171                                     accommodates
    8       3.339125                                   minimum_nights
    7       2.894336                                   maximum_nights
    6       2.537586                                     extra_people
    5       1.562831                                  guests_included
    29      0.859737                              description_topic_2
    10      0.768167                          property_type_Apartment
    24      0.590900  cancellation_policy_strict_14_with_grace_period
    22      0.557872                     cancellation_policy_flexible
    14      0.540885                              property_type_Other
    23      0.485314                     cancellation_policy_moderate
    28      0.476538                              description_topic_1
    27      0.447435                              description_topic_0
    21      0.396167                        cancellation_policy_Other



```python
features = feature_import['feature']
importances = feature_import['importance %']

fig,ax = plt.subplots(figsize=(8,10))
y_pos = np.arange(len(features))
ax.barh(y_pos, importances, align='center',color='green')
ax.set_yticks(y_pos)
ax.set_yticklabels(features)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Importance %')
ax.set_title('Feature Importance')

plt.show()
```


![png](output_80_0.png)


In the fine-tuned model, the type of the room and the location are top 3 important features. Interestingly, __NIMA_score__, the feature captured from web photos is ranked at __4th__ important feature (__6.4%__). The features from __description__ "description_topic_0",  "description_topic_1","description_topic_2" have combined __>1.8%__ of importance. 


```python
# separate the target variable "yield" from the dataset and drop the features from web photos and descriptions
target_500_og = model_df_500['price']
X_df_500_og = model_df_500.drop(['price','NIMA_score','description_topic_0','description_topic_1','description_topic_2'], axis = 1)
```


```python
X_train,X_test,y_train,y_test = train_test_split(X_df_500_og,target_500_og,random_state=seed)
param_grid = {"n_estimators" :[150,175,200,225,250,300],
             "criterion": ['mse'],
             "max_features": ['auto'],
             "max_depth": [3,5,7,9,11,15,20],
             "min_samples_split":[4,6,8,10,12],
             "bootstrap":[True]}

rf_fine_og = RandomForestRegressor(random_state = seed)
rf_cv_og = GridSearchCV(rf_fine_og,param_grid,cv=5).fit(X_train,y_train)
y_rf_cv_pred_og = rf_cv_og.predict(X_test)
print("Root Mean squared error: %.3f" % np.sqrt(mean_squared_error(y_test, y_rf_cv_pred_og)))
print('Variance score: %.3f' % r2_score(y_test, y_rf_cv_pred_og))
print("Best Parameters: {}".format(rf_cv_og.best_params_))
```

    Root Mean squared error: 48.862
    Variance score: 0.630
    Best Parameters: {'bootstrap': True, 'criterion': 'mse', 'max_depth': 15, 'max_features': 'auto', 'min_samples_split': 8, 'n_estimators': 300}


### Model performance in different price ranges


```python
actual, predicted = pd.Series(y_test, name="Actual price"), pd.Series(y_rf_cv_pred, name="Predicted Price")
sns.regplot(predicted, actual, marker='+', scatter_kws={'alpha':0.3}, color = 'green');
```


![png](output_85_0.png)


#### price < 100


```python
# drop those luxury houses
model_df_100 = model_df[model_df['price']<=100]
# separate the target variable "yield" from the dataset
target_100 = model_df_100['price']
X_df_100 = model_df_100.drop(['price'], axis = 1)

print ("Final dataset has {} rows, {} columns.".format(*model_df_100.shape))
```

    Final dataset has 16712 rows, 31 columns.



```python
X_train,X_test,y_train,y_test = train_test_split(X_df_100,target_100,random_state=seed)
```


```python
rf_fine_100 = RandomForestRegressor(random_state = seed)
rf_cv_100 = GridSearchCV(rf_fine_100,param_grid,cv=5).fit(X_train,y_train)
y_rf_cv_pred_100 = rf_cv_100.predict(X_test)
print("Root Mean squared error: %.3f" % np.sqrt(mean_squared_error(y_test, y_rf_cv_pred_100)))
print('Variance score: %.3f' % r2_score(y_test, y_rf_cv_pred_100))
print("Best Parameters: {}".format(rf_cv_100.best_params_))
```

    Root Mean squared error: 15.871
    Variance score: 0.417
    Best Parameters: {'bootstrap': True, 'criterion': 'mse', 'max_depth': 15, 'max_features': 'auto', 'min_samples_split': 6, 'n_estimators': 300}



```python
# save model
rf_reg_100 = rf_cv_100.best_estimator_
joblib.dump(rf_cv_100, "model_100.pkl")
```




    ['model_100.pkl']



#### Price in 100 ~ 300


```python
model_df_300 = model_df[(model_df['price']>100)&(model_df['price']<=300)]

# separate the target variable "yield" from the dataset
target_300 = model_df_300['price']
X_df_300 = model_df_300.drop(['price'], axis = 1)

print ("Final dataset has {} rows, {} columns.".format(*model_df_300.shape))
```

    Final dataset has 15535 rows, 31 columns.



```python
X_train,X_test,y_train,y_test = train_test_split(X_df_300,target_300,random_state=seed)
```


```python
rf_fine_300 = RandomForestRegressor(random_state = seed)
rf_cv_300 = GridSearchCV(rf_fine_300,param_grid,cv=5).fit(X_train,y_train)
y_rf_cv_pred_300 = rf_cv_300.predict(X_test)
print("Root Mean squared error: %.3f" % np.sqrt(mean_squared_error(y_test, y_rf_cv_pred_300))) 
print('Variance score: %.3f' % r2_score(y_test, y_rf_cv_pred_300))
print("Best Parameters: {}".format(rf_cv_300.best_params_))
```

    Root Mean squared error: 41.680
    Variance score: 0.342
    Best Parameters: {'bootstrap': True, 'criterion': 'mse', 'max_depth': 15, 'max_features': 'auto', 'min_samples_split': 8, 'n_estimators': 300}



```python
# save model
rf_reg_300 = rf_cv_300.best_estimator_
joblib.dump(rf_cv_300, "model_300.pkl")
```




    ['model_300.pkl']




```python
rmse = [48.840,15.871,41.680]
dataset = ['0~500','<100','100~300']
plt.bar(range(len(rmse)),rmse)
plt.xticks(range(len(rmse)),dataset)
plt.ylabel('RMSE')
plt.title('Performance on different price range')
```




    Text(0.5,1,'Performance on different price range')




![png](output_96_1.png)


This simple test shows that models trained on different price ranges gave different performance. Our model gave much lower RMSE on relatively cheap listings (price  < \\$100) while the RMSE increases when the model is trained on listings with higher price. This suggests that the data similarity between cheap listings and luxury listings might be poor, which might be the reason why the RMSE is quite high when using the whole dataset (price between \\$0 and \\$500). Data similarity among cheap listings might also be better than similarity among luxury listings, so the model performs better on cheap listings.


## VI. Future work

### Price range prediction

One major challenge in this project is that the responsive variable: price, is highly right skewed. The random forest model suffers the problem of data similarity. One possible solution is to predict price range instead of predicting the specific price itself. Previous studies have shown that for skewed dataset,classification model may give better prediction. Suggesting a price range is also more meangingful from practical point of view.

### Dynamic Pricing

InsideAirbnb also provides the information on prices on different days for each listings. The price of listings might be influenced by seasonality. Higher prices are charged during the peak season, or during special-event periods. So, further investigations can focus on how to predict price based on information of listings and also booking time.


```python
df_calen = pd.read_csv('calendar.csv')
df_calen.head(5)
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
      <th>listing_id</th>
      <th>date</th>
      <th>available</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2515</td>
      <td>2019-12-02</td>
      <td>t</td>
      <td>$89.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21456</td>
      <td>2019-12-05</td>
      <td>t</td>
      <td>$148.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21456</td>
      <td>2019-12-04</td>
      <td>t</td>
      <td>$148.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21456</td>
      <td>2019-12-03</td>
      <td>t</td>
      <td>$148.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21456</td>
      <td>2019-12-02</td>
      <td>t</td>
      <td>$148.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
import datetime

# convert the date to datetime format
df_calen['new_date'] = pd.to_datetime(df_calen['date'],format="%Y-%m-%d")


# delete the $ before price and convert it to numerical variable
df_calen['price'] = df_calen['price'].replace('\$','',regex = True)
df_calen['price'] = df_calen['price'].apply(pd.to_numeric,errors = "coerce")

# drop missing values in price
df_calen = df_calen.dropna()
```

#### Weekdays VS Weekends

First, I analyzed the price different between weekdays and weekends.Comparing the boxplot of price distribution, however, there seems to be no significant difference between weekdays and weekends.


```python
df_calen['weekdays'] = df_calen['new_date'].apply(lambda x: x.weekday())
df_calen_weekdays = df_calen[df_calen['weekdays']<5]
df_calen_weekends = df_calen[df_calen['weekdays']>4]
df_calen['If_weekdays'] = df_calen['weekdays'].apply(lambda x: 'Weekdays' if x<5 else 'Weekends') 

sns.boxplot(data = df_calen, x = 'If_weekdays', y = 'price', order = ['Weekdays','Weekends'])
sns.boxplot(data = df_calen, x = 'If_weekdays', y = 'price', order = ['Weekdays','Weekends'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x15ed2e0b8>




![png](output_107_1.png)


There seems to be no difference among different days in a week neither.

#### Price trend in  2019


```python
df_2019 = df_calen[(df_calen['new_date']>='2018-01-01') & (df_calen['new_date']<='2019-12-31')]

df_2019 = df_2019.groupby('new_date',as_index = False).aggregate({
    'listing_id':'first',
    'price':['mean','median'],
    'weekdays':'first',
    'If_weekdays':'first'
})
```


```python
import matplotlib.dates as mdates
fig,axs = plt.subplots(figsize = (8,6))
plt.plot(df_2019['new_date'],df_2019['price']['mean'],label = "Mean")
plt.plot(df_2019['new_date'],df_2019['price']['median'],color = 'purple',label = "Median")
plt.axvline(x='2018-12-31',color ='r',linestyle = 'dashed')
plt.legend()
try:
    plt.annotate('New years eve',
             xy=('2018-12-31', 205),
             xytext = ('2019-03-20',203),
             arrowprops=dict(facecolor='red', shrink=0.05),
             fontsize=16
            )
except:
    pass

save_fig('Price trend per day in 2019')
```

    Saving figure Price trend per day in 2019



![png](output_111_1.png)



```python
import matplotlib.dates as mdates
fig,axs = plt.subplots(figsize = (8,6))
plt.plot(df_2019['new_date'][0:70],df_2019['price']['mean'][0:70],label = "Mean")
plt.plot(df_2019['new_date'][0:70],df_2019['price']['median'][0:70],color = 'purple',label = "Median")
plt.legend()

save_fig('price trend in first 70 days')
```

    Saving figure price trend in first 70 days



![png](output_112_1.png)


Due to skewed data, mean is higher than median of the price, but they show similar trend: there is a weekly cyclicality trends of the data. So I plotted the price per day of the week.


```python
week = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
week_data = df_calen.groupby('weekdays',as_index = False).aggregate({
    'price':'mean'
})

plt.bar(range(len(week_data)),week_data['price'])
plt.xticks(range(len(week_data)),week)
plt.ylabel('Averaged Price ($)')
plt.title('Averaged Price per day of the week')

save_fig('Averaged Price per Day  of the week')
```

    Saving figure Averaged Price per Day  of the week



![png](output_114_1.png)


The prices on Friday and Saturday are slightly higher. 

#### Seasonality


```python
df_calen['month'] = df_calen['new_date'].apply(lambda x: x.month)
```


```python
df_month = df_calen.groupby('month',as_index = False).aggregate({
    'price':'mean'
})
```


```python
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
fig,axs = plt.subplots(figsize = (10,5))
plt.plot(range(len(months)),df_month['price'],'bo',range(len(months)),df_month['price'],'bo-')
plt.title('Price trend in 2019')
plt.xticks(range(len(months)),months);

save_fig('Price trend in 2019')
```

    Saving figure Price trend in 2019



![png](output_119_1.png)


#### Price trend in July.


```python
df_july = df_calen[df_calen['month']==7]
```


```python
df_july.head(5)
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
      <th>listing_id</th>
      <th>date</th>
      <th>available</th>
      <th>price</th>
      <th>new_date</th>
      <th>weekdays</th>
      <th>If_weekdays</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>464</th>
      <td>30253778</td>
      <td>2019-07-31</td>
      <td>t</td>
      <td>37.0</td>
      <td>2019-07-31</td>
      <td>2</td>
      <td>Weekdays</td>
      <td>7</td>
    </tr>
    <tr>
      <th>465</th>
      <td>30253778</td>
      <td>2019-07-30</td>
      <td>t</td>
      <td>37.0</td>
      <td>2019-07-30</td>
      <td>1</td>
      <td>Weekdays</td>
      <td>7</td>
    </tr>
    <tr>
      <th>466</th>
      <td>30253778</td>
      <td>2019-07-29</td>
      <td>t</td>
      <td>37.0</td>
      <td>2019-07-29</td>
      <td>0</td>
      <td>Weekdays</td>
      <td>7</td>
    </tr>
    <tr>
      <th>467</th>
      <td>30253778</td>
      <td>2019-07-28</td>
      <td>t</td>
      <td>37.0</td>
      <td>2019-07-28</td>
      <td>6</td>
      <td>Weekends</td>
      <td>7</td>
    </tr>
    <tr>
      <th>468</th>
      <td>30253778</td>
      <td>2019-07-27</td>
      <td>t</td>
      <td>37.0</td>
      <td>2019-07-27</td>
      <td>5</td>
      <td>Weekends</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_july_days = df_july.groupby('new_date',as_index = False).aggregate({
    'price': 'mean'
})
```


```python
fig,axs = plt.subplots(figsize = (10,5))
plt.plot(range(1,len(df_july_days)+1),df_july_days['price'],'bo',range(1,len(df_july_days)+1),df_july_days['price'],'bo-')
plt.title('Price change in July')
plt.xticks(range(1,32));

save_fig('Price change in July')
```

    Saving figure Price change in July



![png](output_124_1.png)


#### Price trend in  December


```python
df_december = df_calen[df_calen['month'] == 12]
df_december_days = df_december.groupby('new_date',as_index = False).aggregate({
    'price': 'mean'
})
```


```python
fig,axs = plt.subplots(figsize = (10,5))
plt.plot(range(1,len(df_december_days)+1),df_december_days['price'],'bo',range(1,len(df_december_days)+1),df_december_days['price'],'bo-')
plt.title('Price change in December')
plt.xticks(range(1,32));

save_fig('Price Change in  December')
```

    Saving figure Price Change in  December



![png](output_127_1.png)


Seasonality is important to estabilsh the price for a listing. Based on my preliminary exploratory analysis, here are my current conclusions:
1. There seems to be no obvious fluctuation in price. Overall, the price is higher in summer and lower in December, January and Feburary. This can be explained by the weather in New York. The winter in New York is always terribly cold sometimes with unexpected snowstorms. In summer, there are always various events or cultural festivals happending in the city.
2. Even though the averaged price in winter is lower than the price in other seasons, Christmas eve and New Year's eve have the highest prices of the year. 
3. There is a weekly cyclicality trends of the data. Within a week, the price in Friday and Saturday is slightly higher than other days in a  week.

Studying time-series models and predicting the trend of price will allow a much deeper understanding on determing the best price for a new listing, so further investigation will focus on studying the seasonal nature of the data.

## V. Conclusion

The original goal of this project is to apply machine learning algorithms to predict prices of new listings for potential hosts. Combined my own experience of browsing places to stay in Airbnb website, I added two additional features into the model: image score and topic modeling from web photos and descriptions. It turned out that these two features actually contain lots of valuable informations and are important features in the fine-tuned random forest model.

So far, the random forest model is not perfect yet due to the skewed distribution of price. Simply removing the listings with high price indeed can greatly improve the model. However, it doesn't mean that those luxury listings are not important. Models trained on different price ranges gave different performance. The model gave reasonable RMSE on cheap listings (price < \\$100) while the RMSE increases a lot when training on luxury listings. This suggest that the data similarity between cheap listings and luxury listings might be poor, which can influence the performance when training on the whole dataset. One possible solution is to predict the price range instead of predicting specific price, which will change this project into a classification problem. Other models such as SVM, XGBoost might also worth trying.

Another promising problem is dynamic pricing. Preliminary data analysis shows that the price of airbnb listings is influenced by holiday seasons. Also, price in summer is higher than other seasons in a year, Friday and Saturday have the highest price within a week. Thus, studing variaous seasonal time series models will be the next step.
