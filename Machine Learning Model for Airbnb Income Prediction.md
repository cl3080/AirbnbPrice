
##  I. Introduction

Airbnb is a great platform that provides people online marketplace and service to arrange or offer lodging. As a travel enthusiast, Airbnb is always my first choice when I am planning a trip. Hosts need to provide details for their listed houses so that guests can use filters on the website to search for their preferred accomodations. For potential hosts, they must be very interested in how much they could earn from listing their houses on Airbnb. As far as I know, there is no such a model in public for predicting the yield of a new house on Airbnb. So, the object of this project is to apply machine learning models to help potential hosts gain some intuitions about the yield of their listed houses.

Fortunately, [Inside Airbnb](http://insideairbnb.com/get-the-data.html) has already aggregated all the publicly available informations from Airbnb site for public discussion. So, the dataset obtained from this website directly should be a good starting point for my machine learning model. In particular, I will the dataset collected in Los Angeles city compiled on 06 December, 2018. When selecting features for machine learning model, besides the variables provided in the datasets, the featured photo on the listing's website and the description of listing can be crucial for attracting more guests. So, I will analyze featured photos and text mining on the descriptions and add these two new features to improve the machine learning model. 

The project will be described as follows:
    1. Exploratory data analysis and data preprocessing.
    2. Feature engineering.
    3. Machine learning model.
    4. Model evaulation.



```python
# load the dataset 
import pandas as pd

df = pd.read_csv('listings.csv')
print ('There are {} rows and {} columns in the dataset'.format(*df.shape))
df.head(3)
```

    There are 43047 rows and 96 columns in the dataset





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
      <td>109</td>
      <td>https://www.airbnb.com/rooms/109</td>
      <td>20181206172531</td>
      <td>2018-12-07</td>
      <td>Amazing bright elegant condo park front *UPGRA...</td>
      <td>*** Unit upgraded with new bamboo flooring, br...</td>
      <td>*** Unit upgraded with new bamboo flooring, br...</td>
      <td>*** Unit upgraded with new bamboo flooring, br...</td>
      <td>none</td>
      <td>NaN</td>
      <td>...</td>
      <td>f</td>
      <td>NaN</td>
      <td>{"Culver City"," CA"}</td>
      <td>f</td>
      <td>f</td>
      <td>strict_14_with_grace_period</td>
      <td>t</td>
      <td>f</td>
      <td>1</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>1</th>
      <td>344</td>
      <td>https://www.airbnb.com/rooms/344</td>
      <td>20181206172531</td>
      <td>2018-12-07</td>
      <td>Family perfect;Pool;Near Studios!</td>
      <td>This home is perfect for families; aspiring ch...</td>
      <td>Cheerful &amp; comfortable; near studios, amusemen...</td>
      <td>This home is perfect for families; aspiring ch...</td>
      <td>none</td>
      <td>Quiet-yet-close to all the fun in LA! Hollywoo...</td>
      <td>...</td>
      <td>f</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>t</td>
      <td>f</td>
      <td>flexible</td>
      <td>f</td>
      <td>f</td>
      <td>1</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2708</td>
      <td>https://www.airbnb.com/rooms/2708</td>
      <td>20181206172531</td>
      <td>2018-12-06</td>
      <td>Gold Memory Foam Bed &amp; Breakfast in West Holly...</td>
      <td>Our best memory foam pillows you'll ever sleep...</td>
      <td>Flickering fireplace display heater.  Decorate...</td>
      <td>Our best memory foam pillows you'll ever sleep...</td>
      <td>none</td>
      <td>We are minutes away from the Mentor Language I...</td>
      <td>...</td>
      <td>f</td>
      <td>NaN</td>
      <td>{"City of Los Angeles"," CA"}</td>
      <td>t</td>
      <td>f</td>
      <td>strict_14_with_grace_period</td>
      <td>f</td>
      <td>f</td>
      <td>2</td>
      <td>0.24</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 96 columns</p>
</div>




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



## II. Exploratory data analysis and data preprocessing

### Data cleaning

There are 49056 observations and 96 columns in the dataset. However, not all the columns are needed for machine learning model. Especially, for a new house, there won't be any information about reviews. So columns containing informations about reviews should be dropped. These features are "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin", "review_scores_communication", "review_scores_location", "review_scores_value", "reviews_per_month". After carefully considering each features, these features are kept for further data analysis: 
> - **listing_url:** from the url, photos of the houses can be scraped. Needless to say, a comfortable featured photo of the apartment can attract more viewers and improve the yield.<br>
- **description:** description with more details about the apartment can help tourists to make the decision.<br>
- **latitude, longitude:** these two columns provide the information about the location. There are some other columns such as "transit", "zipcode", "street" are actually closely related to the location.<br>
- **property_type, room_type, bathrooms, bedrooms, bed_type, square_feet, amenities:** these columns describe the properties of the house, such as how large is the aparment, how many bathrooms or bedrooms it has.<br>
- **guests_included, cleaning_fee, extra_people, minimum_nights, maximum_nights, availability_365, cancellation_policy:** these columns provide informations about the policy of booking a room. The house with more flexible policy may be more prefered for some tourists who are not so sure about their schedules. <br>
- **reviews_per_month:** this column is kept because it will be used later for calculating the yield. <br>
- **scrape_id:** this id is kept for later image scraping.

The data cleaning process will be performed as follows:
1. Drop all the unnecessary columns.
2. "cleaning_fee","extra_people","price" have the dollar sign before the number. Need to remove the "\\$" and change the datetype from string to numerical values.
3. "property_type" has many categories, however, most of them only have few observations, so those categories can be combined into one category and name it "Other".
4. Handle missing values. First, columns including "bathrooms","bedrooms","cleaning_fee" and "reviews_per_month" have NULL values. They can be filled in with the median. There is also a column: "square_feet" whose majority of observations is missing, so this feature can be deleted. 
5. Check the distribution of variables. The distribution of "available_365" shows that some houses are only available for a few days within a year. Rooms that only available for a short time are not considered in this project. 


```python
import os
import numpy as np
import re
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
```


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
# drop all the unnecessary columns
feature_to_keep = ['listing_url','id','description','latitude','longitude','property_type','room_type','accommodates','bathrooms',
                  'bedrooms','bed_type','price','square_feet','guests_included','cleaning_fee','extra_people','minimum_nights',
                  'maximum_nights','availability_365','cancellation_policy','reviews_per_month']
new_df = df[feature_to_keep]

# remove the dollar sign before "cleaning_fee", "extra_people", "price" and change the datatype to numerical variables
feature_to_remove_dollar = ['cleaning_fee','extra_people','price']
new_df[feature_to_remove_dollar] = new_df[feature_to_remove_dollar].replace('\$','',regex = True)
new_df[feature_to_remove_dollar] = new_df[feature_to_remove_dollar].apply(pd.to_numeric,errors = "coerce")

# merge small catergories in property_type into one category "Other"
Other = ['Bed and breakfast','Resort','Boutique hotel','Guesthouse','Hostel','Hotel','Bungalow','Villa','Tiny house','Boat','Aparthotel',
         'Tent','Cottage','Camper/RV','Cabin','Casa particular (Cuba)','Nature lodge','Houseboat','Castle','Timeshare','Train','Cave','Bus',
         'Island','Earth house']
new_df['property_type'].loc[new_df['property_type'].isin(Other)] = "Other"

# drop the column "square_feet"
new_df = new_df.drop('square_feet', axis = 1)

# fill NaN with median value for 'bathrooms', 'bedrooms', 'cleaning_fee', 'price'
new_df['bathrooms'] = new_df['bathrooms'].fillna(new_df['bathrooms'].median())
new_df['bedrooms'] = new_df['bedrooms'].fillna(new_df['bedrooms'].median())
new_df['cleaning_fee'] = new_df['cleaning_fee'].fillna(new_df['cleaning_fee'].median())
new_df['price'] = new_df['price'].fillna(new_df['price'].median())

# there are 523 rows missing description, drop those rows
new_df = new_df.dropna()

# EDA of other variables and drop rows with availability_365 smaller than 10
%matplotlib inline
fig,axs = plt.subplots(ncols = 2, nrows = 3, figsize = (16,8))
plt.subplots_adjust(left=0, bottom=0, right=1, top=0.9,hspace=0.5,wspace=0.3)
sns.set(style = "white",font_scale=1.5)

sns.distplot(pd.Series(new_df['availability_365'],name = "Availability during a Year (Before cleaning)"), color ="g",ax = axs[0,0])
sns.distplot(pd.Series(new_df['price'], name = "Price (Before cleaning)"), color = "purple",ax = axs[1,0])

# remove houses that are only available for a short time and houses with high prices
new_df = new_df[new_df['availability_365']>10]
new_df = new_df[new_df['price']<500]

sns.distplot(pd.Series(new_df['availability_365'],name = "Availability during a Year (After cleaning)"), color ="r",ax = axs[0,1])
sns.distplot(pd.Series(new_df['price'],name = "Price (After cleaning)"),color = "y", ax = axs[1,1])
sns.distplot(pd.Series(new_df['bathrooms'],name = "Number of bathrooms"),color = 'blue', ax = axs[2,1])
sns.distplot(pd.Series(new_df['bedrooms'], name = "Number of bedrooms"), color = "orange",ax = axs[2,0])

save_fig("Distribution_of_variables")

print ("Dataset has {} rows and {} columns.".format(*new_df.shape))
```

    Saving figure Distribution_of_variables
    Dataset has 27826 rows and 20 columns.



![png](output_9_1.png)


After cleaning up the data, the new dataset now has 27826 rows and 20 columns without any missing values.

### Yield calculation

Inside Airbnb's ["San Francisco Model"](http://insideairbnb.com/about.html) will be used for yield calculation. The caculation is as follows:
> Yield $=$ Average length of stay $\times$ Price $\times$ Number of reviews $\times$ 12 Months $/$ Review_rate<br>

Here is how the website explained the model:<br>
> Inside Airbnb's __"San Francisco Model"__ uses as a modified methodology as follows:<br>
-  A __review rate of 50%__ is used to convert __reviews__ to __estimated bookings.__<br>
-  An __average length of stay__ is configured for each city, and this, multiplied by the __estimated bookings__ for each listings over a period gives the __occupancy rate__<br>
    - Where statements have been made about the average length of stay of Airbnb guests for a city, this was used.
    - For example, Airbnb reported 5.5 nights as the average length of stay for guests using Airbnb in San Francisco.
    - Where no public statements were made about average stays, a value of __3 nights per booking__ was used.
    - If a listing has a __higher minimum nights__ value than the average length of stay, the minimum nights value was used instead.
- The __occupancy rate__ was __capped at 70%__ - relatively high, but reasonable number for a highly occupied "hotel".
- __Number of nights__ booked or availble per year for the __high availability__ and __frequently rented__ metrics and filters were generally aligned with a city's short term rental laws designed to __protect residential housing.__<br>

In our case, the __Average length of stay__ will be 3 nights since there is no reported value. Also, if the minimum night is higher than 3 days, the average length of stay will be the value of minimum nights. 50% will be used as the review rate. The __Price__ in the model should be the sum of 'price' and 'cleaning_fee' in the dataset.


```python
# calculate the Yield using San Francisco Model
review_rate = 0.5
new_df['average_length_of_stay'] = [3 if x < 3 else x for x in new_df['minimum_nights']]
new_df['yield'] = new_df['average_length_of_stay']*(new_df['price']+new_df['cleaning_fee'])*new_df['reviews_per_month']*12/review_rate

# reviews_per_month can be dropped now
new_df = new_df.drop('reviews_per_month',axis = 1)
new_df.head(3)

# save the current dataframe into a csv file
new_df.to_csv('cleaned_df.csv')
```

## III. Feature engineering

### Image analysis on featured photos

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
listings = new_df['listing_url']
```


```python
new_df = new_df.reset_index()
```


```python
# extract the url for the feature photo from 'listings_url'
listings = new_df['listing_url']
image_link = {}
for i in range(len(listings)):
    file_url = new_df['listing_url'][i]
    page = requests.get(file_url)    
    soup = BeautifulSoup(page.text,"html.parser")
    img_tags = soup.find_all('img')
    img_urls = [img['src'] for img in img_tags]
    for url in img_urls:
        if not url.startswith("https://a0.muscache.com/im/pictures/"):
            continue
        image_link[file_url] = url
        break
```


```python
# add this featured photo url to the dataframe
new_df['image_link'] = new_df['listing_url'].map(image_link)
new_df.to_csv('cleaned_df_with_image_link.csv')

#some listings are no longer availble, so their image_link is missing.
new_df = new_df.dropna()

# set up the path for the photos output
ROOT_DIR = os.path.dirname(os.path.realpath('__file__'))
Photo_Path = os.path.join(ROOT_DIR,'Photos')

if not os.path.exists('Photos'):
    os.makedirs('Photos')
Photo_path = os.path.join('Photos')

# scraping images from the link
df_image = new_df[['id','image_link']].reset_index()

new_df = new_df.drop(['level_0'],axis = 1)
new_df = new_df.reset_index()

for i in range(len(df_image)):
   # link = df_image['image_link'][i]
    url_link = new_df['listing_url'][i]
   # print (url_link)
    link = image_link[url_link]
    photo_id = df_image['id'][i]
    image_name = os.path.join(Photo_Path,str(photo_id)+str('.jpg'))

    if not os.path.isfile(image_name):
        f = open(image_name,'wb')
        f.write(requests.get(link).content)
        f.close()
```


```python
# take random samples and check if their NIMA scores make sense
sample = df_image['image_link'][65]
photo_id = df_image['id'][65]
image_name = os.path.join(Photo_path, str(photo_id)+str('.jpg'))
img = scipy.misc.imread(image_name)
plt.imshow(img)
```




    <matplotlib.image.AxesImage at 0x152bf94e0>




![png](output_22_1.png)



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
        
    for i in range(len(df_image)): 
        try:
            photo_id = df_image['id'][i]
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
new_df['NIMA_score'] = new_df['NIMA_score'].fillna(new_df['NIMA_score'].median())

# save file into a csv
new_df.to_csv('new_df_withNIMA.csv')
```

### Sentiment analysis on 'description'

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

# save the file
final_df.to_csv('final_df_with_topicmodel.csv')
```


```python
pyLDAvis.display(p_description)
```





<link rel="stylesheet" type="text/css" href="https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.css">


<div id="ldavis_el6071057821599129736726849"></div>
<script type="text/javascript">

var ldavis_el6071057821599129736726849_data = {"mdsDat": {"Freq": [40.491729736328125, 31.83792495727539, 27.670347213745117], "cluster": [1, 1, 1], "topics": [1, 2, 3], "x": [-0.06994586652410197, 0.038645294000575287, 0.03130057252352663], "y": [-0.004045112481568678, -0.05576157456559123, 0.059806687047159926]}, "tinfo": {"Category": ["Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3"], "Freq": [9640.0, 4975.0, 5225.0, 5955.0, 2225.0, 4268.0, 2429.0, 5454.0, 3275.0, 3521.0, 4413.0, 2743.0, 6202.0, 4015.0, 2903.0, 884.0, 3144.0, 573.0, 4237.0, 3487.0, 2007.0, 6657.0, 1699.0, 1076.0, 4690.0, 2671.0, 569.0, 2627.0, 2621.0, 7550.0, 59.2674446105957, 23.509469985961914, 34.60538101196289, 36.64546585083008, 63.51087188720703, 33.126922607421875, 16.994001388549805, 25.84456443786621, 18.24403190612793, 67.82208251953125, 75.19086456298828, 12.902198791503906, 13.147143363952637, 24.418840408325195, 10.468385696411133, 10.468385696411133, 16.915624618530273, 10.839380264282227, 134.8049774169922, 10.22481632232666, 11.011080741882324, 10.952645301818848, 84.72509002685547, 13.207030296325684, 204.14312744140625, 152.411865234375, 23.2514705657959, 7.985642910003662, 16.25983428955078, 11.19469165802002, 192.583984375, 380.2549743652344, 71.06719207763672, 354.3975830078125, 289.24560546875, 194.26112365722656, 402.8204040527344, 1075.42529296875, 2227.1123046875, 137.1876678466797, 300.6102600097656, 875.4873657226562, 2283.37744140625, 346.0173034667969, 697.3974609375, 590.27880859375, 3314.248779296875, 2385.96533203125, 4018.047607421875, 902.6085205078125, 494.1896667480469, 6561.56103515625, 4310.798828125, 708.7283935546875, 4471.39794921875, 424.29132080078125, 436.5165100097656, 1564.6845703125, 941.1237182617188, 1286.7310791015625, 2744.833251953125, 975.5126953125, 3072.016845703125, 534.75732421875, 687.7343139648438, 786.52783203125, 1268.3548583984375, 1181.7440185546875, 4043.25537109375, 1403.1434326171875, 2078.4482421875, 2045.5211181640625, 788.453369140625, 1516.6708984375, 2787.7548828125, 2653.448486328125, 1484.3023681640625, 2040.001953125, 2140.828369140625, 2003.6278076171875, 2287.095703125, 1452.8909912109375, 1697.381591796875, 1834.1102294921875, 1504.9232177734375, 1408.4578857421875, 1296.895263671875, 76.40714263916016, 38.747615814208984, 30.9755859375, 79.54205322265625, 23.604549407958984, 51.180572509765625, 27.111181259155273, 18.350412368774414, 17.598432540893555, 15.101274490356445, 16.956607818603516, 14.046452522277832, 32.79350280761719, 13.820971488952637, 17.729211807250977, 108.9894790649414, 22.728300094604492, 18.147367477416992, 14.030412673950195, 13.487970352172852, 12.636012077331543, 51.93184280395508, 12.550119400024414, 12.737165451049805, 12.69933795928955, 13.528374671936035, 14.8134765625, 24.47097396850586, 12.263671875, 16.15709114074707, 519.2645874023438, 279.859375, 90.31820678710938, 30.935266494750977, 22.49565887451172, 81.09864044189453, 36.106143951416016, 23.901912689208984, 258.29156494140625, 122.28813171386719, 104.92585754394531, 336.53375244140625, 251.05908203125, 93.61693572998047, 130.6544647216797, 53.540428161621094, 604.00146484375, 144.3817901611328, 1794.4833984375, 387.6262512207031, 377.563720703125, 644.6021728515625, 3316.98095703125, 154.7203369140625, 324.18316650390625, 2690.904296875, 712.4210205078125, 678.34423828125, 3080.513916015625, 2139.486083984375, 1291.821044921875, 1111.24853515625, 405.35968017578125, 1954.21875, 2299.952392578125, 312.7602233886719, 2073.238037109375, 1447.171142578125, 1677.8089599609375, 614.895263671875, 610.8948364257812, 340.36114501953125, 920.0960693359375, 387.1738586425781, 423.883544921875, 1408.153564453125, 2132.077392578125, 926.0435180664062, 1037.352783203125, 1559.484130859375, 632.86669921875, 937.76513671875, 1503.06689453125, 953.9351196289062, 1241.3345947265625, 744.0128784179688, 1368.809814453125, 1187.1812744140625, 1071.181396484375, 1587.5455322265625, 1242.1883544921875, 1037.697021484375, 1077.2392578125, 1181.47265625, 1168.291259765625, 1274.3604736328125, 1211.1424560546875, 1232.43505859375, 38.62040328979492, 66.58897399902344, 40.41254425048828, 24.393753051757812, 21.752235412597656, 70.95478820800781, 20.09578514099121, 31.482648849487305, 29.55845832824707, 17.375423431396484, 17.29244041442871, 37.38094711303711, 15.872665405273438, 16.29778289794922, 16.806276321411133, 16.794828414916992, 15.395228385925293, 14.563313484191895, 18.62812042236328, 13.099933624267578, 27.694978713989258, 13.1093168258667, 33.10337829589844, 12.120177268981934, 17.502546310424805, 15.77754020690918, 11.553109169006348, 9.720754623413086, 71.7364730834961, 10.314618110656738, 521.6962890625, 80.938720703125, 43.18495178222656, 66.34910583496094, 30.455970764160156, 57.081825256347656, 94.1796875, 34.29403305053711, 706.6537475585938, 184.4838409423828, 276.0912780761719, 64.50847625732422, 1512.60693359375, 130.27684020996094, 728.1323852539062, 253.8160400390625, 297.59149169921875, 291.0951843261719, 215.30552673339844, 131.37318420410156, 68.51583862304688, 79.55425262451172, 70.26889038085938, 840.9307250976562, 725.5567626953125, 497.60211181640625, 2651.74267578125, 1581.4544677734375, 245.02328491210938, 404.20062255859375, 131.24545288085938, 1245.2293701171875, 353.56756591796875, 960.7653198242188, 851.6365966796875, 668.3401489257812, 420.918212890625, 636.0288696289062, 440.4400939941406, 1959.90771484375, 1017.0690307617188, 370.6728210449219, 1144.37255859375, 2274.48779296875, 1389.6881103515625, 1114.751220703125, 925.7451782226562, 1056.6156005859375, 1095.433837890625, 2426.920166015625, 1293.141845703125, 548.0172119140625, 1194.63427734375, 792.1218872070312, 1076.0748291015625, 921.2096557617188, 873.9007568359375, 868.15625, 1067.9920654296875, 967.7614135742188, 843.6439208984375, 974.7084350585938, 903.9950561523438, 802.1718139648438], "Term": ["room", "beach", "apartment", "house", "please", "la", "center", "bed", "place", "downtown", "living", "size", "bedroom", "away", "large", "hidden", "queen", "number", "walk", "walking", "heart", "kitchen", "building", "phone", "bathroom", "min", "convention", "unit", "location", "private", "mirrored", "ceilings!", "generously", "spoken", "quartz", "powder", "velvet", "suitcase", "scanner", "mirror", "formal", "group!", "copier", "divided", "furnished!!!", "broil", "recycle", "elm", "comforter", "renovated!!", "shelving", "renovated!", "printer", "cooling", "counter", "silverware", "osmosis", "lot!!", "foyer", "gleaming", "granite", "stainless", "pan", "steel", "inch", "storage", "dishwasher", "closet", "size", "tile", "hair", "master", "large", "iron", "refrigerator", "stove", "living", "queen", "bed", "sofa", "oven", "room", "bedroom", "king", "kitchen", "twin", "second", "dining", "shower", "floor", "full", "microwave", "bathroom", "screen", "mattress", "table", "bath", "washer", "private", "fully", "new", "free", "cable", "two", "parking", "house", "unit", "one", "access", "area", "home", "guest", "space", "apartment", "available", "street", "coffee", "walt", "financial", "dtla!", "broad", "football", "regency", "hotel!", "building!", "barber", "prestige", "see!", "admire", "spots!", "ace", "preview", "concert", "more!!", "dash", "hammer", "doorman", "madam", "union", "commuting", "waking", "doorstep!", "windward", "access!", "tanning", "touristic", "capitan", "convention", "silver", "staple", "fob", "epicenter", "junction", "china", "clubhouse", "boardwalk", "aquarium", "mary", "echo", "mi", "vine", "sand", "rise", "ocean", "dodger", "center", "lake", "marina", "famous", "beach", "incredible", "museum", "la", "block", "gym", "apartment", "downtown", "heart", "building", "bike", "walking", "walk", "rose", "away", "location", "distance", "sunset", "amazing", "pier", "best", "district", "fame", "great", "parking", "everything", "park", "close", "long", "modern", "access", "studio", "street", "many", "one", "place", "enjoy", "home", "available", "unit", "neighborhood", "space", "area", "bedroom", "kitchen", "private", "emergency!", "efficiency", "possible!", "antelope", "love!", "studios!", "colima", "specifically", "days!", "couches!", "humble", "vary", "studio!!!!", "unlock", "carefully!!!!!", "posted", "yurt", "fragrant", "selected", "wellness", "questions!", "glamp!", "mart", "journey!", "whose", "manor", "feminine", "hive!", "cabin", "boiler", "number", "positive", "shuttle", "tiny", "hangout", "similar", "hosting", "alcohol", "hidden", "respect", "booking", "sanctuary", "please", "send", "phone", "read", "ask", "know", "mountain", "mind", "chat", "cup", "booked", "check", "time", "may", "house", "place", "book", "food", "affordable", "min", "mall", "use", "like", "welcome", "want", "day", "always", "home", "need", "call", "quiet", "private", "close", "guest", "city", "stay", "neighborhood", "room", "space", "love", "away", "studio", "available", "also", "coffee", "enjoy", "bathroom", "area", "great", "kitchen", "parking", "access"], "Total": [9640.0, 4975.0, 5225.0, 5955.0, 2225.0, 4268.0, 2429.0, 5454.0, 3275.0, 3521.0, 4413.0, 2743.0, 6202.0, 4015.0, 2903.0, 884.0, 3144.0, 573.0, 4237.0, 3487.0, 2007.0, 6657.0, 1699.0, 1076.0, 4690.0, 2671.0, 569.0, 2627.0, 2621.0, 7550.0, 60.62390899658203, 24.200143814086914, 35.89117431640625, 38.123172760009766, 66.12162017822266, 34.638851165771484, 17.799283981323242, 27.07170867919922, 19.199522018432617, 71.39411926269531, 79.30428314208984, 13.631508827209473, 13.91970157623291, 25.975622177124023, 11.151816368103027, 11.151816368103027, 18.079572677612305, 11.59110164642334, 144.2915496826172, 10.95651912689209, 11.803558349609375, 11.753849029541016, 90.99308776855469, 14.211788177490234, 220.15725708007812, 164.58734130859375, 25.23297882080078, 8.669044494628906, 17.65445327758789, 12.171246528625488, 210.6060333251953, 420.9266052246094, 77.44529724121094, 393.57098388671875, 322.7651062011719, 217.90696716308594, 461.1394348144531, 1271.920654296875, 2743.17236328125, 155.07186889648438, 350.18792724609375, 1060.50048828125, 2903.71142578125, 407.8155517578125, 846.9932861328125, 714.2841186523438, 4413.89990234375, 3144.28759765625, 5454.69384765625, 1134.7855224609375, 600.103271484375, 9640.6982421875, 6202.486328125, 894.5645751953125, 6657.2490234375, 516.86962890625, 534.6141357421875, 2163.631103515625, 1240.3154296875, 1751.6165771484375, 4104.6806640625, 1301.1815185546875, 4690.54150390625, 674.5848999023438, 903.474365234375, 1059.6971435546875, 1867.6793212890625, 1720.119873046875, 7550.17822265625, 2128.739013671875, 3505.62744140625, 3466.610595703125, 1084.3935546875, 2542.076171875, 5823.8271484375, 5955.58203125, 2627.962890625, 4141.2666015625, 4446.06689453125, 4139.68017578125, 5834.548828125, 2982.19482421875, 4171.99609375, 5225.30615234375, 3823.1865234375, 3430.664306640625, 2734.6826171875, 77.76670837402344, 39.81101989746094, 31.847686767578125, 82.09535217285156, 24.465198516845703, 53.21110534667969, 28.19777488708496, 19.137399673461914, 18.439910888671875, 15.845486640930176, 17.848018646240234, 14.820578575134277, 34.61124801635742, 14.60908031463623, 18.761367797851562, 115.39366149902344, 24.13658905029297, 19.27890968322754, 14.914591789245605, 14.340505599975586, 13.459099769592285, 55.3573112487793, 13.388443946838379, 13.591156959533691, 13.563199043273926, 14.451570510864258, 15.830028533935547, 26.162229537963867, 13.112863540649414, 17.294708251953125, 569.4131469726562, 314.5965576171875, 99.40923309326172, 33.37261962890625, 24.175310134887695, 89.48107147216797, 39.22919464111328, 25.779706954956055, 293.85931396484375, 136.4785614013672, 116.73567962646484, 389.6318664550781, 289.2296142578125, 104.82772827148438, 149.63555908203125, 59.2369499206543, 758.7967529296875, 169.99417114257812, 2429.825439453125, 481.5738525390625, 475.6535339355469, 849.3275146484375, 4975.56494140625, 186.0596923828125, 420.56561279296875, 4268.13232421875, 996.83349609375, 950.121337890625, 5225.30615234375, 3521.269775390625, 2007.53564453125, 1699.38623046875, 560.7691040039062, 3487.70361328125, 4237.005859375, 423.7643737792969, 4015.7392578125, 2621.434814453125, 3126.95166015625, 946.8240356445312, 942.8811645507812, 468.8026428222656, 1569.7923583984375, 548.467529296875, 623.7764892578125, 3180.7236328125, 5823.8271484375, 1882.854736328125, 2216.361572265625, 4014.51806640625, 1093.851318359375, 1991.05859375, 4446.06689453125, 2157.5537109375, 3430.664306640625, 1441.56591796875, 4141.2666015625, 3275.78173828125, 2733.8896484375, 5834.548828125, 3823.1865234375, 2627.962890625, 3013.73828125, 4171.99609375, 4139.68017578125, 6202.486328125, 6657.2490234375, 7550.17822265625, 39.32156753540039, 67.98442840576172, 41.6359977722168, 25.168214797973633, 22.467113494873047, 73.32164764404297, 20.790462493896484, 32.5817985534668, 30.626354217529297, 18.07601547241211, 18.042072296142578, 39.018035888671875, 16.58055305480957, 17.04454803466797, 17.585086822509766, 17.628704071044922, 16.168054580688477, 15.326667785644531, 19.631427764892578, 13.844120025634766, 29.2861328125, 13.886786460876465, 35.149288177490234, 12.91435432434082, 18.68769073486328, 16.862436294555664, 12.35446548461914, 10.395118713378906, 76.72130584716797, 11.037120819091797, 573.4002075195312, 88.0297622680664, 46.69566345214844, 72.10650634765625, 32.849754333496094, 62.811038970947266, 106.39861297607422, 37.49842071533203, 884.510498046875, 217.86782836914062, 345.6596374511719, 74.74710845947266, 2225.542236328125, 161.74514770507812, 1076.70849609375, 343.0935363769531, 414.0098876953125, 408.65350341796875, 292.1334228515625, 170.2478485107422, 81.75483703613281, 97.02678680419922, 84.7433853149414, 1469.98583984375, 1255.140625, 817.732177734375, 5955.58203125, 3275.78173828125, 365.2503356933594, 677.7321166992188, 176.89610290527344, 2671.0859375, 586.9034423828125, 2046.378662109375, 1775.0709228515625, 1319.935791015625, 748.355712890625, 1286.26611328125, 808.6719970703125, 5834.548828125, 2474.47509765625, 661.3847045898438, 2979.635009765625, 7550.17822265625, 4014.51806640625, 2982.19482421875, 2323.480712890625, 2840.0693359375, 3013.73828125, 9640.6982421875, 4171.99609375, 1142.283203125, 4015.7392578125, 2157.5537109375, 3823.1865234375, 2911.3349609375, 2734.6826171875, 2733.8896484375, 4690.54150390625, 4139.68017578125, 3180.7236328125, 6657.2490234375, 5823.8271484375, 4446.06689453125], "loglift": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.8813999891281128, 0.8751000165939331, 0.8676000237464905, 0.8644999861717224, 0.8637999892234802, 0.8593999743461609, 0.8578000068664551, 0.857699990272522, 0.8529999852180481, 0.8526999950408936, 0.8507999777793884, 0.8490999937057495, 0.847000002861023, 0.8422999978065491, 0.8407999873161316, 0.8407999873161316, 0.8374999761581421, 0.8370000123977661, 0.8360999822616577, 0.8349999785423279, 0.8345999717712402, 0.8335000276565552, 0.8327000141143799, 0.8307999968528748, 0.8285999894142151, 0.8271999955177307, 0.8223000168800354, 0.8220000267028809, 0.8217999935150146, 0.8203999996185303, 0.8145999908447266, 0.8025000095367432, 0.8180999755859375, 0.7991999983787537, 0.7943999767303467, 0.7892000079154968, 0.7688999772071838, 0.736299991607666, 0.6956999897956848, 0.781499981880188, 0.7513999938964844, 0.7124000191688538, 0.6636999845504761, 0.7397000193595886, 0.7096999883651733, 0.7134000062942505, 0.6175000071525574, 0.6280999779701233, 0.5983999967575073, 0.6751999855041504, 0.7099000215530396, 0.5192999839782715, 0.5401999950408936, 0.6711999773979187, 0.5060999989509583, 0.7067000269889832, 0.7013999819755554, 0.5799999833106995, 0.628000020980835, 0.5956000089645386, 0.5016999840736389, 0.6159999966621399, 0.48089998960494995, 0.6718000173568726, 0.6312000155448914, 0.6060000061988831, 0.5170999765396118, 0.5286999940872192, 0.27959999442100525, 0.48730000853538513, 0.3813000023365021, 0.3765000104904175, 0.5853999853134155, 0.38760000467300415, 0.16740000247955322, 0.09560000151395798, 0.3328000009059906, 0.19599999487400055, 0.17319999635219574, 0.17839999496936798, -0.03240000084042549, 0.1850000023841858, 0.004800000227987766, -0.1429000049829483, -0.028300000354647636, 0.013799999840557575, 0.15800000727176666, 1.1268999576568604, 1.117400050163269, 1.1167000532150269, 1.1129000186920166, 1.1087000370025635, 1.1055999994277954, 1.1052000522613525, 1.1024999618530273, 1.0978000164031982, 1.0964000225067139, 1.0932999849319458, 1.09089994430542, 1.0906000137329102, 1.0891000032424927, 1.0879000425338745, 1.087399959564209, 1.0844000577926636, 1.0839999914169312, 1.083400011062622, 1.0831999778747559, 1.0814000368118286, 1.0806000232696533, 1.0799000263214111, 1.0795999765396118, 1.0786999464035034, 1.0785000324249268, 1.0780999660491943, 1.0777000188827515, 1.0776000022888184, 1.0765000581741333, 1.052299976348877, 1.027500033378601, 1.0485999584197998, 1.0686999559402466, 1.0724999904632568, 1.0462000370025635, 1.0615999698638916, 1.0688999891281128, 1.0154999494552612, 1.0347000360488892, 1.0378999710083008, 0.9980000257492065, 1.003000020980835, 1.0313999652862549, 1.0089000463485718, 1.0434000492095947, 0.9164000153541565, 0.9811999797821045, 0.8414000272750854, 0.9275000095367432, 0.9136000275611877, 0.8687000274658203, 0.7390000224113464, 0.960099995136261, 0.8841999769210815, 0.6832000017166138, 0.8086000084877014, 0.8076000213623047, 0.616100013256073, 0.6463000178337097, 0.7037000060081482, 0.7196999788284302, 0.8199999928474426, 0.5652999877929688, 0.5335000157356262, 0.8407999873161316, 0.48339998722076416, 0.5504000186920166, 0.5218999981880188, 0.7128999829292297, 0.7105000019073486, 0.8242999911308289, 0.6103000044822693, 0.7962999939918518, 0.7581999897956848, 0.3296999931335449, 0.1396999955177307, 0.4348999857902527, 0.38530001044273376, 0.1988999992609024, 0.5972999930381775, 0.39160001277923584, 0.05999999865889549, 0.32839998602867126, 0.12790000438690186, 0.4830999970436096, 0.03750000149011612, 0.12950000166893005, 0.20749999582767487, -0.15710000693798065, 0.0203000009059906, 0.21529999375343323, 0.11569999903440475, -0.11710000038146973, -0.12060000002384186, -0.43799999356269836, -0.5595999956130981, -0.6680999994277954, 1.266800045967102, 1.2640999555587769, 1.2549999952316284, 1.253600001335144, 1.252500057220459, 1.2519999742507935, 1.2508000135421753, 1.250499963760376, 1.2493000030517578, 1.245300054550171, 1.242400050163269, 1.2418999671936035, 1.2411999702453613, 1.2400000095367432, 1.2395000457763672, 1.2364000082015991, 1.23580002784729, 1.2337000370025635, 1.232300043106079, 1.229599952697754, 1.2288999557495117, 1.2272000312805176, 1.2247999906539917, 1.2213000059127808, 1.2193000316619873, 1.2182999849319458, 1.2177000045776367, 1.2177000045776367, 1.2175999879837036, 1.2171000242233276, 1.1902999877929688, 1.2007999420166016, 1.2065999507904053, 1.2015999555587769, 1.2091000080108643, 1.1892000436782837, 1.1627999544143677, 1.1955000162124634, 1.0602999925613403, 1.118499994277954, 1.0600999593734741, 1.1375000476837158, 0.8985999822616577, 1.0684000253677368, 0.8935999870300293, 0.9833999872207642, 0.9545999765396118, 0.9455999732017517, 0.9797000288963318, 1.0255999565124512, 1.1081000566482544, 1.086300015449524, 1.097499966621399, 0.7263000011444092, 0.7366999983787537, 0.788100004196167, 0.4756999909877777, 0.5565999746322632, 0.8855999708175659, 0.7680000066757202, 0.986299991607666, 0.5216000080108643, 0.777999997138977, 0.5286999940872192, 0.5504000186920166, 0.6043000221252441, 0.7093999981880188, 0.5806000232696533, 0.6772000193595886, 0.193900004029274, 0.39570000767707825, 0.7057999968528748, 0.3278999924659729, 0.08500000089406967, 0.2240000069141388, 0.30079999566078186, 0.3646000027656555, 0.296099990606308, 0.2727999985218048, -0.09459999948740005, 0.11349999904632568, 0.5503000020980835, 0.07240000367164612, 0.28279998898506165, 0.017000000923871994, 0.13410000503063202, 0.14399999380111694, 0.13770000636577606, -0.19499999284744263, -0.16859999299049377, -0.04230000078678131, -0.6365000009536743, -0.5781000256538391, -0.4275999963283539], "logprob": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, -8.379599571228027, -9.304300308227539, -8.917699813842773, -8.860400199890137, -8.310500144958496, -8.961400032043457, -9.628800392150879, -9.209600448608398, -9.557900428771973, -8.244799613952637, -8.14169979095459, -9.90429973602295, -9.885499954223633, -9.266400337219238, -10.113300323486328, -10.113300323486328, -9.633500099182129, -10.078499794006348, -7.5578999519348145, -10.136899948120117, -10.062800407409668, -10.068099975585938, -8.022299766540527, -9.880999565124512, -7.142899990081787, -7.435100078582764, -9.315299987792969, -10.384099960327148, -9.67300033569336, -10.046299934387207, -7.201200008392334, -6.520899772644043, -8.198100090026855, -6.591300010681152, -6.794400215148926, -7.192500114440918, -6.463200092315674, -5.481200218200684, -4.753200054168701, -7.54040002822876, -6.755899906158447, -5.6869001388549805, -4.728300094604492, -6.615200042724609, -5.914400100708008, -6.081099987030029, -4.3557000160217285, -4.684299945831299, -4.1631999015808105, -5.656400203704834, -6.258800029754639, -3.6726999282836914, -4.092800140380859, -5.898200035095215, -4.05620002746582, -6.411300182342529, -6.382900238037109, -5.106299877166748, -5.61460018157959, -5.301799774169922, -4.5441999435424805, -5.578700065612793, -4.431600093841553, -6.179900169372559, -5.928299903869629, -5.794099807739258, -5.316199779510498, -5.38700008392334, -4.156899929046631, -5.215199947357178, -4.822299957275391, -4.8383002281188965, -5.791600227355957, -5.137400150299072, -4.52869987487793, -4.578100204467773, -5.158999919891357, -4.841000080108643, -4.792799949645996, -4.859000205993652, -4.7266998291015625, -5.1803998947143555, -5.024899959564209, -4.947400093078613, -5.145199775695801, -5.21150016784668, -5.294000148773193, -7.885200023651123, -8.564200401306152, -8.788100242614746, -7.84499979019165, -9.059800148010254, -8.285900115966797, -8.921299934387207, -9.311599731445312, -9.353500366210938, -9.506500244140625, -9.390600204467773, -9.578900337219238, -8.730999946594238, -9.595100402832031, -9.346099853515625, -7.53000020980835, -9.097700119018555, -9.322699546813965, -9.579999923706055, -9.619500160217285, -9.684700012207031, -8.271300315856934, -9.691499710083008, -9.6766996383667, -9.679699897766113, -9.616499900817871, -9.525699615478516, -9.023799896240234, -9.714599609375, -9.438899993896484, -5.968900203704834, -6.586999893188477, -7.717899799346924, -8.789400100708008, -9.10789966583252, -7.8256001472473145, -8.63479995727539, -9.047300338745117, -6.667200088500977, -7.414899826049805, -7.567999839782715, -6.402599811553955, -6.6956000328063965, -7.68209981918335, -7.348700046539307, -8.240799903869629, -5.817699909210205, -7.248799800872803, -4.728799819946289, -6.261199951171875, -6.287499904632568, -5.752600193023682, -4.114500045776367, -7.179599761962891, -6.440000057220459, -4.323599815368652, -5.652599811553955, -5.701600074768066, -4.188399791717529, -4.5528998374938965, -5.057499885559082, -5.208000183105469, -6.2164998054504395, -4.643499851226807, -4.480599880218506, -6.475800037384033, -4.584400177001953, -4.943900108337402, -4.796000003814697, -5.799799919128418, -5.806300163269043, -6.391300201416016, -5.3968000411987305, -6.262400150299072, -6.171800136566162, -4.971199989318848, -4.556399822235107, -5.3902997970581055, -5.276800155639648, -4.869200229644775, -5.770999908447266, -5.377799987792969, -4.906000137329102, -5.3607001304626465, -5.097300052642822, -5.6092000007629395, -4.999599933624268, -5.141900062561035, -5.244699954986572, -4.85129976272583, -5.09660005569458, -5.276500225067139, -5.239099979400635, -5.146699905395508, -5.1579999923706055, -5.071100234985352, -5.1219000816345215, -5.104499816894531, -8.427200317382812, -7.882400035858154, -8.381799697875977, -8.88659954071045, -9.001299858093262, -7.818900108337402, -9.080499649047852, -8.631500244140625, -8.694600105285645, -9.225899696350098, -9.23069953918457, -8.459799766540527, -9.316399574279785, -9.289899826049805, -9.259200096130371, -9.259900093078613, -9.34689998626709, -9.40250015258789, -9.156299591064453, -9.508399963378906, -8.759699821472168, -9.507599830627441, -8.581299781799316, -9.586099624633789, -9.218600273132324, -9.322400093078613, -9.633999824523926, -9.806699752807617, -7.808000087738037, -9.747400283813477, -5.82390022277832, -7.687300205230713, -8.315500259399414, -7.886000156402588, -8.66469955444336, -8.036499977111816, -7.535799980163574, -8.545999526977539, -5.520400047302246, -6.863399982452393, -6.46019983291626, -7.9141998291015625, -4.759399890899658, -7.211299896240234, -5.490499973297119, -6.544400215148926, -6.385200023651123, -6.407299995422363, -6.708899974822998, -7.202899932861328, -7.853899955749512, -7.704500198364258, -7.82859992980957, -5.346499919891357, -5.49399995803833, -5.871200084686279, -4.197999954223633, -4.714900016784668, -6.579599857330322, -6.079100131988525, -7.20389986038208, -4.95389986038208, -6.212900161743164, -5.213200092315674, -5.333799839019775, -5.576200008392334, -6.03849983215332, -5.625699996948242, -5.993199825286865, -4.50029993057251, -5.156300067901611, -6.165599822998047, -5.038400173187256, -4.351500034332275, -4.844099998474121, -5.064599990844727, -5.250400066375732, -5.118100166320801, -5.082099914550781, -4.286600112915039, -4.916100025177002, -5.774700164794922, -4.9953999519348145, -5.406300067901611, -5.099899768829346, -5.255300045013428, -5.308000087738037, -5.314599990844727, -5.107399940490723, -5.205999851226807, -5.343200206756592, -5.198800086975098, -5.274099826812744, -5.393599987030029]}, "token.table": {"Topic": [1, 2, 3, 1, 2, 2, 2, 1, 2, 3, 1, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 2, 3, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 3, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 3, 3, 1, 2, 3, 2, 1, 2, 3, 1, 2, 3, 1, 3, 1, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 3, 1, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 2, 1, 2, 3, 2, 1, 2, 3, 1, 2, 3, 1, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 3, 1, 2, 1, 2, 3, 1, 2, 1, 2, 3, 2, 1, 2, 3, 1, 2, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 2, 3, 1, 3, 1, 2, 3, 1, 2, 3, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 2, 3, 3, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 3, 1, 3, 1, 2, 3, 2, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 3, 1, 2, 3, 1, 2, 3, 1, 3, 1, 2, 3, 2, 3, 1, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 2, 3, 1, 2, 3, 2, 1, 3, 1, 2, 3, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 3, 1, 2, 3, 1, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 3, 1, 2, 3, 1, 3, 1, 1, 2, 3, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 3, 1, 3, 1, 2, 3], "Freq": [0.4815492033958435, 0.33805158734321594, 0.18038414418697357, 0.06317108124494553, 0.9475662112236023, 0.9583081007003784, 0.9446324706077576, 0.04522428661584854, 0.21481536328792572, 0.7405477166175842, 0.08000336587429047, 0.906704843044281, 0.3675290048122406, 0.31600624322891235, 0.31634971499443054, 0.18919909000396729, 0.2671045958995819, 0.5441019535064697, 0.1856013387441635, 0.6480137705802917, 0.1665109097957611, 0.9535837173461914, 0.35098421573638916, 0.5896305441856384, 0.05951804295182228, 0.09525305777788162, 0.8939132690429688, 0.0073271580040454865, 0.48409536480903625, 0.2821474075317383, 0.23383449018001556, 0.17873969674110413, 0.10144685208797455, 0.7197895646095276, 0.39365068078041077, 0.3248598873615265, 0.2814406156539917, 0.18626707792282104, 0.5162187814712524, 0.2975790798664093, 0.9761435389518738, 0.678917407989502, 0.20827986299991608, 0.11243899911642075, 0.6549350619316101, 0.11747044324874878, 0.22769226133823395, 0.1997763067483902, 0.6666579842567444, 0.1336531639099121, 0.7366132736206055, 0.14336276054382324, 0.12008006870746613, 0.6950438618659973, 0.2054014950990677, 0.09947624057531357, 0.17136023938655853, 0.5860647559165955, 0.2427072525024414, 0.1373114138841629, 0.7222223877906799, 0.14087794721126556, 0.20765754580497742, 0.7142617106437683, 0.0772445946931839, 0.1122986376285553, 0.8779711723327637, 0.01020896714180708, 0.9060333967208862, 0.186173677444458, 0.1423681080341339, 0.6707728505134583, 0.08260231465101242, 0.08260231465101242, 0.8260231614112854, 0.14465096592903137, 0.05496736615896225, 0.7984732985496521, 0.02436191588640213, 0.9744765758514404, 0.012180957943201065, 0.8967149257659912, 0.2824549078941345, 0.6537654399871826, 0.0635523572564125, 0.9405666589736938, 0.02606837823987007, 0.03910256549715996, 0.9384616017341614, 0.726673424243927, 0.19181227684020996, 0.08115135133266449, 0.13456615805625916, 0.30541983246803284, 0.5609443187713623, 0.05782115459442139, 0.9251384735107422, 0.05782115459442139, 0.9667282104492188, 0.9917296171188354, 0.16420932114124298, 0.7383246421813965, 0.09753786772489548, 0.11008522659540176, 0.06115845963358879, 0.8439867496490479, 0.2517031133174896, 0.17619217932224274, 0.5721143484115601, 0.0764736607670784, 0.9176838994026184, 0.2466127574443817, 0.35507073998451233, 0.3985399901866913, 0.2652871310710907, 0.3883405029773712, 0.346243292093277, 0.845178484916687, 0.07783504575490952, 0.07704883068799973, 0.038790199905633926, 0.9309648275375366, 0.474278062582016, 0.20623965561389923, 0.31959831714630127, 0.9619795680046082, 0.9356057047843933, 0.034652065485715866, 0.027721650898456573, 0.9709866046905518, 0.01733197458088398, 0.9445925951004028, 0.03466394916176796, 0.07376015186309814, 0.9114647507667542, 0.015805747359991074, 0.9147335886955261, 0.07036412507295609, 0.9339280724525452, 0.9404727220535278, 0.9266104102134705, 0.05904870107769966, 0.013626622967422009, 0.16490291059017181, 0.020612863823771477, 0.8245145678520203, 0.05187015235424042, 0.9336627721786499, 0.2674407660961151, 0.23789788782596588, 0.49445444345474243, 0.032651618123054504, 0.9795485138893127, 0.7233210802078247, 0.17239537835121155, 0.10445403307676315, 0.8739222288131714, 0.10408999025821686, 0.021685415878891945, 0.3182012736797333, 0.5366248488426208, 0.1451893299818039, 0.1987355500459671, 0.7056024074554443, 0.0948096215724945, 0.9239432215690613, 0.03849763423204422, 0.03849763423204422, 0.1000034287571907, 0.8470878601074219, 0.04706043750047684, 0.9065231084823608, 0.9584759473800659, 0.19311214983463287, 0.607451319694519, 0.19907590746879578, 0.9733830690383911, 0.06416312605142593, 0.8649190068244934, 0.07186270505189896, 0.014709251001477242, 0.014709251001477242, 0.9855197668075562, 0.9490038156509399, 0.9918221235275269, 0.2907944619655609, 0.3917495310306549, 0.3174963593482971, 0.04136451706290245, 0.9100193381309509, 0.04136451706290245, 0.27458304166793823, 0.49180638790130615, 0.23315659165382385, 0.16993266344070435, 0.6797306537628174, 0.15069499611854553, 0.1389334499835968, 0.7594243288040161, 0.10243397951126099, 0.971308708190918, 0.025118673220276833, 0.9796282649040222, 0.7347498536109924, 0.1883973926305771, 0.0770716592669487, 0.059929367154836655, 0.928905189037323, 0.21394884586334229, 0.18886518478393555, 0.5961057543754578, 0.9809852838516235, 0.9457244277000427, 0.025219319388270378, 0.025219319388270378, 0.906286895275116, 0.05664293095469475, 0.05664293095469475, 0.9786863327026367, 0.5902018547058105, 0.22384977340698242, 0.18606069684028625, 0.6687487363815308, 0.23168672621250153, 0.09964234381914139, 0.6590756177902222, 0.26447582244873047, 0.0761013925075531, 0.8967149257659912, 0.9751700758934021, 0.027862003073096275, 0.9361417293548584, 0.9037693738937378, 0.08216085284948349, 0.9164029955863953, 0.07597123086452484, 0.009496403858065605, 0.2920719087123871, 0.44266656041145325, 0.2653484344482422, 0.9536728858947754, 0.4872250556945801, 0.13915924727916718, 0.3738856911659241, 0.24312683939933777, 0.713593065738678, 0.04315238445997238, 0.8595384955406189, 0.07995706796646118, 0.06282340735197067, 0.9386780261993408, 0.060883257538080215, 0.030441628769040108, 0.9132488369941711, 0.2659977674484253, 0.643575131893158, 0.0901602953672409, 0.1254931390285492, 0.07574810832738876, 0.799312174320221, 0.9619899988174438, 0.391975462436676, 0.27217185497283936, 0.3359299898147583, 0.08458756655454636, 0.028195856139063835, 0.8834701776504517, 0.03546379134058952, 0.9575223326683044, 0.44546443223953247, 0.10914130508899689, 0.4452965259552002, 0.9422420859336853, 0.8953880071640015, 0.07745570689439774, 0.024785825982689857, 0.06449542939662933, 0.8330659866333008, 0.10749238729476929, 0.8484227657318115, 0.10543982684612274, 0.0465896911919117, 0.9291986227035522, 0.033526644110679626, 0.9052193760871887, 0.05587773770093918, 0.792564332485199, 0.1419685035943985, 0.06595387309789658, 0.6715987324714661, 0.18190696835517883, 0.14645689725875854, 0.12235304713249207, 0.16395308077335358, 0.7120947241783142, 0.19352726638317108, 0.6304865479469299, 0.17618948221206665, 0.07683141529560089, 0.8056915998458862, 0.11836190521717072, 0.7862351536750793, 0.10503798723220825, 0.10848185420036316, 0.1926683634519577, 0.32787421345710754, 0.47998082637786865, 0.7508099675178528, 0.1343483179807663, 0.11486440896987915, 0.20713846385478973, 0.5519877672195435, 0.2407078742980957, 0.2614614963531494, 0.5786892771720886, 0.15998518466949463, 0.9228237271308899, 0.18209144473075867, 0.33879515528678894, 0.4797409176826477, 0.9792090058326721, 0.9658892750740051, 0.24705937504768372, 0.14993949234485626, 0.6031656265258789, 0.0593034103512764, 0.0593034103512764, 0.9488545656204224, 0.30452996492385864, 0.5161054134368896, 0.1789720505475998, 0.09250430762767792, 0.7946960926055908, 0.1156303808093071, 0.056900154799222946, 0.9388526082038879, 0.09422997385263443, 0.899467945098877, 0.008566360920667648, 0.8250821232795715, 0.08675149083137512, 0.0876944437623024, 0.7615047097206116, 0.16823941469192505, 0.06973081082105637, 0.23112702369689941, 0.1601991504430771, 0.6090013384819031, 0.07260667532682419, 0.8678225874900818, 0.05877682939171791, 0.7500874996185303, 0.03919514641165733, 0.2105778455734253, 0.21264760196208954, 0.3212176561355591, 0.46610257029533386, 0.09398063272237778, 0.13509716093540192, 0.7694664001464844, 0.9524593949317932, 0.014006756246089935, 0.042020268738269806, 0.9732133746147156, 0.01649514213204384, 0.01649514213204384, 0.46608373522758484, 0.4711061716079712, 0.06328292191028595, 0.04143087565898895, 0.9529101252555847, 0.17800085246562958, 0.0855773389339447, 0.7359650731086731, 0.16168701648712158, 0.7703911066055298, 0.06895475834608078, 0.2723810076713562, 0.3164307475090027, 0.4109962582588196, 0.2790554165840149, 0.3573634922504425, 0.36333611607551575, 0.592761218547821, 0.2170795351266861, 0.18998025357723236, 0.0767352357506752, 0.0122078787535429, 0.9103589057922363, 0.11729095876216888, 0.7959970831871033, 0.08697981387376785, 0.4926029145717621, 0.33057519793510437, 0.17675751447677612, 0.9115055203437805, 0.039630673825740814, 0.039630673825740814, 0.8231916427612305, 0.04165949672460556, 0.13497677445411682, 0.9167761206626892, 0.01291233953088522, 0.06456170231103897, 0.20799855887889862, 0.4678839445114136, 0.32395437359809875, 0.4787229895591736, 0.3660822808742523, 0.15522438287734985, 0.14581476151943207, 0.1783212423324585, 0.6761347055435181, 0.20477700233459473, 0.7252519130706787, 0.07039209455251694, 0.15477222204208374, 0.3623562455177307, 0.4826328754425049, 0.19680596888065338, 0.12356539070606232, 0.6798343062400818, 0.02271958813071251, 0.05679897218942642, 0.9201433658599854, 0.024017678573727608, 0.9607071280479431, 0.05672566592693329, 0.9643363356590271, 0.9526875019073486, 0.02886931784451008, 0.02886931784451008, 0.9466418027877808, 0.053301017731428146, 0.9594182968139648, 0.9341368675231934, 0.04395938292145729, 0.021979691460728645, 0.5354840755462646, 0.16317495703697205, 0.3011849522590637, 0.9679133892059326, 0.015123646706342697, 0.015123646706342697, 0.7588364481925964, 0.17014983296394348, 0.07092226296663284, 0.03414585441350937, 0.9560838937759399, 0.3946792185306549, 0.22116802632808685, 0.383939653635025, 0.12533025443553925, 0.13407422602176666, 0.7403228878974915, 0.940287709236145, 0.055311042815446854, 0.8229109048843384, 0.023612938821315765, 0.15348409116268158, 0.9584465622901917, 0.037586137652397156, 0.9358636736869812, 0.9126986265182495, 0.06425914168357849, 0.0872088372707367, 0.844548761844635, 0.084406778216362, 0.9115931987762451, 0.01688135601580143, 0.6806560754776001, 0.06762995570898056, 0.2517452538013458, 0.14394791424274445, 0.738618016242981, 0.11799009889364243, 0.0535137765109539, 0.09364910423755646, 0.8695988655090332, 0.09356065094470978, 0.8754603862762451, 0.03341451659798622, 0.937523365020752, 0.7930803298950195, 0.12896819412708282, 0.07856683433055878, 0.8174119591712952, 0.1047484427690506, 0.07856133580207825, 0.9524866938591003, 0.05093872919678688, 0.9678358435630798, 0.1483815759420395, 0.043277960270643234, 0.8037335276603699, 0.9319223761558533, 0.7586779594421387, 0.06288722902536392, 0.1781804859638214, 0.04283053055405617, 0.04283053055405617, 0.9208564162254333, 0.04450143873691559, 0.8900288343429565, 0.06675215810537338, 0.9235218167304993, 0.060758013278245926, 0.012151602655649185, 0.07960384339094162, 0.015920767560601234, 0.9074838161468506, 0.8118337988853455, 0.11519509553909302, 0.07290828973054886, 0.7957450747489929, 0.16919496655464172, 0.03613017499446869, 0.40675973892211914, 0.2830779254436493, 0.30992358922958374, 0.030691983178257942, 0.9514514803886414, 0.970538318157196, 0.026230765506625175, 0.026230765506625175, 0.028892342001199722, 0.9534472823143005, 0.028892342001199722, 0.9027702212333679, 0.07364704459905624, 0.021381400525569916, 0.05029713734984398, 0.9053484797477722, 0.04023770987987518, 0.2841479778289795, 0.34400567412376404, 0.3721740245819092, 0.8994565606117249, 0.08384764194488525, 0.01778586395084858, 0.8902881741523743, 0.05048025771975517, 0.059658486396074295, 0.8260018229484558, 0.06020013615489006, 0.11340025067329407, 0.4104161262512207, 0.36173751950263977, 0.22765269875526428, 0.19049352407455444, 0.4421674311161041, 0.367082417011261, 0.9649859070777893, 0.013638537377119064, 0.013638537377119064, 0.9683361053466797, 0.9604122042655945, 0.03693893179297447, 0.15736821293830872, 0.6495398879051208, 0.1932777315378189, 0.7426649928092957, 0.1009722426533699, 0.15664853155612946, 0.03822304308414459, 0.9173530340194702, 0.03822304308414459, 0.8834613561630249, 0.051588982343673706, 0.06448622792959213, 0.2573416829109192, 0.16492176055908203, 0.5784212350845337, 0.04160512238740921, 0.04160512238740921, 0.9153127074241638, 0.9151319265365601, 0.8203229308128357, 0.09093201905488968, 0.08899729698896408, 0.5967563390731812, 0.1994432806968689, 0.2041638344526291, 0.036128923296928406, 0.9393519759178162, 0.018064461648464203, 0.5646959543228149, 0.3949827551841736, 0.04033542424440384, 0.9387165904045105, 0.40461719036102295, 0.1260763704776764, 0.4696100652217865, 0.02562917396426201, 0.9482793807983398, 0.955094575881958, 0.05723676458001137, 0.8967093229293823, 0.04769730195403099, 0.9565042853355408, 0.2971437871456146, 0.5428361892700195, 0.160018652677536, 0.31109294295310974, 0.5602540373802185, 0.12845128774642944, 0.012858972884714603, 0.977281928062439, 0.012858972884714603, 0.14832518994808197, 0.2886327803134918, 0.5625666975975037, 0.687161386013031, 0.25114527344703674, 0.06220496818423271, 0.2500121593475342, 0.2431936413049698, 0.506085216999054, 0.9390268325805664, 0.05351116135716438, 0.9632008671760559, 0.06919663399457932, 0.9687528610229492, 0.9277554154396057], "Term": ["access", "access", "access", "access!", "access!", "ace", "admire", "affordable", "affordable", "affordable", "alcohol", "alcohol", "also", "also", "also", "always", "always", "always", "amazing", "amazing", "amazing", "antelope", "apartment", "apartment", "apartment", "aquarium", "aquarium", "aquarium", "area", "area", "area", "ask", "ask", "ask", "available", "available", "available", "away", "away", "away", "barber", "bath", "bath", "bath", "bathroom", "bathroom", "bathroom", "beach", "beach", "beach", "bed", "bed", "bed", "bedroom", "bedroom", "bedroom", "best", "best", "best", "bike", "bike", "bike", "block", "block", "block", "boardwalk", "boardwalk", "boardwalk", "boiler", "book", "book", "book", "booked", "booked", "booked", "booking", "booking", "booking", "broad", "broad", "broad", "broil", "building", "building", "building", "building!", "cabin", "cabin", "cabin", "cable", "cable", "cable", "call", "call", "call", "capitan", "capitan", "capitan", "carefully!!!!!", "ceilings!", "center", "center", "center", "chat", "chat", "chat", "check", "check", "check", "china", "china", "city", "city", "city", "close", "close", "close", "closet", "closet", "closet", "clubhouse", "clubhouse", "coffee", "coffee", "coffee", "colima", "comforter", "comforter", "comforter", "commuting", "concert", "concert", "concert", "convention", "convention", "convention", "cooling", "cooling", "copier", "couches!", "counter", "counter", "counter", "cup", "cup", "cup", "dash", "dash", "day", "day", "day", "days!", "days!", "dining", "dining", "dining", "dishwasher", "dishwasher", "dishwasher", "distance", "distance", "distance", "district", "district", "district", "divided", "divided", "divided", "dodger", "dodger", "dodger", "doorman", "doorstep!", "downtown", "downtown", "downtown", "dtla!", "echo", "echo", "echo", "efficiency", "efficiency", "efficiency", "elm", "emergency!", "enjoy", "enjoy", "enjoy", "epicenter", "epicenter", "epicenter", "everything", "everything", "everything", "fame", "fame", "fame", "famous", "famous", "famous", "feminine", "financial", "financial", "floor", "floor", "floor", "fob", "fob", "food", "food", "food", "football", "formal", "formal", "formal", "foyer", "foyer", "foyer", "fragrant", "free", "free", "free", "full", "full", "full", "fully", "fully", "fully", "furnished!!!", "generously", "generously", "glamp!", "gleaming", "gleaming", "granite", "granite", "granite", "great", "great", "great", "group!", "guest", "guest", "guest", "gym", "gym", "gym", "hair", "hair", "hair", "hammer", "hangout", "hangout", "hangout", "heart", "heart", "heart", "hidden", "hidden", "hidden", "hive!", "home", "home", "home", "hosting", "hosting", "hosting", "hotel!", "hotel!", "house", "house", "house", "humble", "inch", "inch", "inch", "incredible", "incredible", "incredible", "iron", "iron", "iron", "journey!", "junction", "junction", "junction", "king", "king", "king", "kitchen", "kitchen", "kitchen", "know", "know", "know", "la", "la", "la", "lake", "lake", "lake", "large", "large", "large", "like", "like", "like", "living", "living", "living", "location", "location", "location", "long", "long", "long", "lot!!", "love", "love", "love", "love!", "madam", "mall", "mall", "mall", "manor", "manor", "manor", "many", "many", "many", "marina", "marina", "marina", "mart", "mart", "mary", "mary", "mary", "master", "master", "master", "mattress", "mattress", "mattress", "may", "may", "may", "mi", "mi", "mi", "microwave", "microwave", "microwave", "min", "min", "min", "mind", "mind", "mind", "mirror", "mirror", "mirror", "mirrored", "mirrored", "mirrored", "modern", "modern", "modern", "more!!", "more!!", "mountain", "mountain", "mountain", "museum", "museum", "museum", "need", "need", "need", "neighborhood", "neighborhood", "neighborhood", "new", "new", "new", "number", "number", "number", "ocean", "ocean", "ocean", "one", "one", "one", "osmosis", "osmosis", "osmosis", "oven", "oven", "oven", "pan", "pan", "pan", "park", "park", "park", "parking", "parking", "parking", "phone", "phone", "phone", "pier", "pier", "pier", "place", "place", "place", "please", "please", "please", "positive", "positive", "positive", "possible!", "possible!", "posted", "posted", "powder", "powder", "powder", "prestige", "preview", "preview", "printer", "printer", "printer", "private", "private", "private", "quartz", "quartz", "quartz", "queen", "queen", "queen", "questions!", "questions!", "quiet", "quiet", "quiet", "read", "read", "read", "recycle", "recycle", "refrigerator", "refrigerator", "refrigerator", "regency", "regency", "renovated!", "renovated!!", "respect", "respect", "respect", "rise", "rise", "rise", "room", "room", "room", "rose", "rose", "rose", "sanctuary", "sanctuary", "sanctuary", "sand", "sand", "sand", "scanner", "screen", "screen", "screen", "second", "second", "second", "see!", "selected", "selected", "send", "send", "send", "shelving", "shower", "shower", "shower", "shuttle", "shuttle", "shuttle", "silver", "silver", "silver", "silverware", "silverware", "silverware", "similar", "similar", "similar", "size", "size", "size", "sofa", "sofa", "sofa", "space", "space", "space", "specifically", "specifically", "spoken", "spoken", "spoken", "spots!", "spots!", "spots!", "stainless", "stainless", "stainless", "staple", "staple", "staple", "stay", "stay", "stay", "steel", "steel", "steel", "storage", "storage", "storage", "stove", "stove", "stove", "street", "street", "street", "studio", "studio", "studio", "studio!!!!", "studios!", "studios!", "studios!", "suitcase", "suitcase", "sunset", "sunset", "sunset", "table", "table", "table", "tanning", "tanning", "tanning", "tile", "tile", "tile", "time", "time", "time", "tiny", "tiny", "tiny", "touristic", "twin", "twin", "twin", "two", "two", "two", "union", "union", "union", "unit", "unit", "unit", "unlock", "use", "use", "use", "vary", "vary", "velvet", "vine", "vine", "vine", "waking", "walk", "walk", "walk", "walking", "walking", "walking", "walt", "walt", "walt", "want", "want", "want", "washer", "washer", "washer", "welcome", "welcome", "welcome", "wellness", "whose", "whose", "windward", "windward", "yurt"]}, "R": 30, "lambda.step": 0.01, "plot.opts": {"xlab": "PC1", "ylab": "PC2"}, "topic.order": [2, 1, 3]};

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
       new LDAvis("#" + "ldavis_el6071057821599129736726849", ldavis_el6071057821599129736726849_data);
   }(LDAvis);
}else if(typeof define === "function" && define.amd){
   // require.js is available: use it to load d3/LDAvis
   require.config({paths: {d3: "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min"}});
   require(["d3"], function(d3){
      window.d3 = d3;
      LDAvis_load_lib("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js", function(){
        new LDAvis("#" + "ldavis_el6071057821599129736726849", ldavis_el6071057821599129736726849_data);
      });
    });
}else{
    // require.js not available: dynamically load d3 & LDAvis
    LDAvis_load_lib("https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js", function(){
         LDAvis_load_lib("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js", function(){
                 new LDAvis("#" + "ldavis_el6071057821599129736726849", ldavis_el6071057821599129736726849_data);
            })
         });
}
</script>



pyLDAvis package is a great package to visualize the LDA model. The area of the circles means the prevalence of each topic. Here I chose the cluster the corpus into three topics. The red bar represents the estimated term frequency within selected topic and the blue bar represents the overall term frequency. In topic 1, the prevalent term is about layout of the room, for example, there are words "kitchen", "bathroom", "bedroom". Topic 2 is about the living environment because it has words "new","private","space","large". Topic 3 is correlated with location or transit with words "subway", "walk","away". There are some overlaps among these three topics, which can be improved to better serve the machine learninng model. At this moment, I will go ahead with the current model.


## IV. Machine learning model

In this part, __random forest algorithm__ will be applied to the clean dataset and predict the yield. Random forest model can capture the nonlinear relationships in the dataset and it is a complex model to provide high accuracy. 

To measure the accuracy of the model, MSE (mean squared error) is used as evaluation metrics. The target for prediction is "yield". "price" and "reviews_per_month", "average_length_of_stay" have strong correlation with "yield" because they are used for yield calculation. Catergorical features also need to be converted to numerical features so that they can be fed into machine learning algorithms. To split the whole dataset into a training set and a testing set, the dataset will be randomly shuffled first and 25% will be used as the splitting ratio.

Once the model have been applied and trained, they will be compared based on MSE value. The smaller MSE, the better accuracy. The best algorithm will be chosen and the model will be further fine-tuned using GridSearchCV function in scikit-learn. The to-do list in this part is:

1. Clean-up the dataset: separate the "yield" from the dataset and save it as the target, drop "price", "average_length_of_stay" and "reviews_per_month", convert catergorical variables into numerical features. Other columns including "level_0", "id", "listing_url", "description","image_link" can be dropped as well since they are not needed any more.
2. In order to evaluate how the two new features captured from photo and description of the listings can improve the performance of the model, both models with and without these two new features will be built.
3. Randomly shuffle the dataset to remove inherent order and split the dataset into a training set and a test set using 75:25 ratio.
4. Use linear regression, decision tree, and random forest separately to train the model and calculate the MSE value.
5. Select the model with lowest MSE value for further refinement.



```python
final_df =  pd.read_csv('final_df_with_topicmodel.csv')
```


```python
# features to keep
cols_to_keep = ['latitude','longitude','accommodates','bathrooms','bedrooms','guests_included','extra_people',
                'maximum_nights','property_type','room_type','bed_type','cancellation_policy','yield']
model_df = final_df[cols_to_keep]

# convert strings to dummies
categorical_feats = ['property_type','room_type','bed_type','cancellation_policy']
model_df = pd.get_dummies(model_df,columns = categorical_feats,drop_first = False)

# separate the target variable "yield" from the dataset
target = model_df['yield']
X_df = model_df.drop(['yield'],axis = 1)
```

### Random forest


```python
# split the training set and testing set
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
seed = 42
X_train,X_test,y_train,y_test = train_test_split(X_df,target,random_state=seed)
```


```python
from sklearn.ensemble import RandomForestRegressor

for K in [1,3,5,7,10,15,20]:
    rf_reg = RandomForestRegressor(random_state = seed, bootstrap = True, criterion = 'mse', max_depth = K,
                                   max_features = 'auto', min_samples_split = 5, n_estimators = 150).fit(X_train, y_train)
    y_rf_pred = rf_reg.predict(X_test)

    print ("Max_depth = " + str(K))
    print("Mean squared error: %.3f" %mean_squared_error(y_test,y_rf_pred))
    print("Variance score: %.3f" %r2_score(y_test,y_rf_pred))
```

    Max_depth = 1
    Mean squared error: 4235835851.867
    Variance score: 0.060
    Max_depth = 3
    Mean squared error: 4219722807.763
    Variance score: 0.063
    Max_depth = 5
    Mean squared error: 4230893974.845
    Variance score: 0.061
    Max_depth = 7
    Mean squared error: 4242613796.671
    Variance score: 0.058
    Max_depth = 10
    Mean squared error: 4229651455.101
    Variance score: 0.061
    Max_depth = 15
    Mean squared error: 4216468094.997
    Variance score: 0.064
    Max_depth = 20
    Mean squared error: 4213626070.418
    Variance score: 0.065


### Implementation: Incorporating the two engineered features

Two new features engineered from photos and descriptions of the houses will be incorporated into the model. 

### Random forest


```python
# drop unnecessary columns
cols_to_keep =  ['latitude','longitude','accommodates','bathrooms','bedrooms','guests_included','extra_people',
                'maximum_nights','property_type','room_type','bed_type','cancellation_policy','NIMA_score','description_topic','yield']
refine_df = final_df[cols_to_keep]

# convert strings to numerical features
categorical_feats = ['property_type', 'room_type', 'bed_type', 'cancellation_policy','description_topic' ]
refine_df = pd.get_dummies(refine_df, columns = categorical_feats, drop_first = False)

# separate the target variable "yield" from the dataset
target = refine_df['yield']
refine_df = refine_df.drop(['yield'], axis = 1)
```


```python
# split the training set and testing set
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
seed = 42
X_train,X_test,y_train,y_test = train_test_split(refine_df,target,random_state=seed)
```


```python
from sklearn.ensemble import RandomForestRegressor

for K in [1,3,5,7,10,15,20]:
    rf_reg = RandomForestRegressor(random_state = seed, bootstrap = True, criterion = 'mse', max_depth = K,
                                   max_features = 'auto', min_samples_split = 5, n_estimators = 150).fit(X_train, y_train)
    y_rf_pred = rf_reg.predict(X_test)

    print ("Max_depth = " + str(K))
    print("Mean squared error: %.3f" %mean_squared_error(y_test,y_rf_pred))
    print("Variance score: %.3f" %r2_score(y_test,y_rf_pred))
```

    Max_depth = 1
    Mean squared error: 4235835851.867
    Variance score: 0.060
    Max_depth = 3
    Mean squared error: 4226199045.345
    Variance score: 0.062
    Max_depth = 5
    Mean squared error: 4241483004.705
    Variance score: 0.058
    Max_depth = 7
    Mean squared error: 4238381313.703
    Variance score: 0.059
    Max_depth = 10
    Mean squared error: 4214493407.211
    Variance score: 0.064
    Max_depth = 15
    Mean squared error: 4192146357.324
    Variance score: 0.069
    Max_depth = 20
    Mean squared error: 4184092191.506
    Variance score: 0.071


By comparing the MSE score and the variance score, after adding the two new features: NIMA_score and description_topic, the performance of the random forest model got improved.

##  V. Fine tuning and model evaluation

As expected, random forest gave the lowest MSE and highest variance score. So this part will focusing on fine tuning the model and test how robust the model is. It  will be structured into three parts:

1. Using GridSearchCV to fine tuning the model using random forest regressor.
2. Check the importance of each feature in the dataset, especially the two features from image analysis and text mining.
3. Test the robustness of the model by using a different seed.


```python
from sklearn.model_selection import GridSearchCV

param_grid = {"n_estimators" :[150,175,200,225,250,300],
             "criterion": ['mse'],
             "max_features": ['auto'],
             "max_depth": [5,7,9,11,15,20,25,30],
             "min_samples_split":[4,6,8,10,12],
             "bootstrap":[True]}

rf_fine = RandomForestRegressor(random_state = seed)
rf_cv = GridSearchCV(rf_fine,param_grid,cv=5).fit(X_train,y_train)
y_rf_cv_pred = rf_cv.predict(X_test)
print("Mean squared error: %.3f" % mean_squared_error(y_test, y_rf_cv_pred))
print('Variance score: %.3f' % r2_score(y_test, y_rf_cv_pred))
print("Best Parameters: {}".format(rf_cv.best_params_))
```

    Mean squared error: 4177564595.927
    Variance score: 0.073
    Best Parameters: {'bootstrap': True, 'criterion': 'mse', 'max_depth': 25, 'max_features': 'auto', 'min_samples_split': 12, 'n_estimators': 225}



```python
rf_final = rf_cv.best_estimator_
feature_import = rf_final.feature_importances_*100
feature_import = pd.DataFrame(list(zip(feature_import,X_train.columns.values)))
feature_import = feature_import.sort_values(by=0,axis=0,ascending=False)
feature_import.columns = ['importance %','feature']
print(feature_import[:20])
```

        importance %                                          feature
    8      23.591967                                       NIMA_score
    1      19.125749                                        longitude
    0      17.729351                                         latitude
    26     10.940065                        room_type_Entire home/apt
    2       6.383180                                     accommodates
    6       4.358200                                     extra_people
    7       2.714976                                   maximum_nights
    3       2.674377                                        bathrooms
    5       1.931795                                  guests_included
    4       1.883962                                         bedrooms
    13      1.800505                        property_type_Condominium
    9       0.924392                          property_type_Apartment
    34      0.753243                     cancellation_policy_flexible
    40      0.726294                              description_topic_1
    20      0.705806                              property_type_Other
    35      0.684130                     cancellation_policy_moderate
    41      0.615718                              description_topic_2
    37      0.441630  cancellation_policy_strict_14_with_grace_period
    21      0.398022                 property_type_Serviced apartment
    19      0.358925                               property_type_Loft



```python
features = feature_import['feature']
importances = feature_import['importance %']

fig,ax = plt.subplots(figsize=(14,16))
y_pos = np.arange(len(features))
ax.barh(y_pos, importances, align='center',color='green')
ax.set_yticks(y_pos)
ax.set_yticklabels(features)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Importance %')
ax.set_title('Feature Importance')
save_fig("Feature_Importance")

plt.show()
```

    Saving figure Feature_Importance



![png](output_50_1.png)


From the feature importance rank, it's very nice to see that __NIMA score__ has a importance of __23.59%__ and it is the most important feature in this model. Location has a combined importance of 36.86% - 19.13% from longitude and 17.73% from latitude, which make sense to me. A convenient location can be very attractive for viewers. Other features such as "accommodates" also occupied 6.38% importance. The other feature __"description_topic"__ also has combined __>1%__ of importance (sum of "description_topic_1" and "description_topic_2"). This information shows that there are valuable information in the photo and description text.

To test the robustness of the model, random_state for shuffling dataset will be changed, the ratio of training set and test set will also be changed to 0.2. 


```python
random_state = 65
X_train,X_test,y_train,y_test = train_test_split(refine_df,target,test_size = 0.2,random_state=seed)
```


```python
# Fit and make prediction
rf_final.fit(X_train, y_train)
rf_y_pred = rf_final.predict(X_test)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, rf_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, rf_y_pred))
```

    Mean squared error: 4046491479.02
    Variance score: 0.06


There is no significant difference after adjusting the random state and proportions of training set and test set, which demonstrate that the final model is robust.

## VI. Conclusion and reflection

The original goal of this project is to apply machine learning algorithms to give potential hosts some insights on how much they can earn from listing their beloved houses on Airbnb. The information from Inside Airbnb is definitely very helpful. Combined my own experience of browsing accommodations in Airbnb, I added two additional features: image score and topic modeling from web photos and descriptions. It turned out that these two features actually contain lots of valuable informations and play important roles when building machine learning models. Of couse, my solution is not perfect, here are two points I would like to spend more time on further improving my model.

1. There are some overlaps among the three topics, so potential improvement would be implement the topic modeling methods. It would be worthwhile comparing LDA with other algorithms, such as Non-negative matrix factorization.

2. Should I consider time effect? If a host gets very positive reviews from first few guests, it's possible that new viewers will also consider choosing their houses. How should I predict time series?

3. Try other machine learning models.

4. How to convert this to a real data product? A web application ,which allow potential hosts to upload their photos and other informations about their houses so that they can check the expected income given their input information, can be the next step.

