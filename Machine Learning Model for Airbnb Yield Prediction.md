
##  I. Introduction

Airbnb is a great platform that provides people online marketplace and service to arrange or offer lodging. As a travel enthusiast, Airbnb is always my first choice when I am planning a trip. Hosts need to provide details for their listed houses so that guests can use filters on the website to search for their preferred accomodations. For potential hosts, they must be very interested in how much they could earn from listing their houses on Airbnb. As far as I know, there is no such a model in public for predicting the yield of a new house on Airbnb. So, the object of this project is to apply machine learning models to help potential hosts gain some intuitions about the yield of their listed houses.

Fortunately, [Inside Airbnb](http://insideairbnb.com/get-the-data.html) has already aggregated all the publicly available informations from Airbnb site for public discussion. So, the dataset obtained from this website directly should be a good starting point for my machine learning model. In particular, I will the dataset collected in New York city compiled on 06 December, 2018. When selecting features for machine learning model, besides the variables provided in the datasets, the featured photo on the listing's website and the description of listing can be crucial for attracting more guests. So, I will analyze featured photos and text mining on the descriptions and add these two new features to improve the machine learning model. 

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

    There are 49056 rows and 96 columns in the dataset


    /Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/IPython/core/interactiveshell.py:2728: DtypeWarning: Columns (61,87,88) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)





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
sns.distplot(pd.Series(new_df['price'], name = "Price"), color = "purple",ax = axs[1,0])

new_df = new_df[new_df['availability_365']>10]

sns.distplot(pd.Series(new_df['availability_365'],name = "Availability during a Year (After cleaning)"), color ="r",ax = axs[0,1])
sns.distplot(pd.Series(new_df['bedrooms'],name = "Number of bedrooms"),color = "y", ax = axs[1,1])
sns.distplot(pd.Series(new_df['bathrooms'],name = "Number of bathrooms"),color = 'blue', ax = axs[2,1])
sns.distplot(pd.Series(new_df['cleaning_fee'], name = "Cleaning_fee"), color = "orange",ax = axs[2,0])

save_fig("Distribution_of_variables")

print ("Dataset has {} rows and {} columns.".format(*new_df.shape))
```

    Saving figure Distribution_of_variables
    Dataset has 25194 rows and 20 columns.



![png](output_9_1.png)


After cleaning up the data, the new dataset now has 30372 rows and 21 columns without any missing values.

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
cleaned_listings = new_df.to_csv()
```

## III. Feature engineering

### Image analysis on featured photos

In most cases, hosts on Airbnb will upload some photos of their houses. These photos, especially the featured photo on the website, are extremely important to attract more viewers. An ideal photo should has desirable resolution and also be aesthetically attractive. Here I will use __[NIMA: Neural Image Assessment](https://ai.googleblog.com/2017/12/introducing-nima-neural-image-assessment.html)__ to score image quality. In NIMA, a deep convolutional neural network (CNN) is trained to predict whether the image will rated by a viewer as looking good (technically) and attractive (aesthetically). 

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
# extract the url for the feature photo from 'listings_url'
listings = new_df['listing_url']
image_link = {}
for file_url in listings:
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
# take random samples
sample = df_image['image_link'][42]
photo_id = df_image['id'][42]
image_name = os.path.join(Photo_path, str(photo_id)+str('.jpg'))
img = scipy.misc.imread(image_name)
plt.imshow(img)       
```




    <matplotlib.image.AxesImage at 0x1059bd7f0>




![png](output_20_1.png)



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
```


```python
# add NIMA_score to new_df
new_df['NIMA_score'] = new_df['id'].map(NIMA_dic)
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
            if len(word) > 2:
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
p_description
```





<link rel="stylesheet" type="text/css" href="https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.css">


<div id="ldavis_el253286103006928447223460"></div>
<script type="text/javascript">

var ldavis_el253286103006928447223460_data = {"mdsDat": {"Freq": [34.92222595214844, 33.25605010986328, 31.821727752685547], "cluster": [1, 1, 1], "topics": [1, 2, 3], "x": [-0.008218187045556078, -0.0938564911706685, 0.1020746782162246], "y": [-0.09225138541413617, 0.051929816238694686, 0.04032156917544154]}, "tinfo": {"Category": ["Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3"], "Freq": [10722.0, 28305.0, 11397.0, 9902.0, 11609.0, 5693.0, 9669.0, 5436.0, 4595.0, 14352.0, 3591.0, 4666.0, 4530.0, 7497.0, 6973.0, 7409.0, 4732.0, 4698.0, 5642.0, 4458.0, 3070.0, 3958.0, 7090.0, 5106.0, 5893.0, 8431.0, 2351.0, 4637.0, 6842.0, 10792.0, 74.06208801269531, 65.84806823730469, 50.992706298828125, 60.62697219848633, 52.70991897583008, 86.46465301513672, 82.6775894165039, 38.77629470825195, 41.60611343383789, 35.05453109741211, 45.47275161743164, 50.28895568847656, 52.31075668334961, 93.3344955444336, 35.00722885131836, 54.97221755981445, 34.99530792236328, 30.47829818725586, 38.7076416015625, 59.01118850708008, 30.42500877380371, 71.98351287841797, 59.26739501953125, 25.70056915283203, 26.244163513183594, 31.767229080200195, 40.978858947753906, 26.01799964904785, 67.31427001953125, 31.165752410888672, 141.25723266601562, 82.1343002319336, 75.88694763183594, 42.4832649230957, 741.1898803710938, 332.9090881347656, 645.69873046875, 136.09423828125, 93.52503967285156, 987.0364990234375, 3991.68115234375, 285.3797912597656, 96.25721740722656, 2097.676025390625, 1515.4765625, 79.04544830322266, 1452.0977783203125, 804.152099609375, 532.8049926757812, 564.54638671875, 3530.50732421875, 672.5762939453125, 413.2572326660156, 3572.889404296875, 781.9086303710938, 755.0298461914062, 461.9676818847656, 231.2172088623047, 1419.1727294921875, 691.8629150390625, 328.8019104003906, 18350.8125, 677.4935913085938, 475.94580078125, 1037.243896484375, 736.0682373046875, 1411.576904296875, 1314.5538330078125, 1179.9927978515625, 3890.59033203125, 7909.04150390625, 7003.47998046875, 1732.93505859375, 3064.7333984375, 5927.138671875, 1920.6099853515625, 4765.08642578125, 1877.7156982421875, 2196.99755859375, 8545.9248046875, 6340.95751953125, 2665.506103515625, 3062.291259765625, 3395.697998046875, 3963.54345703125, 4600.830078125, 1783.8363037109375, 5623.90185546875, 6277.30615234375, 8384.5712890625, 3556.17138671875, 3375.843017578125, 3896.35888671875, 3330.490234375, 2648.33203125, 2941.9873046875, 3451.815673828125, 2641.10205078125, 2561.0859375, 2629.232666015625, 43.458919525146484, 111.22531127929688, 70.02086639404297, 40.0068473815918, 32.386661529541016, 185.20814514160156, 77.56222534179688, 39.02825927734375, 449.7890625, 40.9261360168457, 26.885433197021484, 38.84908676147461, 28.707426071166992, 27.425058364868164, 48.96398162841797, 25.23552703857422, 25.80536460876465, 24.473081588745117, 20.704256057739258, 19.781803131103516, 21.097938537597656, 25.174842834472656, 18.243139266967773, 31.880075454711914, 51.98185729980469, 17.083303451538086, 22.156036376953125, 53.30710983276367, 811.2877807617188, 16.87030601501465, 663.2855224609375, 277.31512451171875, 261.62335205078125, 176.30943298339844, 41.94338607788086, 1538.6473388671875, 237.58265686035156, 125.83845520019531, 99.18772888183594, 3136.587646484375, 4818.87255859375, 200.10447692871094, 64.6158676147461, 560.1602172851562, 3873.20751953125, 4483.05029296875, 8510.845703125, 3695.315673828125, 1920.08837890625, 8671.2607421875, 1008.3635864257812, 3701.05517578125, 553.5654907226562, 1618.3779296875, 7365.17333984375, 8492.677734375, 6992.80908203125, 1188.9490966796875, 728.3868408203125, 1543.620849609375, 4031.412109375, 1241.4359130859375, 5138.740234375, 1199.824951171875, 1490.5672607421875, 1879.374267578125, 5230.56005859375, 2137.8984375, 2238.6923828125, 5389.70751953125, 3445.38525390625, 3489.275634765625, 2218.736328125, 2712.93701171875, 3914.6005859375, 2373.45703125, 10574.8564453125, 3600.759765625, 2623.080078125, 3806.459716796875, 3494.766845703125, 3315.42041015625, 3884.9287109375, 3677.89404296875, 3403.453857421875, 2902.92431640625, 3910.34814453125, 3641.18603515625, 3791.086181640625, 2948.770751953125, 309.6786804199219, 908.9498901367188, 312.6562805175781, 968.670654296875, 156.47337341308594, 344.8201904296875, 84.12334442138672, 545.2188720703125, 142.5908966064453, 78.91434478759766, 233.50379943847656, 79.17054748535156, 90.48151397705078, 223.58946228027344, 61.928836822509766, 67.64070129394531, 59.94719696044922, 49.87224578857422, 110.02108001708984, 126.1849594116211, 123.28962707519531, 45.11228561401367, 49.64248275756836, 298.82720947265625, 57.61788558959961, 64.25823974609375, 134.75790405273438, 45.81895065307617, 99.78255462646484, 77.8526382446289, 1423.87548828125, 126.18067169189453, 212.70391845703125, 319.98193359375, 121.47489166259766, 721.4634399414062, 405.0027160644531, 433.0552673339844, 134.55775451660156, 1563.54931640625, 908.2807006835938, 1207.1944580078125, 761.1774291992188, 2075.18603515625, 1227.8416748046875, 529.0511474609375, 326.7619323730469, 278.90924072265625, 2528.270263671875, 487.5666198730469, 1286.967529296875, 1007.7186889648438, 1038.978759765625, 968.5419311523438, 1197.9112548828125, 3001.18408203125, 3479.934326171875, 657.010498046875, 5019.20166015625, 3563.095703125, 4649.6572265625, 1689.402099609375, 1896.3232421875, 1022.79931640625, 1699.59716796875, 1636.9390869140625, 7418.72802734375, 4026.444091796875, 4871.24462890625, 5570.35498046875, 8544.6083984375, 11995.4326171875, 5664.83984375, 2501.336181640625, 4312.537109375, 7372.365234375, 1860.0738525390625, 5497.96533203125, 1912.6656494140625, 2766.18017578125, 3713.686279296875, 6163.11181640625, 2755.910400390625, 3417.180908203125, 2782.68896484375, 3193.49365234375, 3249.027587890625, 2565.5380859375, 2136.448486328125, 2103.1728515625, 2236.740966796875, 2100.597412109375], "Term": ["train", "room", "away", "park", "walk", "min", "subway", "walking", "please", "bed", "bus", "minute", "distance", "large", "queen", "close", "station", "high", "block", "use", "sofa", "modern", "size", "fully", "central", "street", "loft", "guest", "place", "full", "men", "lockable", "mingle", "cereal", "snack", "twice", "hostel", "thoroughly", "meant", "guesthouse", "encourage", "permanent", "background", "normally", "puppy", "leasing", "conversation", "polite", "concern", "review", "hygiene", "locker", "accordingly", "ticket", "took", "brooklyn!!", "dorm", "alcohol", "follow", "dinette", "driveway", "loving", "certain", "subject", "lock", "reservation", "booking", "tidy", "booked", "keep", "please", "move", "understand", "check", "share", "advised", "may", "shampoo", "read", "soap", "use", "book", "luggage", "guest", "listing", "ask", "fee", "female", "common", "let", "flexible", "room", "conditioner", "however", "would", "know", "provide", "privacy", "host", "clean", "private", "bathroom", "work", "house", "space", "welcome", "home", "make", "people", "kitchen", "living", "free", "need", "stay", "also", "access", "small", "bed", "bedroom", "apartment", "available", "two", "full", "area", "comfortable", "place", "one", "time", "size", "floor", "shrine", "boardwalk", "riverbank", "luna", "arent", "parkway", "domino", "expressway", "botanical", "caught", "distant", "blink", "marina", "sheepshead", "aquarium", "cooper", "bier", "numb", "eve", "shops!", "highland", "midtown!", "shore", "bars!", "bodega", "ride!", "diversified", "tattoo", "beach", "bowling", "mall", "zoo", "rockaway", "met", "boat", "museum", "cab", "path", "gateway", "bus", "min", "target", "pizzeria", "taxi", "minute", "walking", "train", "distance", "stop", "away", "express", "station", "shop", "ride", "park", "walk", "subway", "prospect", "supermarket", "ave", "block", "across", "close", "grocery", "public", "transportation", "street", "shopping", "within", "neighborhood", "square", "central", "many", "location", "great", "around", "apartment", "place", "east", "city", "time", "available", "new", "one", "access", "quiet", "bedroom", "kitchen", "room", "building", "flooring", "steel", "custom", "stainless", "oak", "granite", "cabinetry", "marble", "housekeeping", "parquet", "para", "reception", "sleek", "oversized", "soaring", "backed", "hay", "unparalleled", "soaking", "tiled", "expansive", "plank", "blazing", "counter", "craving", "burner", "sectional", "playroom", "architectural", "glassware", "dishwasher", "tile", "rise", "concierge", "entertaining", "original", "fireplace", "sleeper", "decorative", "luxury", "exposed", "screen", "wood", "loft", "hardwood", "designer", "contemporary", "industrial", "sofa", "stunning", "flat", "master", "brick", "deck", "throughout", "modern", "high", "designed", "large", "fully", "queen", "top", "brand", "oven", "washer", "unit", "bed", "size", "building", "full", "bedroom", "apartment", "new", "dining", "floor", "kitchen", "table", "living", "furnished", "spacious", "one", "room", "beautiful", "space", "two", "bathroom", "private", "area", "heart", "central", "park", "street"], "Total": [10722.0, 28305.0, 11397.0, 9902.0, 11609.0, 5693.0, 9669.0, 5436.0, 4595.0, 14352.0, 3591.0, 4666.0, 4530.0, 7497.0, 6973.0, 7409.0, 4732.0, 4698.0, 5642.0, 4458.0, 3070.0, 3958.0, 7090.0, 5106.0, 5893.0, 8431.0, 2351.0, 4637.0, 6842.0, 10792.0, 74.9241943359375, 66.73881530761719, 51.73693084716797, 61.56693649291992, 53.68115234375, 88.22702026367188, 84.44430541992188, 39.628570556640625, 42.558982849121094, 35.86351776123047, 46.542747497558594, 51.476322174072266, 53.58339309692383, 95.64986419677734, 35.910343170166016, 56.393455505371094, 35.95586013793945, 31.31802749633789, 39.79114532470703, 60.66679763793945, 31.29671859741211, 74.06916809082031, 60.996742248535156, 26.4681396484375, 27.040918350219727, 32.753150939941406, 42.25129699707031, 26.841100692749023, 69.48886108398438, 32.18136978149414, 146.2718505859375, 85.14607238769531, 78.69145965576172, 43.909461975097656, 792.9243774414062, 352.68988037109375, 695.7412109375, 142.58827209472656, 97.84286499023438, 1085.9285888671875, 4595.9169921875, 306.3241882324219, 101.21002960205078, 2431.337646484375, 1756.2862548828125, 82.78131103515625, 1733.2125244140625, 931.5406494140625, 607.27978515625, 645.6688232421875, 4458.53173828125, 778.8878784179688, 469.13153076171875, 4637.6240234375, 923.7637939453125, 895.2113037109375, 531.4415283203125, 255.132568359375, 1791.3580322265625, 830.8260498046875, 374.0018615722656, 28305.009765625, 821.39501953125, 559.6119995117188, 1306.3656005859375, 913.1748046875, 1882.3160400390625, 1768.3389892578125, 1571.075439453125, 6047.14599609375, 13661.2314453125, 12394.978515625, 2457.317626953125, 4806.9462890625, 11238.115234375, 2893.93115234375, 8699.189453125, 2828.8291015625, 3487.614501953125, 19559.4765625, 13542.4033203125, 4484.59765625, 5516.29638671875, 6361.45068359375, 7925.361328125, 9933.7783203125, 2752.28662109375, 14352.841796875, 18732.26171875, 30954.859375, 8209.0185546875, 8109.11279296875, 10792.361328125, 8562.3818359375, 5514.4501953125, 6842.7216796875, 10843.396484375, 7158.77587890625, 7090.32421875, 8462.5751953125, 44.203556060791016, 113.16128540039062, 71.42603302001953, 40.84333419799805, 33.15556335449219, 190.43719482421875, 79.80096435546875, 40.190914154052734, 463.5207824707031, 42.18195724487305, 27.73054313659668, 40.089942932128906, 29.634273529052734, 28.34779930114746, 50.64997863769531, 26.12206268310547, 26.732463836669922, 25.401817321777344, 21.536069869995117, 20.591585159301758, 21.977731704711914, 26.25442123413086, 19.04581069946289, 33.314239501953125, 54.32953643798828, 17.85974884033203, 23.179834365844727, 55.77927017211914, 849.0616455078125, 17.662216186523438, 700.6010131835938, 290.9854736328125, 276.73297119140625, 186.4053497314453, 43.98021697998047, 1709.0509033203125, 256.1028137207031, 133.95559692382812, 105.5755615234375, 3591.427978515625, 5693.162109375, 219.3142547607422, 68.52921295166016, 635.3048706054688, 4666.46923828125, 5436.36279296875, 10722.017578125, 4530.5869140625, 2314.9716796875, 11397.544921875, 1195.8717041015625, 4732.07861328125, 641.7493896484375, 2000.3358154296875, 9902.96484375, 11609.7490234375, 9669.361328125, 1478.4583740234375, 881.93603515625, 1980.5413818359375, 5642.46240234375, 1571.8017578125, 7409.56103515625, 1519.5799560546875, 1963.696044921875, 2556.10546875, 8431.49609375, 3085.021728515625, 3293.573974609375, 9419.01171875, 5674.55615234375, 5893.7783203125, 3369.884521484375, 4440.44091796875, 7376.1787109375, 3807.856201171875, 30954.859375, 6842.7216796875, 4382.1103515625, 7642.2626953125, 7158.77587890625, 8209.0185546875, 12073.9052734375, 10843.396484375, 9933.7783203125, 6679.2333984375, 18732.26171875, 19559.4765625, 28305.009765625, 9060.818359375, 310.93634033203125, 912.8560791015625, 314.3395080566406, 974.1853637695312, 157.40208435058594, 347.24169921875, 84.81884765625, 549.767333984375, 143.93527221679688, 79.7361831665039, 235.9411163330078, 79.99761962890625, 91.47528839111328, 226.09658813476562, 62.63115692138672, 68.43549346923828, 60.681339263916016, 50.5767822265625, 111.64178466796875, 128.08840942382812, 125.29788208007812, 45.853965759277344, 50.474586486816406, 303.9657287597656, 58.63755416870117, 65.41585540771484, 137.2024688720703, 46.67349624633789, 101.65480041503906, 79.33821105957031, 1480.5732421875, 128.80654907226562, 218.443359375, 330.5320739746094, 124.42533111572266, 757.8299560546875, 420.7275390625, 453.56011962890625, 138.09695434570312, 1714.9248046875, 986.799560546875, 1334.86181640625, 826.9479370117188, 2351.45654296875, 1365.3795166015625, 578.6024169921875, 348.9314270019531, 295.5924377441406, 3070.873046875, 535.3238525390625, 1509.4036865234375, 1167.575439453125, 1208.919677734375, 1121.78271484375, 1421.2403564453125, 3958.340087890625, 4698.47216796875, 753.1785888671875, 7497.837890625, 5106.958984375, 6973.8740234375, 2270.0302734375, 2602.30126953125, 1266.5701904296875, 2319.497314453125, 2257.70263671875, 14352.841796875, 7090.32421875, 9060.818359375, 10792.361328125, 18732.26171875, 30954.859375, 12073.9052734375, 4031.94091796875, 8462.5751953125, 19559.4765625, 2759.41796875, 13542.4033203125, 2916.7177734375, 5700.8798828125, 10843.396484375, 28305.009765625, 6673.03515625, 11238.115234375, 8109.11279296875, 12394.978515625, 13661.2314453125, 8562.3818359375, 4837.126953125, 5893.7783203125, 9902.96484375, 8431.49609375], "loglift": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 1.0405000448226929, 1.038599967956543, 1.037600040435791, 1.0367000102996826, 1.0338000059127808, 1.0319000482559204, 1.030900001525879, 1.0303000211715698, 1.0293999910354614, 1.0291999578475952, 1.0288000106811523, 1.0286999940872192, 1.027999997138977, 1.027500033378601, 1.0266000032424927, 1.0264999866485596, 1.024999976158142, 1.024899959564209, 1.024399995803833, 1.024399995803833, 1.023800015449524, 1.0234999656677246, 1.023300051689148, 1.0226000547409058, 1.0220999717712402, 1.0214999914169312, 1.0214999914169312, 1.020900011062622, 1.020300030708313, 1.0199999809265137, 1.017199993133545, 1.0160000324249268, 1.0157999992370605, 1.0190000534057617, 0.9846000075340271, 0.9943000078201294, 0.977400004863739, 1.0053999423980713, 1.0068999528884888, 0.95660001039505, 0.9110999703407288, 0.9811999797821045, 1.0018999576568604, 0.9043999910354614, 0.9046000242233276, 1.0059000253677368, 0.8751000165939331, 0.9049999713897705, 0.9211999773979187, 0.9178000092506409, 0.8187000155448914, 0.9053000211715698, 0.9251999855041504, 0.7911999821662903, 0.8852999806404114, 0.8816999793052673, 0.911899983882904, 0.9535999894142151, 0.819100022315979, 0.8690000176429749, 0.9232000112533569, 0.6187000274658203, 0.8593999743461609, 0.8901000022888184, 0.821399986743927, 0.8363999724388123, 0.7642999887466431, 0.7555000185966492, 0.7657999992370605, 0.6110000014305115, 0.5055000185966492, 0.4812000095844269, 0.7027999758720398, 0.6018999814987183, 0.4122999906539917, 0.6420999765396118, 0.45010000467300415, 0.6421999931335449, 0.589900016784668, 0.2240000069141388, 0.29330000281333923, 0.5317999720573425, 0.4634999930858612, 0.4242999851703644, 0.35910001397132874, 0.2822999954223633, 0.618399977684021, 0.11509999632835388, -0.04129999876022339, -0.2540999948978424, 0.21549999713897705, 0.17569999396800995, 0.0333000011742115, 0.10779999941587448, 0.31859999895095825, 0.2079000025987625, -0.09260000288486481, 0.05490000173449516, 0.03370000049471855, -0.1168999969959259, 1.083899974822998, 1.0836999416351318, 1.0810999870300293, 1.080199956893921, 1.0774999856948853, 1.073099970817566, 1.0724999904632568, 1.0715999603271484, 1.0708999633789062, 1.0707000494003296, 1.0700000524520874, 1.0694999694824219, 1.069200038909912, 1.0678000450134277, 1.0671000480651855, 1.0664000511169434, 1.065600037574768, 1.0636999607086182, 1.0614999532699585, 1.0607999563217163, 1.0600999593734741, 1.058899998664856, 1.0578999519348145, 1.0569000244140625, 1.0568000078201294, 1.05649995803833, 1.055799961090088, 1.0556000471115112, 1.055400013923645, 1.0550999641418457, 1.0462000370025635, 1.0528000593185425, 1.044800043106079, 1.045300006866455, 1.0535000562667847, 0.9958999752998352, 1.0259000062942505, 1.0384000539779663, 1.0384999513626099, 0.965499997138977, 0.9341999888420105, 1.0092999935150146, 1.042099952697754, 0.9750999808311462, 0.9146000146865845, 0.9081000089645386, 0.8700000047683716, 0.8970999717712402, 0.9139000177383423, 0.8274999856948853, 0.930400013923645, 0.8551999926567078, 0.9531000256538391, 0.8889999985694885, 0.8048999905586243, 0.7882999777793884, 0.7768999934196472, 0.8830000162124634, 0.909600019454956, 0.8517000079154968, 0.7646999955177307, 0.8650000095367432, 0.7350000143051147, 0.8647000193595886, 0.8252999782562256, 0.79339998960495, 0.6234999895095825, 0.7342000007629395, 0.714900016784668, 0.5426999926567078, 0.6019999980926514, 0.57669997215271, 0.6830000281333923, 0.6082000136375427, 0.4674000144004822, 0.6281999945640564, 0.026900000870227814, 0.45890000462532043, 0.5878000259399414, 0.40389999747276306, 0.383899986743927, 0.19429999589920044, -0.032999999821186066, 0.019700000062584877, 0.02979999966919422, 0.26759999990463257, -0.4657000005245209, -0.5802000164985657, -0.909500002861023, -0.02160000056028366, 1.1410000324249268, 1.1406999826431274, 1.1397000551223755, 1.139299988746643, 1.1390999555587769, 1.1380000114440918, 1.1368000507354736, 1.1367000341415405, 1.135599970817566, 1.1346999406814575, 1.134600043296814, 1.134600043296814, 1.1340999603271484, 1.1339000463485718, 1.1337000131607056, 1.1332999467849731, 1.1327999830245972, 1.13100004196167, 1.1303999423980713, 1.1299999952316284, 1.1289000511169434, 1.1287000179290771, 1.1283999681472778, 1.128000020980835, 1.127500057220459, 1.1272000074386597, 1.1269999742507935, 1.1265000104904175, 1.1263999938964844, 1.126099944114685, 1.1059999465942383, 1.124400019645691, 1.118399977684021, 1.1125999689102173, 1.121000051498413, 1.0958000421524048, 1.1068999767303467, 1.0987999439239502, 1.1190999746322632, 1.0526000261306763, 1.0621000528335571, 1.0444999933242798, 1.0621000528335571, 1.0199999809265137, 1.0388000011444092, 1.0555000305175781, 1.0793999433517456, 1.086899995803833, 0.9506000280380249, 1.0515999794006348, 0.9855999946594238, 0.9977999925613403, 0.9934999942779541, 0.9980999827384949, 0.9740999937057495, 0.8682000041007996, 0.8447999954223633, 1.0083999633789062, 0.7437000274658203, 0.7850000262260437, 0.7396000027656555, 0.8496000170707703, 0.828499972820282, 0.9312999844551086, 0.8341000080108643, 0.8234999775886536, 0.48510000109672546, 0.579200029373169, 0.524399995803833, 0.483599990606308, 0.36010000109672546, 0.19699999690055847, 0.38830000162124634, 0.6675999760627747, 0.4708999991416931, 0.16930000483989716, 0.7505999803543091, 0.243599995970726, 0.7231000065803528, 0.4219000041484833, 0.07349999994039536, -0.37940001487731934, 0.260699987411499, -0.045499999076128006, 0.0754999965429306, -0.21119999885559082, -0.29120001196861267, -0.06019999831914902, 0.3278000056743622, 0.11460000276565552, -0.34279999136924744, -0.24469999969005585], "logprob": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, -8.805399894714355, -8.92300033569336, -9.178600311279297, -9.005599975585938, -9.145500183105469, -8.65060043334961, -8.69540023803711, -9.452500343322754, -9.382100105285645, -9.553400039672852, -9.29319953918457, -9.192500114440918, -9.15310001373291, -8.57409954071045, -9.554800033569336, -9.103500366210938, -9.555100440979004, -9.693300247192383, -9.454299926757812, -9.032600402832031, -9.694999694824219, -8.833900451660156, -9.028200149536133, -9.863800048828125, -9.842900276184082, -9.651900291442871, -9.397299766540527, -9.851499557495117, -8.900899887084961, -9.670999526977539, -8.159700393676758, -8.70199966430664, -8.781100273132324, -9.361200332641602, -6.502099990844727, -7.3024001121521, -6.639999866485596, -8.196999549865723, -8.572099685668945, -6.21560001373291, -4.818299770355225, -7.456500053405762, -8.543299674987793, -5.461699962615967, -5.786799907684326, -8.740300178527832, -5.829500198364258, -6.420499801635742, -6.832200050354004, -6.7743000984191895, -4.941100120544434, -6.5991997718811035, -7.08620023727417, -4.929200172424316, -6.448599815368652, -6.483500003814697, -6.974800109863281, -7.666900157928467, -5.852499961853027, -6.570899963378906, -7.314899921417236, -3.2929000854492188, -6.591899871826172, -6.945000171661377, -6.165999889373779, -6.508999824523926, -5.857800006866455, -5.929100036621094, -6.0370001792907715, -4.843999862670898, -4.134500026702881, -4.256100177764893, -5.652699947357178, -5.082600116729736, -4.422999858856201, -5.549900054931641, -4.641200065612793, -5.572500228881836, -5.415500164031982, -4.05709981918335, -4.355500221252441, -5.2221999168396, -5.083399772644043, -4.980000019073486, -4.825399875640869, -4.676300048828125, -5.623799800872803, -4.475500106811523, -4.365600109100342, -4.076200008392334, -4.933899879455566, -4.985899925231934, -4.84250020980835, -4.9994001388549805, -5.228600025177002, -5.123499870300293, -4.963699817657471, -5.231400012969971, -5.2621002197265625, -5.235899925231934, -9.289600372314453, -8.349900245666504, -8.812600135803223, -9.372400283813477, -9.583700180053711, -7.839900016784668, -8.71030044555664, -9.397100448608398, -6.952600002288818, -9.349699974060059, -9.769800186157227, -9.401700019836426, -9.704299926757812, -9.75, -9.170299530029297, -9.833200454711914, -9.810799598693848, -9.863800048828125, -10.031100273132324, -10.076700210571289, -10.012200355529785, -9.835599899291992, -10.157600402832031, -9.59939956665039, -9.11050033569336, -10.223299980163574, -9.963299751281738, -9.085399627685547, -6.362800121307373, -10.235899925231934, -6.564199924468994, -7.436299800872803, -7.494500160217285, -7.889200210571289, -9.32509994506836, -5.722799777984619, -7.59089994430542, -8.226400375366211, -8.464400291442871, -5.010499954223633, -4.581099987030029, -7.762599945068359, -8.892999649047852, -6.7332000732421875, -4.799600124359131, -4.65339994430542, -4.01230001449585, -4.84660005569458, -5.501299858093262, -3.9937000274658203, -6.145299911499023, -4.84499979019165, -6.744999885559082, -5.6722002029418945, -4.156899929046631, -4.014500141143799, -4.208799839019775, -5.980599880218506, -6.470600128173828, -5.7195000648498535, -4.759500026702881, -5.937399864196777, -4.516900062561035, -5.971499919891357, -5.754499912261963, -5.52269983291626, -4.499100208282471, -5.393799781799316, -5.347799777984619, -4.469200134277344, -4.916600227355957, -4.9039998054504395, -5.3566999435424805, -5.155600070953369, -4.789000034332275, -5.289299964904785, -3.7952001094818115, -4.872499942779541, -5.189300060272217, -4.816999912261963, -4.902400016784668, -4.955100059509277, -4.796599864959717, -4.85129976272583, -4.928899765014648, -5.087900161743164, -4.789999961853027, -4.861400127410889, -4.821000099182129, -5.072299957275391, -7.281799793243408, -6.204999923706055, -7.272200107574463, -6.14139986038208, -7.964399814605713, -7.174300193786621, -8.585000038146973, -6.716100215911865, -8.05739974975586, -8.64900016784668, -7.5640997886657715, -8.645700454711914, -8.512200355529785, -7.607500076293945, -8.891300201416016, -8.803099632263184, -8.92389965057373, -9.10789966583252, -8.316699981689453, -8.17959976196289, -8.202799797058105, -9.208200454711914, -9.112500190734863, -7.317500114440918, -8.963500022888184, -8.854399681091309, -8.113900184631348, -9.19260025024414, -8.414299964904785, -8.662500381469727, -5.756199836730957, -8.17959976196289, -7.657400131225586, -7.249100208282471, -8.217599868774414, -6.436100006103516, -7.013400077819824, -6.946499824523926, -8.115300178527832, -5.662600040435791, -6.2058000564575195, -5.921299934387207, -6.382500171661377, -5.379499912261963, -5.904300212860107, -6.746200084686279, -7.228099822998047, -7.38640022277832, -5.182000160217285, -6.827899932861328, -5.8572998046875, -6.101900100708008, -6.071300029754639, -6.141499996185303, -5.928999900817871, -5.0106000900268555, -4.862599849700928, -6.529600143432617, -4.496300220489502, -4.838900089263916, -4.572800159454346, -5.58519983291626, -5.469699859619141, -6.086999893188477, -5.57919979095459, -5.616700172424316, -4.105599880218506, -4.716700077056885, -4.526199817657471, -4.392099857330322, -3.9642999172210693, -3.6250998973846436, -4.37529993057251, -5.192800045013428, -4.648099899291992, -4.111800193786621, -5.488999843597412, -4.405200004577637, -5.461100101470947, -5.092100143432617, -4.797599792480469, -4.290999889373779, -5.095799922943115, -4.880799770355225, -5.08620023727417, -4.948500156402588, -4.93120002746582, -5.167399883270264, -5.350399971008301, -5.366099834442139, -5.304599761962891, -5.367400169372559]}, "token.table": {"Topic": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 3, 1, 1, 2, 3, 1, 2, 3, 2, 3, 1, 2, 3, 1, 2, 3, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 3, 1, 2, 3, 1, 2, 3, 1, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 3, 2, 3, 1, 2, 3, 2, 3, 1, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 1, 2, 3, 1, 2, 3, 1, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 2, 3, 3, 1, 2, 3, 1, 2, 3, 2, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 3, 1, 3, 1, 2, 3, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 1, 2, 3, 1, 2, 3, 1, 3, 1, 3, 1, 2, 3, 1, 2, 1, 2, 3, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 3, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 3, 1, 2, 3, 1, 2, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 3, 2, 3, 1, 2, 3, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 3, 1, 2, 3, 1, 2, 3, 2, 2, 2, 1, 2, 3, 2, 3, 1, 2, 3, 1, 2, 3, 1, 3, 1, 2, 3, 1, 2, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 3, 1, 2, 3, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3], "Freq": [0.4631671607494354, 0.3425685465335846, 0.19418592751026154, 0.9672647714614868, 0.016394319012761116, 0.016394319012761116, 0.11197340488433838, 0.7895397543907166, 0.09797672927379608, 0.9543216824531555, 0.03624006360769272, 0.9686636924743652, 0.5001664757728577, 0.33411726355552673, 0.16579686105251312, 0.27087831497192383, 0.34162649512290955, 0.38749974966049194, 0.9674239158630371, 0.01974334567785263, 0.009837213903665543, 0.009837213903665543, 0.9837213754653931, 0.3889104723930359, 0.31136196851730347, 0.29968297481536865, 0.9651471972465515, 0.32196593284606934, 0.6231852769851685, 0.054623913019895554, 0.8433762788772583, 0.14521710574626923, 0.012287601828575134, 0.43318212032318115, 0.4038241505622864, 0.1628696471452713, 0.07472704350948334, 0.7795848250389099, 0.14591969549655914, 0.15389278531074524, 0.7607778906822205, 0.08528152108192444, 0.993636429309845, 0.9704499244689941, 0.018662499263882637, 0.018662499263882637, 0.030017193406820297, 0.9605501890182495, 0.030017193406820297, 0.5649868845939636, 0.177329882979393, 0.25760433077812195, 0.043577518314123154, 0.9551721215248108, 0.001177770784124732, 0.21174772083759308, 0.37524154782295227, 0.4130054712295532, 0.39183878898620605, 0.09127112478017807, 0.5169011354446411, 0.3350903391838074, 0.2087308019399643, 0.4561648964881897, 0.9726002216339111, 0.03740769997239113, 0.9905974864959717, 0.024943912401795387, 0.9728125333786011, 0.024943912401795387, 0.09800685942173004, 0.7144044041633606, 0.1875067949295044, 0.008836944587528706, 0.9809008240699768, 0.008836944587528706, 0.04547499120235443, 0.9549748301506042, 0.022737495601177216, 0.03681238740682602, 0.9571220874786377, 0.8640524744987488, 0.06291020661592484, 0.07446514815092087, 0.9607241153717041, 0.02044093795120716, 0.02044093795120716, 0.9285061359405518, 0.05892995744943619, 0.012935844250023365, 0.008629602380096912, 0.9708302617073059, 0.021574005484580994, 0.9625065922737122, 0.11336120218038559, 0.15793713927268982, 0.7285858988761902, 0.05955730751156807, 0.08106411248445511, 0.8594450354576111, 0.9770052433013916, 0.030531413853168488, 0.136963352560997, 0.3254672884941101, 0.537589430809021, 0.015286813490092754, 0.015286813490092754, 0.9783560633659363, 0.11945109069347382, 0.8734686970710754, 0.00723946001380682, 0.05857022851705551, 0.9293142557144165, 0.015618727542459965, 0.9903458952903748, 0.023706817999482155, 0.971979558467865, 0.023706817999482155, 0.05107080563902855, 0.5919801592826843, 0.35681694746017456, 0.9907915592193604, 0.016242483630776405, 0.9657973051071167, 0.02541571855545044, 0.01270785927772522, 0.8628994822502136, 0.11351776123046875, 0.0238551814109087, 0.24586959183216095, 0.4980200529098511, 0.25607597827911377, 0.6434440612792969, 0.2789745628833771, 0.07755725085735321, 0.20851437747478485, 0.6935633420944214, 0.0979815125465393, 0.4801929295063019, 0.16901050508022308, 0.35071492195129395, 0.7921364307403564, 0.13006891310214996, 0.07759476453065872, 0.9801175594329834, 0.025131219998002052, 0.018152549862861633, 0.015127125196158886, 0.9681360125541687, 0.824207603931427, 0.03165346756577492, 0.14365804195404053, 0.022927140817046165, 0.04012249782681465, 0.9371469020843506, 0.973415732383728, 0.02781187742948532, 0.9570453763008118, 0.038281816989183426, 0.013159378431737423, 0.0032898446079343557, 0.9836634993553162, 0.017053917050361633, 0.9891272187232971, 0.0031812735833227634, 0.995738685131073, 0.0953838899731636, 0.04100615903735161, 0.8638036251068115, 0.01448257826268673, 0.007241289131343365, 0.9775740504264832, 0.09692256152629852, 0.0318649522960186, 0.8723030686378479, 0.041479259729385376, 0.044935863465070724, 0.9142720103263855, 0.963290274143219, 0.031073879450559616, 0.26984521746635437, 0.10987263917922974, 0.6202967762947083, 0.014183695428073406, 0.023639492690563202, 0.961789608001709, 0.062464311718940735, 0.8155676126480103, 0.1218385174870491, 0.973655641078949, 0.043140947818756104, 0.9491008520126343, 0.043140947818756104, 0.025062354281544685, 0.9774317741394043, 0.012531177140772343, 0.970384418964386, 0.023667912930250168, 0.9639585614204407, 0.02734634093940258, 0.006836585234850645, 0.030807074159383774, 0.5985700488090515, 0.3705976903438568, 0.9668530821800232, 0.021485624834895134, 0.021485624834895134, 0.01607389748096466, 0.00803694874048233, 0.972470760345459, 0.97510826587677, 0.007980980910360813, 0.007980980910360813, 0.9816606640815735, 0.035468194633722305, 0.044588588178157806, 0.9201463460922241, 0.10452626645565033, 0.8428997993469238, 0.05184502527117729, 0.9703685641288757, 0.02488124556839466, 0.8693336248397827, 0.07150363177061081, 0.06021358445286751, 0.9054116606712341, 0.09014920890331268, 0.003919531125575304, 0.01901468075811863, 0.01901468075811863, 0.9626182317733765, 0.08811426907777786, 0.05962619557976723, 0.8526545763015747, 0.8796747922897339, 0.06149702146649361, 0.05882323533296585, 0.3106619417667389, 0.1797325313091278, 0.5096557140350342, 0.003216092474758625, 0.9969886541366577, 0.9641833305358887, 0.0143907954916358, 0.0287815909832716, 0.5944792032241821, 0.22387738525867462, 0.1817331314086914, 0.3609960675239563, 0.12286467850208282, 0.5161057710647583, 0.1901327222585678, 0.11219984292984009, 0.6976754665374756, 0.2070820927619934, 0.1371404528617859, 0.6558741927146912, 0.05683133378624916, 0.9377170205116272, 0.009471888653934002, 0.012604267336428165, 0.012604267336428165, 0.9831328392028809, 0.0028798384591937065, 0.005759676918387413, 0.9935442805290222, 0.25595909357070923, 0.5307626128196716, 0.21338962018489838, 0.16978375613689423, 0.7896919250488281, 0.04014267399907112, 0.7704375982284546, 0.15266437828540802, 0.07676344364881516, 0.9759221076965332, 0.0644509419798851, 0.03661985322833061, 0.8993836641311646, 0.9887718558311462, 0.10316041111946106, 0.45522889494895935, 0.44158443808555603, 0.11237695813179016, 0.14685624837875366, 0.7406663298606873, 0.9555126428604126, 0.5477522015571594, 0.24209152162075043, 0.21013452112674713, 0.7510778903961182, 0.18840597569942474, 0.06046813353896141, 0.9828963279724121, 0.011842125095427036, 0.6376189589500427, 0.230915829539299, 0.13147640228271484, 0.006947567220777273, 0.993502140045166, 0.8505893349647522, 0.10543019324541092, 0.04288685694336891, 0.9585669636726379, 0.016915181651711464, 0.04059643670916557, 0.9438672065734863, 0.9088995456695557, 0.06446096301078796, 0.02670525573194027, 0.4369237720966339, 0.1861501783132553, 0.3769017159938812, 0.805979311466217, 0.17192764580249786, 0.02299669198691845, 0.2314000427722931, 0.09909523278474808, 0.669393002986908, 0.9752904772758484, 0.01773255318403244, 0.8329060077667236, 0.12999111413955688, 0.03731226176023483, 0.8465367555618286, 0.05087880790233612, 0.10284014046192169, 0.46823298931121826, 0.12575316429138184, 0.4059840738773346, 0.12611360847949982, 0.6109753847122192, 0.2628117501735687, 0.9345153570175171, 0.05675194412469864, 0.007566926069557667, 0.9889297485351562, 0.9720643758773804, 0.013500894419848919, 0.013500894419848919, 0.0642155185341835, 0.05358380824327469, 0.8824318051338196, 0.963050901889801, 0.02348904497921467, 0.011744522489607334, 0.8803501129150391, 0.07673753798007965, 0.04263196885585785, 0.9793519973754883, 0.020992174744606018, 0.06705833226442337, 0.9119933247566223, 0.6638789176940918, 0.18735666573047638, 0.14882482588291168, 0.05138445273041725, 0.9463303685188293, 0.002854691818356514, 0.17211271822452545, 0.6584795117378235, 0.16944201290607452, 0.001818951335735619, 0.005456853657960892, 0.9913284778594971, 0.9785966277122498, 0.11391127109527588, 0.02312484383583069, 0.8633275032043457, 0.8377506732940674, 0.13962511718273163, 0.022501567378640175, 0.9868656992912292, 0.023496802896261215, 0.9876649379730225, 0.013346823863685131, 0.037552569061517715, 0.9441789388656616, 0.016093958169221878, 0.03808882459998131, 0.9522205591201782, 0.13595256209373474, 0.8464540243148804, 0.017564931884407997, 0.9857562184333801, 0.10907604545354843, 0.8299636840820312, 0.060859717428684235, 0.08463143557310104, 0.15713657438755035, 0.7581460475921631, 0.9303868412971497, 0.016322577372193336, 0.05223224312067032, 0.0035107205621898174, 0.9004998207092285, 0.09595969319343567, 0.5550825595855713, 0.29222506284713745, 0.15263864398002625, 0.2663761377334595, 0.5722468495368958, 0.1613757461309433, 0.209045872092247, 0.32176831364631653, 0.4691936671733856, 0.9722961783409119, 0.020909596234560013, 0.010454798117280006, 0.9448142647743225, 0.03936726227402687, 0.006353156175464392, 0.991092324256897, 0.3183504343032837, 0.33919262886047363, 0.3425126075744629, 0.022432472556829453, 0.025071587413549423, 0.9514007568359375, 0.18554045259952545, 0.007105804514139891, 0.8076931238174438, 0.008845777250826359, 0.004422888625413179, 0.9907270073890686, 0.00847669132053852, 0.9917728900909424, 0.030394937843084335, 0.7437166571617126, 0.22589194774627686, 0.026255374774336815, 0.9714488983154297, 0.005251075141131878, 0.990767240524292, 0.014930320903658867, 0.9406101703643799, 0.04479096084833145, 0.6299434900283813, 0.23483100533485413, 0.13533605635166168, 0.971320390701294, 0.019426407292485237, 0.019426407292485237, 0.04377695173025131, 0.9485006332397461, 0.014592316932976246, 0.4299458861351013, 0.5262525677680969, 0.04384220391511917, 0.9813764095306396, 0.021425435319542885, 0.9855700731277466, 0.8685970902442932, 0.10444052517414093, 0.02698046900331974, 0.9579147100448608, 0.7436357140541077, 0.16060268878936768, 0.09613541513681412, 0.5789375305175781, 0.18321920931339264, 0.23782628774642944, 0.022996926680207253, 0.8042160868644714, 0.1731533408164978, 0.7501397132873535, 0.07915779948234558, 0.17106585204601288, 0.19504037499427795, 0.7592824697494507, 0.045831941068172455, 0.9746495485305786, 0.2520837187767029, 0.0811600536108017, 0.666774332523346, 0.3584243655204773, 0.43463072180747986, 0.20690996944904327, 0.8776844143867493, 0.08892112225294113, 0.03458043560385704, 0.9875293970108032, 0.9441722631454468, 0.05103633925318718, 0.0028353522066026926, 0.9725253582000732, 0.016483480110764503, 0.016483480110764503, 0.16397246718406677, 0.808864176273346, 0.026995467022061348, 0.9518610835075378, 0.004577845800668001, 0.018311383202672005, 0.9750811457633972, 0.01400049775838852, 0.9800348281860352, 0.01400049775838852, 0.054203879088163376, 0.9467610716819763, 0.6483304500579834, 0.133933886885643, 0.21773530542850494, 0.072666697204113, 0.023223377764225006, 0.9042134284973145, 0.014576997607946396, 0.007288498803973198, 0.9839473366737366, 0.8630863428115845, 0.01717584766447544, 0.11915744096040726, 0.8626156449317932, 0.11672356724739075, 0.01992841437458992, 0.9524548649787903, 0.035276107490062714, 0.11219333112239838, 0.863265335559845, 0.024931849911808968, 0.07941596955060959, 0.6930258870124817, 0.227875217795372, 0.9712705612182617, 0.9450896978378296, 0.9727724194526672, 0.36119645833969116, 0.0709417462348938, 0.5678160786628723, 0.010931913740932941, 0.9838722944259644, 0.013228676281869411, 0.03086691163480282, 0.9546694755554199, 0.6481882929801941, 0.16967709362506866, 0.18203045427799225, 0.9873111248016357, 0.01862851157784462, 0.008957219310104847, 0.008957219310104847, 0.9852941632270813, 0.8750616312026978, 0.020134160295128822, 0.10531715303659439, 0.9899226427078247, 0.13058175146579742, 0.046240922063589096, 0.8232186436653137, 0.5274016261100769, 0.168533593416214, 0.3040545582771301, 0.2515401244163513, 0.26329270005226135, 0.4851882755756378, 0.06943274289369583, 0.6070959568023682, 0.32337331771850586, 0.0020529974717646837, 0.0030794960912317038, 0.9946772456169128, 0.16250786185264587, 0.782108724117279, 0.05536678805947304, 0.5338404774665833, 0.24852821230888367, 0.21771763265132904, 0.002190925879403949, 0.002190925879403949, 0.9957758188247681, 0.1235436275601387, 0.8293837904930115, 0.047084808349609375, 0.13046319782733917, 0.6204118132591248, 0.2491847276687622, 0.018680281937122345, 0.07098507136106491, 0.911597728729248, 0.9565136432647705, 0.02277413383126259, 0.13144612312316895, 0.7232121825218201, 0.14530432224273682, 0.16100941598415375, 0.8254566788673401, 0.012472559697926044, 0.29861369729042053, 0.02754203975200653, 0.6740551590919495, 0.08663367480039597, 0.911933422088623, 0.004559667315334082, 0.03585561364889145, 0.9501737952232361, 0.017927806824445724, 0.09286879748106003, 0.8814665675163269, 0.025184758007526398, 0.9841384291648865, 0.025234319269657135, 0.09639467298984528, 0.06051052361726761, 0.8429256677627563, 0.9823130965232849, 0.9537951350212097, 0.042079195380210876, 0.007013199385255575, 0.015527160838246346, 0.007763580419123173, 0.9782111048698425, 0.007807107642292976, 0.007807107642292976, 0.9836955666542053, 0.3689178228378296, 0.4882119596004486, 0.14290152490139008, 0.9615058302879333, 0.13832414150238037, 0.1176195740699768, 0.7440429329872131, 0.18606573343276978, 0.7937871813774109, 0.020145462825894356, 0.19561007618904114, 0.7351027131080627, 0.06924597173929214, 0.9747580885887146, 0.011334395967423916, 0.011334395967423916, 0.4163217544555664, 0.24059352278709412, 0.3431941270828247, 0.9485225677490234, 0.039521776139736176, 0.009880444034934044, 0.1913449466228485, 0.08371341228485107, 0.7250733375549316, 0.9885959029197693, 0.7919647693634033, 0.14735792577266693, 0.06078234314918518, 0.1317857950925827, 0.7315403819084167, 0.13669545948505402, 0.08571907877922058, 0.8246322274208069, 0.08976589888334274, 0.13494303822517395, 0.13235627114772797, 0.7329174280166626, 0.6638029217720032, 0.20836707949638367, 0.12785376608371735, 0.12782466411590576, 0.6798086166381836, 0.19249606132507324, 0.03385944664478302, 0.04474284127354622, 0.9202514290809631, 0.7052405476570129, 0.15219847857952118, 0.14283867180347443, 0.7938053607940674, 0.14773812890052795, 0.0581766702234745, 0.030929379165172577, 0.9519375562667847, 0.017182989045977592], "Term": ["access", "access", "access", "accordingly", "accordingly", "accordingly", "across", "across", "across", "advised", "advised", "alcohol", "also", "also", "also", "apartment", "apartment", "apartment", "aquarium", "aquarium", "architectural", "architectural", "architectural", "area", "area", "area", "arent", "around", "around", "around", "ask", "ask", "ask", "available", "available", "available", "ave", "ave", "ave", "away", "away", "away", "backed", "background", "background", "background", "bars!", "bars!", "bars!", "bathroom", "bathroom", "bathroom", "beach", "beach", "beach", "beautiful", "beautiful", "beautiful", "bed", "bed", "bed", "bedroom", "bedroom", "bedroom", "bier", "bier", "blazing", "blink", "blink", "blink", "block", "block", "block", "boardwalk", "boardwalk", "boardwalk", "boat", "boat", "boat", "bodega", "bodega", "book", "book", "book", "booked", "booked", "booked", "booking", "booking", "booking", "botanical", "botanical", "botanical", "bowling", "brand", "brand", "brand", "brick", "brick", "brick", "brooklyn!!", "brooklyn!!", "building", "building", "building", "burner", "burner", "burner", "bus", "bus", "bus", "cab", "cab", "cab", "cabinetry", "caught", "caught", "caught", "central", "central", "central", "cereal", "cereal", "certain", "certain", "certain", "check", "check", "check", "city", "city", "city", "clean", "clean", "clean", "close", "close", "close", "comfortable", "comfortable", "comfortable", "common", "common", "common", "concern", "concern", "concierge", "concierge", "concierge", "conditioner", "conditioner", "conditioner", "contemporary", "contemporary", "contemporary", "conversation", "conversation", "cooper", "cooper", "counter", "counter", "counter", "craving", "craving", "custom", "custom", "deck", "deck", "deck", "decorative", "decorative", "decorative", "designed", "designed", "designed", "designer", "designer", "designer", "dinette", "dinette", "dining", "dining", "dining", "dishwasher", "dishwasher", "dishwasher", "distance", "distance", "distance", "distant", "diversified", "diversified", "diversified", "domino", "domino", "domino", "dorm", "dorm", "driveway", "driveway", "driveway", "east", "east", "east", "encourage", "encourage", "encourage", "entertaining", "entertaining", "entertaining", "eve", "expansive", "expansive", "expansive", "exposed", "exposed", "exposed", "express", "express", "express", "expressway", "expressway", "fee", "fee", "fee", "female", "female", "female", "fireplace", "fireplace", "fireplace", "flat", "flat", "flat", "flexible", "flexible", "flexible", "floor", "floor", "floor", "flooring", "flooring", "follow", "follow", "follow", "free", "free", "free", "full", "full", "full", "fully", "fully", "fully", "furnished", "furnished", "furnished", "gateway", "gateway", "gateway", "glassware", "glassware", "glassware", "granite", "granite", "granite", "great", "great", "great", "grocery", "grocery", "grocery", "guest", "guest", "guest", "guesthouse", "hardwood", "hardwood", "hardwood", "hay", "heart", "heart", "heart", "high", "high", "high", "highland", "home", "home", "home", "host", "host", "host", "hostel", "hostel", "house", "house", "house", "housekeeping", "housekeeping", "however", "however", "however", "hygiene", "industrial", "industrial", "industrial", "keep", "keep", "keep", "kitchen", "kitchen", "kitchen", "know", "know", "know", "large", "large", "large", "leasing", "leasing", "let", "let", "let", "listing", "listing", "listing", "living", "living", "living", "location", "location", "location", "lock", "lock", "lock", "lockable", "locker", "locker", "locker", "loft", "loft", "loft", "loving", "loving", "loving", "luggage", "luggage", "luggage", "luna", "luxury", "luxury", "luxury", "make", "make", "make", "mall", "mall", "mall", "many", "many", "many", "marble", "marble", "marble", "marina", "master", "master", "master", "may", "may", "may", "meant", "meant", "men", "men", "met", "met", "met", "midtown!", "midtown!", "min", "min", "min", "mingle", "minute", "minute", "minute", "modern", "modern", "modern", "move", "move", "move", "museum", "museum", "museum", "need", "need", "need", "neighborhood", "neighborhood", "neighborhood", "new", "new", "new", "normally", "normally", "normally", "numb", "numb", "oak", "oak", "one", "one", "one", "original", "original", "original", "oven", "oven", "oven", "oversized", "oversized", "oversized", "para", "para", "park", "park", "park", "parkway", "parkway", "parkway", "parquet", "path", "path", "path", "people", "people", "people", "permanent", "permanent", "permanent", "pizzeria", "pizzeria", "pizzeria", "place", "place", "place", "plank", "playroom", "playroom", "please", "please", "please", "polite", "privacy", "privacy", "privacy", "private", "private", "private", "prospect", "prospect", "prospect", "provide", "provide", "provide", "public", "public", "public", "puppy", "queen", "queen", "queen", "quiet", "quiet", "quiet", "read", "read", "read", "reception", "reservation", "reservation", "reservation", "review", "review", "review", "ride", "ride", "ride", "ride!", "rise", "rise", "rise", "riverbank", "riverbank", "riverbank", "rockaway", "rockaway", "room", "room", "room", "screen", "screen", "screen", "sectional", "sectional", "sectional", "shampoo", "shampoo", "shampoo", "share", "share", "share", "sheepshead", "sheepshead", "shop", "shop", "shop", "shopping", "shopping", "shopping", "shops!", "shore", "shrine", "size", "size", "size", "sleek", "sleek", "sleeper", "sleeper", "sleeper", "small", "small", "small", "snack", "snack", "soaking", "soaking", "soaking", "soap", "soap", "soap", "soaring", "sofa", "sofa", "sofa", "space", "space", "space", "spacious", "spacious", "spacious", "square", "square", "square", "stainless", "stainless", "stainless", "station", "station", "station", "stay", "stay", "stay", "steel", "steel", "steel", "stop", "stop", "stop", "street", "street", "street", "stunning", "stunning", "stunning", "subject", "subject", "subway", "subway", "subway", "supermarket", "supermarket", "supermarket", "table", "table", "table", "target", "target", "target", "tattoo", "tattoo", "tattoo", "taxi", "taxi", "taxi", "thoroughly", "thoroughly", "throughout", "throughout", "throughout", "ticket", "tidy", "tidy", "tidy", "tile", "tile", "tile", "tiled", "tiled", "tiled", "time", "time", "time", "took", "top", "top", "top", "train", "train", "train", "transportation", "transportation", "transportation", "twice", "twice", "twice", "two", "two", "two", "understand", "understand", "understand", "unit", "unit", "unit", "unparalleled", "use", "use", "use", "walk", "walk", "walk", "walking", "walking", "walking", "washer", "washer", "washer", "welcome", "welcome", "welcome", "within", "within", "within", "wood", "wood", "wood", "work", "work", "work", "would", "would", "would", "zoo", "zoo", "zoo"]}, "R": 30, "lambda.step": 0.01, "plot.opts": {"xlab": "PC1", "ylab": "PC2"}, "topic.order": [2, 1, 3]};

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
       new LDAvis("#" + "ldavis_el253286103006928447223460", ldavis_el253286103006928447223460_data);
   }(LDAvis);
}else if(typeof define === "function" && define.amd){
   // require.js is available: use it to load d3/LDAvis
   require.config({paths: {d3: "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min"}});
   require(["d3"], function(d3){
      window.d3 = d3;
      LDAvis_load_lib("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js", function(){
        new LDAvis("#" + "ldavis_el253286103006928447223460", ldavis_el253286103006928447223460_data);
      });
    });
}else{
    // require.js not available: dynamically load d3 & LDAvis
    LDAvis_load_lib("https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js", function(){
         LDAvis_load_lib("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js", function(){
                 new LDAvis("#" + "ldavis_el253286103006928447223460", ldavis_el253286103006928447223460_data);
            })
         });
}
</script>



pyLDAvis package is a great package to visualize the LDA model. The area of the circles means the prevalence of each topic. Here I chose the cluster the corpus into three topics. The red bar represents the estimated term frequency within selected topic and the blue bar represents the overall term frequency. In topic 1, the prevalent term is about layout of the room, for example, there are words "kitchen", "bathroom", "bedroom". Topic 2 is about the living environment because it has words "new","private","space","large". Topic 3 is correlated with location or transit with words "subway", "walk","away". There are some overlaps among these three topics, which can be improved to better serve the machine learninng model. At this moment, I will go ahead with the current model.


## IV. Machine learning model

In this part, 3 regression algorithms: __linear regression, decision tree__ and __random forest__ will be trained to predict the yield. Linear regression is the simplest algorithm and will be used as the baseline model. Decision tree model can capture the nonlinear relationships in the dataset while random forest is a more complex model and able to provide higher accuracy. 

To measure the accuracy of the model, MSE (mean squared error) is used as evaluation metrics. The target for prediction is "yield". "price" and "reviews_per_month", "average_length_of_stay" have strong correlation with "yield" because they are used for yield calculation. Catergorical features also need to be converted to numerical features so that they can be fed into machine learning algorithms. To split the whole dataset into a training set and a testing set, the dataset will be randomly shuffled first and 25% will be used as the splitting ratio.

Once the 3 algorithms have been applied and trained, they will be compared based on MSE value. The smaller MSE, the better accuracy. The best algorithm will be chosen and the model will be further fine-tuned using GridSearchCV function in scikit-learn. The to-do list in this part is:

1. Clean-up the dataset: separate the "yield" from the dataset and save it as the target, drop "price", "average_length_of_stay" and "reviews_per_month", convert catergorical variables into numerical features. Other columns including "level_0", "id", "listing_url", "description","image_link" can be dropped as well since they are not needed any more.
2. Randomly shuffle the dataset to remove inherent order and split the dataset into a training set and a test set using 75:25 ratio.
3. Use linear regression, decision tree, and random forest separately to train the model and calculate the MSE value.
4. Select the model with lowest MSE value for further refinement.



```python
# drop unnecessary columns
cols_to_drop = ['price','average_length_of_stay','minimum_nights','cleaning_fee','index', 'id','listing_url','description','image_link']
final_df = final_df.drop(cols_to_drop, axis = 1)

# convert strings to numerical features
categorical_feats = ['property_type', 'room_type', 'bed_type', 'cancellation_policy','description_topic' ]
final_df = pd.get_dummies(final_df, columns = categorical_feats, drop_first = False)

# separate the target variable "yield" from the dataset
target = final_df['yield']
final_df = final_df.drop(['yield'], axis = 1)

print ("Final dataset has {} rows, {} columns.".format(*final_df.shape))
```

    Final dataset has 21770 rows, 35 columns.



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
      <th>index</th>
      <th>listing_url</th>
      <th>id</th>
      <th>description</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>property_type</th>
      <th>room_type</th>
      <th>accommodates</th>
      <th>bathrooms</th>
      <th>...</th>
      <th>extra_people</th>
      <th>minimum_nights</th>
      <th>maximum_nights</th>
      <th>availability_365</th>
      <th>cancellation_policy</th>
      <th>average_length_of_stay</th>
      <th>yield</th>
      <th>image_link</th>
      <th>NIMA_score</th>
      <th>description_topic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>https://www.airbnb.com/rooms/21456</td>
      <td>21456</td>
      <td>An adorable, classic, clean, light-filled one-...</td>
      <td>40.797642</td>
      <td>-73.961775</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>1.0</td>
      <td>...</td>
      <td>28.0</td>
      <td>5</td>
      <td>365</td>
      <td>248</td>
      <td>moderate</td>
      <td>5</td>
      <td>15552.00</td>
      <td>https://a0.muscache.com/im/pictures/111808/a94...</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>https://www.airbnb.com/rooms/2539</td>
      <td>2539</td>
      <td>Renovated apt home in elevator building. Spaci...</td>
      <td>40.647486</td>
      <td>-73.972370</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>4</td>
      <td>1.0</td>
      <td>...</td>
      <td>25.0</td>
      <td>1</td>
      <td>730</td>
      <td>365</td>
      <td>moderate</td>
      <td>3</td>
      <td>3132.00</td>
      <td>https://a0.muscache.com/im/pictures/3949d073-a...</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>https://www.airbnb.com/rooms/2595</td>
      <td>2595</td>
      <td>Find your romantic getaway to this beautiful, ...</td>
      <td>40.753621</td>
      <td>-73.983774</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1</td>
      <td>1125</td>
      <td>350</td>
      <td>strict_14_with_grace_period</td>
      <td>3</td>
      <td>8658.00</td>
      <td>https://a0.muscache.com/im/pictures/f028bdf9-e...</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>https://www.airbnb.com/rooms/21644</td>
      <td>21644</td>
      <td>A great space in a beautiful neighborhood- min...</td>
      <td>40.828028</td>
      <td>-73.947308</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>1</td>
      <td>1.0</td>
      <td>...</td>
      <td>55.0</td>
      <td>1</td>
      <td>60</td>
      <td>365</td>
      <td>strict_14_with_grace_period</td>
      <td>3</td>
      <td>4369.68</td>
      <td>https://a0.muscache.com/im/pictures/43197335/5...</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>https://www.airbnb.com/rooms/3330</td>
      <td>3330</td>
      <td>This is a spacious, clean, furnished master be...</td>
      <td>40.708558</td>
      <td>-73.942362</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>...</td>
      <td>50.0</td>
      <td>5</td>
      <td>730</td>
      <td>216</td>
      <td>strict_14_with_grace_period</td>
      <td>5</td>
      <td>8190.00</td>
      <td>https://a0.muscache.com/im/pictures/41842659/5...</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
# split the training set and testing set
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
seed = 42
X_train,X_test,y_train,y_test = train_test_split(final_df,target,random_state=seed)
```


```python
final_df.to_csv('final_df.csv')
```

### Linear regression 


```python
from sklearn.linear_model import LinearRegression

linreg = LinearRegression().fit(X_train, y_train)
y_pred_linreg = linreg.predict(X_test)

print("Mean squared error: %.3f" %mean_squared_error(y_test,y_pred_linreg))
print("Variance score: %.3f" %r2_score(y_test,y_pred_linreg))
```

    Mean squared error: 4781289307.734
    Variance score: 0.104


### Decision trees


```python
from sklearn.tree import DecisionTreeRegressor
for K in [1,3,5,7,10,15,20,30]:
    dt_reg = DecisionTreeRegressor(random_state = seed, max_depth = K).fit(X_train,y_train)
    y_dt_pred = dt_reg.predict(X_test)
    print ("max_depth = " + str(K))
    print ("Mean squared error: %.3f" %mean_squared_error(y_test,y_dt_pred))
    print ("Variance score: %.3f" %r2_score(y_test,y_dt_pred))
```

    max_depth = 1
    Mean squared error: 5024803151.882
    Variance score: 0.059
    max_depth = 3
    Mean squared error: 4789684788.300
    Variance score: 0.103
    max_depth = 5
    Mean squared error: 4588601834.546
    Variance score: 0.140
    max_depth = 7
    Mean squared error: 4941400033.278
    Variance score: 0.074
    max_depth = 10
    Mean squared error: 7397470407.079
    Variance score: -0.386
    max_depth = 15
    Mean squared error: 8291334303.409
    Variance score: -0.553
    max_depth = 20
    Mean squared error: 8903546780.280
    Variance score: -0.668
    max_depth = 30
    Mean squared error: 9127202991.476
    Variance score: -0.710


### Random forest


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
    Mean squared error: 4933940765.512
    Variance score: 0.076
    Max_depth = 3
    Mean squared error: 4792797066.583
    Variance score: 0.102
    Max_depth = 5
    Mean squared error: 4714308643.520
    Variance score: 0.117
    Max_depth = 7
    Mean squared error: 4708215874.375
    Variance score: 0.118
    Max_depth = 10
    Mean squared error: 4678417980.839
    Variance score: 0.124
    Max_depth = 15
    Mean squared error: 4653522451.420
    Variance score: 0.128
    Max_depth = 20
    Mean squared error: 4671374608.185
    Variance score: 0.125


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
             "max_depth": [3,5,7,9,11,15,20],
             "min_samples_split":[4,6,8,10,12],
             "bootstrap":[True]}

rf_fine = RandomForestRegressor(random_state = seed)
rf_cv = GridSearchCV(rf_fine,param_grid,cv=5).fit(X_train,y_train)
y_rf_cv_pred = rf_cv.predict(X_test)
print("Mean squared error: %.3f" % mean_squared_error(y_test, y_rf_cv_pred))
print('Variance score: %.3f' % r2_score(y_test, y_rf_cv_pred))
print("Best Parameters: {}".format(rf_cv.best_params_))
```

    Mean squared error: 4637113425.384
    Variance score: 0.131
    Best Parameters: {'bootstrap': True, 'criterion': 'mse', 'max_depth': 15, 'max_features': 'auto', 'min_samples_split': 12, 'n_estimators': 300}



```python
rf_final = rf_cv.best_estimator_
feature_import = rf_final.feature_importances_*100
feature_import = pd.DataFrame(list(zip(feature_import,X_train.columns.values)))
feature_import = feature_import.sort_values(by=0,axis=0,ascending=False)
feature_import.columns = ['importance %','feature']
print(feature_import[:20])
```

        importance %                                          feature
    1      18.040199                                        longitude
    0      17.378363                                         latitude
    2      12.976208                                     accommodates
    8      10.915914                                 availability_365
    9       8.753188                                       NIMA_score
    18      6.882709                        room_type_Entire home/apt
    7       5.939127                                   maximum_nights
    4       4.540373                                         bedrooms
    5       3.179635                                  guests_included
    6       3.088270                                     extra_people
    3       2.322952                                        bathrooms
    13      1.532435                              property_type_House
    34      0.658664                              description_topic_2
    14      0.462011                               property_type_Loft
    11      0.456593                        property_type_Condominium
    29      0.406894  cancellation_policy_strict_14_with_grace_period
    26      0.400477                     cancellation_policy_flexible
    32      0.353673                              description_topic_0
    19      0.343429                           room_type_Private room
    17      0.230167                          property_type_Townhouse


Location has a combined importance of 36% - 18.4% from longitude and 17.5% from latitude, which make sense to me. A convenient location can be very attractive for viewers. Other features such as "accommodates" and "availability_365" also occupied 13.5% and 11.1% importance. Interestingly, the __NIMA score__ engineered from photos on the website have __9.3%__ of importance. The other feature __"description_topic"__ also has combined __>1%__ of importance (sum of "description_topic_2" and "description_topic_0"). This information shows that there are valuable information in the photo and description text.

To test the robustness of the model, random_state for shuffling dataset will be changed, the ratio of training set and test set will also be changed to 0.3. 


```python
random_state = 35
X_train,X_test,y_train,y_test = train_test_split(final_df,target,test_size = 0.3,random_state=seed)
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

    Mean squared error: 4209814319.47
    Variance score: 0.14


There is no significant difference after adjusting the random state and proportions of training set and test set, which demonstrate that the final model is robust.

## VI. Conclusion and reflection

The original goal of this project is to apply machine learning algorithms to give potential hosts some insights on how much they can earn from listing their beloved houses on Airbnb. The information from Inside Airbnb is definitely very helpful. Combined my own experience of browsing accommodations in Airbnb, I added two additional features: image score and topic modeling from web photos and descriptions. It turned out that these two features actually contain valuable informations. Of couse, my solution is not perfect, here are two points I would like to spend more time on further improving my model.

1. There are some overlaps among the three topics, so potential improvement would be implement the topic modeling methods. It would be worthwhile comparing LDA with other algorithms, such as Non-negative matrix factorization.

2. Should I consider time effect? If a host gets very positive reviews from first few guests, it's possible that new viewers will also consider choosing their houses. How should I predict time series?

Having spent lots of time on data in chemistry as a chemistry PhD candidate, I always want to know the data in real world and what we can learn from it. I had a great time when doing this project. From designing the workflow, analyzing images to text mining, I learnt a lot and am very pleased to see that my model has suggested some informations from images and text on the website. In the end, I would also like to learn how to design a web application where hosts can actually upload their photos and informations about their houses and get an estimation of their yield using the model.

