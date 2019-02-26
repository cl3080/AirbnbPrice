# Price Prediction for New Airbnb Listings

<p align="justify"> 
Airbnb is a popular home-sharing platform that provides people all over the world a profitable option to share their empty houses or extra rooms. For potential hosts, it might be difficult for them to decide how much they should charge for new listings. As far as I know, there is no public model on Airbnb website for predicting the price of a new listing on Airbnb. So, the goal of this project is to build a prediction model to help potential hosts get some descent ideas on optimal price for their beloved houses. </p>

<p align="justify"> 
Fortunately, Inside Airbnb has already aggregated the publicly available information from Airbnb site for public discussion. So, the dataset downloaded from this website is a good starting point. In particular, I will use the dataset collected in New York, compiled on 6th, December, 2018. This dataset contains 49056 listings and 96 columns.:</p> 

<p align="justify"> 
Besides the features already in the dataset, there are two features I think might also be important: photo and description of the listings. An attractive photo with appropriate description can boost the room’s popularity, so I engineered two more features and added them into the model.</p> 

1.	**Photo of the listed house.** The original dataset has the URL link to the webpage of the corresponding listings, so the photos can be scraped from the internet. An attractive photo should have desirable resolution and also be aesthetically attractive. Here I used NIMA: Neural Image Assessment provided by Google to score the image quality on scale of 1 to 10. NIMA scores of some random samples in the dataset are shown below.
![NIMA score](https://github.com/cl3080/Machine_Learning_Models_for_Airbnb_Yield_Prediction/blob/master/NIMA_score_sample_check.png)

2.	**Description of the house.** Sentiment analysis on description of the house was carried out using nature language processing. To discover the abstract topics hidden in the text, topic model latent Dirichlet allocation was used. The screenshot of visualizing LDA topic model is shown below, the interactive html page is named as lda_description.html in the repository.
![LDA topic model](https://github.com/cl3080/Machine_Learning_Models_for_Airbnb_Yield_Prediction/blob/master/Screen_shot_forLDA_model.png))

The project will be described as follows:
 1. **Exploratory data analysis and data preprocessing:** Get an insight of the dataset and clean the data for later analysis.
 2. **Feature engineering:** Analyze the text in "description" and web photo for each listings. Topic and image score will be added as two new features for the machine learning model.
 3. **Machine learning model and refinement:** Apply machine learning algorithms and fine tune the best model.
 4. **Future work:** 
 5. **Conclusion:**
 
<p align="justify"> 
In conclusion, the purpose of this project is to help potential hosts to decide the price for their new listings. It also shows that there are lots of valuable information can be captured from web photos and descriptions when building a similar machine learning model. Of course, the model shown here is not perfect and needs further investigation.</p> 
 
 The overall summary of the project can be checked here: https://chaoli.blog/2019/01/25/machine-learning-model-for-airbnb-yield-prediction/
    
 ## Author
 __Chao Li__
