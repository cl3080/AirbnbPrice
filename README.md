# Airbnb Rental Income Prediction

<p align="justify"> 
Airbnb is a popular home-sharing platform that provides people all over the world a profitable option to share their empty houses or extra rooms. For potential hosts, they must be interested in how much they can earn from listing their houses on Airbnb. So far, there is no public model for predicting the yield of a new listing on Airbnb. So, the goal of this project is to build a prediction model to help potential hosts gain some intuitions about the yield of their listings. </p>

<p align="justify"> 
Fortunately, Inside Airbnb has already aggregated the publicly available information from Airbnb site for public discussion. So, the dataset downloaded from this website is a good starting point. In particular, I will use the dataset collected in Los Angeles, compiled on 6th, December, 2018. This dataset contains 43047 listings and 96 columns. Before any model, I cleaned up the dataset as follows:</p> 

1.	 For a new house, there won’t be any information about reviews, so columns containing information about reviews should be dropped. 
2.	Check the distribution of variables. Some listings with extremely high prices were removed. Rooms that are unavailable for most days within a year are not representative, they were also removed.
3.	Clean up the dataset: handling missing values, correct data type for some columns.

<p align="justify"> 
Besides the features already in the dataset, there are two features are also important: photo and description of your house. An attractive photo with appropriate description can boost the room’s popularity, so I engineered two more features and added them into the model to improve the accuracy.</p> 

1.	**Photo of the listed house.** The original dataset has the URL link to the webpage of the corresponding listings, so the photos can be scraped from the internet. An attractive photo should have desirable resolution and also be aesthetically attractive. Here I used NIMA: Neural Image Assessment provided by Google to score the image quality on scale of 1 to 10. NIMA scores of some random samples in the dataset are shown below.
![NIMA score](https://github.com/cl3080/Machine_Learning_Models_for_Airbnb_Yield_Prediction/blob/master/NIMA_score_sample_check.png)

2.	**Description of the house.** Sentiment analysis on description of the house was carried out using nature language processing. To discover the abstract topics hidden in the text, topic model latent Dirichlet allocation was used. The screenshot of visualizing LDA topic model is shown below, the interactive html page is named as lda_description.html in the repository.
![LDA topic model](https://github.com/cl3080/Machine_Learning_Models_for_Airbnb_Yield_Prediction/blob/master/Screen_shot_forLDA_model.png))

The project will be described as follows:
 1. **Exploratory data analysis and data preprocessing:** Get an insight of the dataset and clean the data for later analysis.
 2. **Feature engineering:** Analyze the text in "description" and featured photo on the listing website. Topic and image score will be added as two new features for the machine learning model.
 3. **Machine learning model:** Apply different machine learning algorithms and fine tune the best model.
 4. **Model evaulation:** Evaluate the robustness of the final model and analyze the importance of features.
 
<p align="justify"> 
In conclusion, the purpose of this project is to help potential hosts to predict the income they can get by listing their extra houses on Airbnb. Furthermore, it is nice to show that the performance of the machine learning model can be improved by incorporating the information captured from photo and description of the houses. From business point of view, it will be useful for the data science team in such a company to know that there is lots of valuable information in text and images when building similar prediction models.</p> 
 
 The overall summary of the project can be checked here: https://chaoli.blog/2019/01/25/machine-learning-model-for-airbnb-yield-prediction/
    
 ## Author
 __Chao Li__
