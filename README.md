# Airbnb Rental Income Prediction

<p align="justify"> 
Airbnb is a popular home-sharing platform that provides people all over the world a profitable option to share their empty houses or extra rooms. For potential hosts, they must be interested in how much they can earn from listing their houses on Airbnb. So far, there is no public model for predicting the yield of a new listing on Airbnb. So, the goal of this project is to build a prediction model to help potential hosts gain some intuitions about the yield of their listings. </p>

 <p align="justify"> 
Fortunately, Inside Airbnb (http://insideairbnb.com) has already aggregated all the publicly available informations from Airbnb site for public discussion. So, the dataset obtained from this website directly should be a good starting point for my machine learning model. In particular, I will the dataset collected in Los Angeles compiled on 06 December, 2018. 

Besides the features already in the dataset, there are two features are also important: **photo** and **description** of your house. An attractive photo with appropriate description can boost the room’s popularity, so I will engineer two more features and add them into the model to improve the accuracy. </p>

The project will be described as follows:
 1. **Exploratory data analysis and data preprocessing:** Get an insight of the dataset and clean the data for later analysis.
 2. **Feature engineering:** Analyze the text in "description" and featured photo on the listing website. Topic and image score will be added as two new features for the machine learning model.
 3. **Machine learning model:** Apply different machine learning algorithms and fine tune the best model.
 4. **Model evaulation:** Evaluate the robustness of the final model
NIMA Score of same radom samples in the dataset
![NIMA score](https://github.com/cl3080/Machine_Learning_Models_for_Airbnb_Yield_Prediction/blob/master/NIMA_score_sample_check.png)


 LDA Model 
 ![Image of LDA Model](https://github.com/cl3080/Machine_Learning_Models_for_Airbnb_Yield_Prediction/blob/master/Screen_shot_forLDA_model.png)
 
In conclusion, the purpose of this project is helping potential hosts to predict the income they can get by listing their extra houses on Airbnb. Furthermore, it would be nice to show that the performance of the machine learning model can be improved by incorporating the information captured from photo and description of the houses. From business point of view, it will be useful for the data science team in such a company to know that there is lots of valuable information in text and images when building similar prediction models.
 
 The overall summary of the project can be checked here: https://chaoli.blog/2019/01/25/machine-learning-model-for-airbnb-yield-prediction/
    
 ## Author
 __Chao Li__
