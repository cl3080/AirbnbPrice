# Machine Learning Models for Airbnb Yield Prediction

<p align="justify">   Airbnb is a great platform that provides people online marketplace and service to arrange or offer lodging. As a travel enthusiast, Airbnb is always my first choice when I am planning a trip. By using filters such as location and room type, I can search for accomodations efficiently. Hosts need to provide details for their houses including prices. For potential hosts, I guess they must be very interested in how much they could earn from putting their houses on Airbnb. As far as I know, there is no such a model for predicting the yield of a new house on Airbnb. So, the object of this project is to build a machine learning model to help potential hosts have intuition about the yield of their listed houses.  </p>

 <p align="justify">   Fortunately, [Inside Airbnb](http://insideairbnb.com/get-the-data.html) has already aggregated all the publicly available informations from Airbnb site for public discussion. So, the dataset obtained from this website directly should be a good starting point for my machine learning model. In particular, I will the dataset collected in New York city compiled on 06 December, 2018. When selecting features for machine learning model, besides the variables provided in the datasets, the __featured photo__ on the listing's website and the __description__ of listing can be crucial for attracting more guests. So, I will analyze featured photos and text mining on the descriptions and add these two new features to improve the machine learning model. </p>

The project will be described as follows:
 1. **Exploratory data analysis and data preprocessing:** Get an insight of the dataset and clean the data for later analysis.
 2. **Feature engineering:** Analyze the text in "description" and featured photo on the listing website. Topic and image score will be added as two new features for the machine learning model.
 3. **Machine learning model:** Apply different machine learning algorithms and fine tune the best model.
 4. **Model evaulation:** Evaluate the robustness of the final model
    
 ## Author
 __Chao Li__
 
## Reference
https://towardsdatascience.com/improving-airbnb-yield-prediction-with-text-mining-9472c0181731

