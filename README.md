
<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="https://i.imgur.com/6wj0hh6.jpg" alt="Project logo"></a>
</p>

<h3 align="center">Predicting IMDB Ratings</h3>

<div align="center">

</div>


## 📝 Table of Contents
- [About](#about)
- [Data Understanding and Exploring](#data_understanding_and_exploring)
- [Data Preparation](#data-preparation)
- [Data Modeling and Conclusion](#data-modeling)

## 🧐 About <a name = "about"></a>

Film Industry is not only a industry or a centre of entertainment but also it is a centre of global business. The world is now excited about a movie's box office success, failure, popularity etc. A huge data is coming every day and available online about these movies success or popularity. 

I have used hollywood movie data set and their rating from IMDb movie rating. I applied some machine learning classification algorithms such as linear regression,Random forest, KNN model. Lastly an efficient model is developed to predict a movie's IMDb rating.

## 🎈 Data Understanding and Exploring <a name="data_understanding_and_exploring"></a>

![alt text](https://github.com/cghimire/Predicting-IMDB-Ratings/blob/master/Img/distribution_rating.png "Distribution Plot")

*This plot demonstrates the distribution of IMDb ratings*.
![alt text](https://github.com/cghimire/Predicting-IMDB-Ratings/blob/master/Img/fblikes_rating.png "fblikesVSrating")

*This plot demonstrates that social media would be an excellent place to estimate the popularity of movies. From the scatter plot above, we can find that the movies with high facebook likes tend to be the ones that have IMDb scores around 7.0 to 8.0. It is interesting to see that the good movies, with an IMDb score of around 9.0, does not have more Facebook popularity.*

## ⛏️ Data Preparation <a name = "data-preparation"></a>

![alt text](https://github.com/cghimire/Predicting-IMDB-Ratings/blob/master/Img/correlation.png "correlation matrix")

*This figure shows that the correlation matrix between variables. The "cast_total_facebook_likes" has a strong positive correlation with the "actor_1_facebook_likes", and has smaller positive correlation with both "actor_2_facebook_likes" and "actor_3_facebook_likes" ans so on.*

## 🚀 Data Modeling and Conclusion <a name = "data-modeling"></a>

In order to model the data, I performed regression techniques: 1)Linear Regression 2)KNN Model 3) Random Forest Model.

![alt text](https://github.com/cghimire/Predicting-IMDB-Ratings/blob/master/Img/KNN.png "KNN")

* Above plot shows that the accuracy Vs value of K*.

The accuracy for linear regression model is 66%. Which is not a good model to predict imdb rating. The KNN model gives 81% of accuracy. The Random forest model has learned how to predict the imdb rating with 90% accuracy.

By comparing the results, I would suggest Random Forest Model could be useful to deploy into production.
