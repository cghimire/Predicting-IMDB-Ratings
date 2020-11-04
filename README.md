
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
- [Data Modeling](#data-modeling)
- [Model Evaluation and Conclusion](#model-evaluation-and-conclusion)
- [Acknowledgments](#acknowledgement)

## 🧐 About <a name = "about"></a>

Film Industry is not only a industry or a centre of entertainment but also it is a centre of global business. The world is now excited about a movie's box office success, failure, popularity etc. A huge data is coming every day and available online about these movies success or popularity. 

I have used hollywood movie data set and their rating from IMDb movie rating. I applied some machine learning classification algorithms such as linear regression,Random forest, KNN model. Lastly an efficient model is developed to predict a movie's IMDb rating.

## 🎈 Data Understanding and Exploring <a name="data_understanding_and_exploring"></a>

![alt text](https://github.com/cghimire/Predicting-IMDB-Ratings/blob/master/Img/distribution_rating.png "Distribution Plot")

*This plot demonstrates the distribution of IMDb ratings*.
![alt text](https://github.com/cghimire/Predicting-IMDB-Ratings/blob/master/Img/fblikes_rating.png "fblikesVSrating")

*This plot shows the number of clients Vs job category. The highest number of clients are from the job category "admin" followed by blue-color category. Similarly,
there are less students involved in the telemarketing campaign*.

## ⛏️ Data Preparation <a name = "data-preparation"></a>

![alt text](https://github.com/cghimire/Predicting-IMDB-Ratings/blob/master/Img/correlation.png "correlation matrix")

*This figure shows that the correlation matrix between variables. The "cast_total_facebook_likes" has a strong positive correlation with the "actor_1_facebook_likes", and has smaller positive correlation with both "actor_2_facebook_likes" and "actor_3_facebook_likes" ans so on.*

## 🚀 Data Modeling <a name = "data-modeling"></a>

In order to model the data, I am performing three data-mining classification techniques: 1) Logistic Regression 2)Decision Tree Model 3) Random Forest Model.

![alt text](https://github.com/cghimire/Bank-Marketing-Data-Mining/blob/master/Figures/Decision%20Tree_final.png "Decision Tree")


*This figure represents the decision tree structure. For example, If number of employed is greater than 5088, then that client belongs to NO category with 94% of probability: that means the client is more likely to say NO*.

## Model Evaluation and Conclusion <a name = "model-evaluation-and-conclusion"></a>

![alt text](https://github.com/cghimire/Bank-Marketing-Data-Mining/blob/master/Figures/AccuracyVsTreeSize.png "Accuracy Vs Treesize")

*This figure shows Effect of increasing tree count on accuracy in Random Forest Model*.

I performed three different classification models to classify whether a customer would open a bank account or not. Based on the model build for this project, Decision Tree and Random Forest model are more accurate to predict the output. The Random Forest model is a recommended model for this classification problem.

Since I have been using different data mining techniques, I am expecting the proposed classification models are powerful to predict the output. However, the proposed methods has some limitations. It is not feasible to study all the variables in detail, which might be interesting to predict the output, because of time limitation.

## 🎉 Acknowledgements <a name = "acknowledgement"></a>
