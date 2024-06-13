# Asteroid-analysis
In this project, I took on the important task of identifying hazardous asteroids to keep our planet safe. To achieve this, I harnessed the power of machine learning, employing eight different models to aid in this crucial mission. Each model brought its unique capabilities, allowing me to explore various ways to spot potentially dangerous asteroids.

First, I used logistic regression, providing a solid foundation to understand how different factors contribute to an asteroid being hazardous. Next, I implemented random forests, acting like a diverse group of advisors by creating numerous decision trees to classify asteroids.

I then employed a decision tree model, which helped me make decisions by asking yes-or-no questions to determine if an asteroid is hazardous based on specific features. Additionally, I utilized the support vector machine (SVM), functioning like a wise judge drawing clear lines between different types of asteroids for accurate classification.

For another approach, I used the K-Nearest Neighbors (KNN) model, which relies on nearby asteroids to decide if a new one is hazardous. I also incorporated a neural network, inspired by the human brain, to recognize complex patterns in asteroid data.

Moreover, I applied the Gaussian Naive Bayes classifier, akin to a detective using probabilities to assess if an asteroid is hazardous based on past data. Lastly, I utilized Gradient Boosting Machines (GBM), where a team of models joined forces, each contributing their strengths to predict asteroid hazards.

With these eight models at my disposal, I explored, learned, and worked towards making our planet safer from asteroid threats.

5. Project Setup
5.1. Environment Setup
The project was conducted entirely on Google Colab, leveraging its hosted Jupyter environment. This choice allowed for seamless collaboration and access to powerful computing resources.
5.2. Libraries
We incorporated several essential libraries to facilitate data analysis and machine learning tasks:
5.3. pandas:
This library was instrumental in data manipulation and analysis, enabling us to read, clean, and preprocess our dataset ebiciently.
5.4. matplotlib and seaborn:
These visualization libraries were utilized for creating informative plots and charts to explore the data visually.
5.5. scikit-learn (sklearn):
A comprehensive machine learning library that provided a wide range of algorithms and tools for model building and evaluation. Key modules included RandomForestClassifier, train_test_split, LabelEncoder, MinMaxScaler, DecisionTreeClassifier, SVC, KNeighborsClassifier, MLPClassifier, GaussianNB, GradientBoostingClassifier, GridSearchCV, and LogisticRegression.
5.6. imbalanced-learn (imblearn):
This library played a crucial role in handling class imbalance within the dataset, employing techniques such as random undersampling, oversampling (SMOTE), and combination sampling (SMOTEENN).
5.7. numpy:
Used for ebicient numerical computing and array operations, numpy provided essential support for handling large datasets and performing mathematical computations.
5.8. collections:
Specifically, the Counter class from the collections module was utilized for counting occurrences of elements within our data, aiding in data preprocessing tasks.
By leveraging these libraries, we were able to streamline our data preprocessing workflow and build robust machine learning models ebectively.

6. Data Preprocessing 6.1. Data Read and Analysis
First, we read our asteroid dataset called 'nasa.csv' using a tool called pandas. This helped us understand what our data looks like, how many rows and columns it has, and what type of information each column holds.
6.2. Missing Values Check
Next, we looked for any missing information in our dataset. We wanted to make sure that all the data was complete and there were no empty spaces where important information should be.
6.3. Data Visualization
We then created histograms to visualize our data better. Histograms are like bar graphs but show us how common or rare diberent values are in our dataset. This helped us see if there were any patterns or unusual things in our data.
6.4. Target Variable Transformation
Our main goal was to predict whether asteroids are hazardous or not. So, we converted the information about whether an asteroid is hazardous into numbers. We made it so that '1' means hazardous and '0' means not hazardous. This makes it easier for computers to understand and work with this information.
6.5. Encoding Categorical Variables
Some of our data was not in numbers; it was in categories like 'asteroid class.' We used a method called LabelEncoder to change these categories into numbers. This helps the computer understand these categories better when we use them in our analysis.
6.6. Feature Selection based on Correlation
We wanted to focus on the most important information for predicting if an asteroid is hazardous. So, we looked at how each piece of information relates to our main goal. We kept only the information that seemed most useful and related to our prediction.

6.7. Feature Importance Calculation
We used a special tool called RandomForestClassifier to figure out which pieces of information were the most important for our prediction. This helped us identify the most helpful features and ignore the ones that didn't make much of a diberence.
6.8. Outlier Detection and Removal
Sometimes, our data can have strange values that don't fit with the rest. We call these outliers. We found and removed these outliers from our dataset to make sure our analysis wasn't abected by these unusual values.
6.9. Data Normalization
Lastly, we wanted to make sure all our data was in a similar range. We used a method called min-max normalization, which scaled our data so that it all fell between 0 and 1, you can see figure 1 for visual representation of the numeric columns. This makes it easier for the computer to understand and work with our data accurately.
7. Models
7.1. Logistic Regression
Logistic regression is a simple but useful tool for categorizing things. Here's how we made it work and saw how well it did:
•
7.1.1 Making the Logistic Regression Model:
First, we picked some settings for our model, like how strict it should be and how much importance to give to diberent features. Then, we set up the logistic regression model using these settings.
Next, we used a method called GridSearchCV, which is like a smart way of trying out diberent settings to find the best ones. This helps us make sure our model works as best as it can.
After training our model with the best settings on our training data, we checked how good it was at predicting things on the test data.
7.1.2 What We Found Out:
Best Settings: We found that setting 'C' to 100 and using 'penalty' as 'l1' gave us the best results.

• Accuracy: Our model was about 98.9% accurate, which means it got things right most of the time.
• What the Numbers Mean: The classification report gave us a detailed breakdown of how well our model did in diberent areas.
• The Right Balance: We found that setting the penalty as 'l1' and 'C' as 10 worked best for our model.
7.1.3 How We Used It:
We trained our logistic regression model with these best settings on our training data. Then, we checked how well it could predict things on both the training and testing data.
7.1.4 Our model did pretty well:
• On Training Data: It was right about 99.1% of the time.
• On Testing Data: It was right about 98.9% of the time.
Seeing that the model performed almost as well on the testing data as it did on the training data tells us that it's doing a good job without getting too caught up in the specifics of the training data. This means it's likely to work well with new data too.
7.2. Random Forest
Random forest is like a big group of decision trees working together to make better predictions. Here's how we used it and what we found out:
7.2.1 Setting Up the Random Forest Model:
We started by deciding on some values to try out for diberent settings in our random forest model. These settings help our model make decisions in a smart way.
Then, we set up the random forest classifier using these settings and used GridSearchCV, a smart way to try out diberent settings and find the best ones.
After training our model with the best settings on our training data, we checked how good it was at predicting things on the test data.
7.2.2 What We Found Out:
• Best Settings: The best settings we found were 'max_features' as 'auto', 'min_samples_leaf' as 1, and 'min_samples_split' as 2.
• Accuracy: Our model was incredibly accurate, with an accuracy of about 99.8%, which means it got things right almost all the time.
• What the Numbers Mean: The classification report gave us a detailed breakdown of how well our model did in diberent areas.

• Visualizing Results: We visualized the confusion matrix to see how our model performed in predicting positive and negative outcomes.
7.2.3 How We Used It:
We trained our random forest model with these best settings on our training data. Then, we checked how well it could predict things on the testing data.
7.2.4 Our model performed exceptionally well:
• On Testing Data: It was right about 99.8% of the time, which is remarkable.
• Understanding the Results: The precision, recall, and F1-score were all perfect, indicating that our model did an excellent job of predicting both
positive and negative outcomes.
7.3. Decision Tree
A decision tree is like a flowchart that helps us make decisions based on diberent conditions. Here's what we did with it and what we found:
7.3.1 Setting Up the Decision Tree Model:
We started by deciding on diberent options to try out for various settings in our decision tree model. These settings help our model make decisions in the best way possible.
Then, we set up the decision tree classifier using these settings and used GridSearchCV, which is like a smart way to try out diberent settings and find the best ones.
After training our model with the best settings on our training data, we checked how good it was at predicting things on the test data.
7.3.2 What We Found Out:
• Best Settings: The best settings we found were 'max_depth' as None, 'min_samples_leaf' as 2, and 'min_samples_split' as 10.
• Accuracy: Our model was highly accurate, with an accuracy of about 99.8%, which means it got things right almost all the time.
• What the Numbers Mean: The classification report gave us a detailed breakdown of how well our model did in diberent areas.
• Visualizing Results: We visualized the confusion matrix to see how our model performed in predicting positive and negative outcomes.
7.3.3 How We Used It:
We trained our decision tree model with these best settings on our training data. Then, we checked how well it could predict things on the testing data.
7.3.4 Our model performed exceptionally well:

• On Testing Data: It was right about 99.8% of the time, which is remarkable.
• Understanding the Results: The precision, recall, and F1-score were all perfect, indicating that our model did an excellent job of predicting both positive and
negative outcomes.
7.4. SVM (Support Vector Machine)
7.4.1 SVM (Support Vector Machine)
SVM, or Support Vector Machine, is a type of algorithm used for classification tasks. Here's how we used it and what we discovered:
7.4.2 Setting Up the SVM Model:
We started by defining diberent options to try out for various settings in our SVM model. These settings help our model classify data more accurately.
Then, we set up the SVM classifier using these settings and used GridSearchCV, which is like a smart way to try out diberent settings and find the best ones.
After training our model with the best settings on our training data, we checked how good it was at predicting things on the test data.
7.4.3 What We Found Out:
• Best Settings: The best settings we found were 'C' as 100, 'gamma' as 'scale', and 'kernel' as 'rbf'.
• Accuracy: Our model was highly accurate, with an accuracy of about 99.4%, which means it got things right almost all the time.
• What the Numbers Mean: The classification report gave us a detailed breakdown of how well our model did in diberent areas.
• Visualizing Results: We visualized the confusion matrix to see how our model performed in predicting positive and negative outcomes.
7.4.4 How We Used It:
We trained our SVM model with these best settings on our training data. Then, we checked how well it could predict things on the testing data.
7.4.5 Our model performed exceptionally well:
• On Testing Data: It was right about 99.4% of the time, which is impressive.
• Understanding the Results: The precision, recall, and F1-score were all excellent, indicating that our model did a fantastic job of predicting both positive and
negative outcomes.

7.5. K-Nearest Neighbors (KNN)
In K-Nearest Neighbors (KNN), we try to predict whether something is hazardous by looking at the "neighbors" closest to it. Here's how we did it:
First, we listed out diberent options for settings that KNN can use. These settings help KNN make decisions about which neighbors to consider when predicting.
Then, we tried out all these diberent settings combinations to see which one gives the most accurate results. For each combination, we trained a KNN model using those settings.
After training each model, we tested it on our test data to see how accurate it was at predicting whether something is hazardous or not.
7.4.6 Here's what we found:
• Best Model Parameters: The best settings we found were using 3 neighbors, with uniform weights, and using the auto algorithm.
• Best Accuracy: The accuracy of our best model was about 99.3%, which means it got things right most of the time.
• Classification Report: This report gives us a detailed breakdown of how well our model did in diberent areas, like precision, recall, and F1-score.
We also visualized the results using a confusion matrix to see how our model performed in predicting positive and negative outcomes.
Overall, KNN did a great job at predicting hazardous asteroids, especially when using the best settings we found.
7.6. Neural Network (MLP)
7.4.7 Neural Network (MLP)
In our neural network (MLP), we used diberent settings to make predictions about hazardous asteroids.
7.4.8 Here's what we did:
First, we tried out diberent options for settings that our neural network can use. These settings help the network decide how to learn from the data.
We tested diberent combinations of these settings to see which one gives the most accurate results. For each combination, we trained a neural network using those settings.

After training each model, we tested it on our test data to see how accurate it was at predicting whether something is hazardous or not.
7.4.9 Here's what we found:
• Best Model Parameters: The best settings we found were using two hidden layers, each with 50 neurons, using the "tanh" activation function, and the "adam" solver.
• Best Accuracy: The accuracy of our best model was about 99.8%, which means it got things right almost all the time.
• Classification Report: This report gives us a detailed breakdown of how well our model did in diberent areas, like precision, recall, and F1-score.
We also visualized the results using a confusion matrix to see how our model performed in predicting positive and negative outcomes.
However, it seems like our model struggled a bit with predicting negative outcomes, as indicated by the low precision and recall values for the negative class. This is something we might need to improve upon in future iterations.
7.7. Gaussian Naive Bayes Classifier
In our Gaussian Naive Bayes classifier, we wanted to see how well it could predict hazardous asteroids based on diberent smoothing parameters. Here's what we did:
We tried out diberent smoothing parameters to see how they abect the classifier's performance. Smoothing helps the model make better predictions by adjusting the probabilities.
For each smoothing parameter, we trained a Naive Bayes classifier using that parameter.
After training each model, we tested it on our test data to see how accurate it was at predicting whether something is hazardous or not.
7.4.10 Here's what we found:
• Best Model Parameters: The best smoothing parameter we found was 1e-9.
• Best Accuracy: The accuracy of our best model was about 87.2%, which means
it got things right most of the time.
• Classification Report: This report gives us a detailed breakdown of how well our
model did in diberent areas, like precision, recall, and F1-score.
We also visualized the results using a confusion matrix to see how our model performed in predicting positive and negative outcomes.

However, similar to our neural network model, it seems like our model struggled with predicting negative outcomes, as indicated by the low precision and recall values for the negative class. This is something we might need to improve upon in future iterations.
7.8. Gradient Boosting Machines (GBM)
In Gradient Boosting Machines (GBM), we aimed to find the best combination of settings to make accurate predictions about hazardous asteroids. Here's what we did:
We tested diberent settings, called hyperparameters, to see how they abect the accuracy of our model. These settings include the number of boosting stages, learning rate, maximum depth of the trees, and the minimum number of samples required to split a node or be a leaf node.
For each combination of hyperparameters, we trained a GBM model and then tested it to see how well it predicts whether an asteroid is hazardous or not.
7.4.11 Here's what we found:
• Best Model Parameters: The best combination of settings we found was:
- Number of Boosting Stages: 150
- Learning Rate: 0.1
- Maximum Depth: 3
- Minimum Samples to Split: 2
- Minimum Samples in Leaf: 1
• Best Accuracy: With these settings, our model achieved an accuracy of about
99.9%, meaning it was highly accurate in its predictions.
• Results Table: We organized our results into a table, showing the diberent
combinations of settings we tried and their corresponding accuracies.
By finding the best combination of settings, our GBM model can better predict whether an asteroid is hazardous or not, which is crucial for identifying potential threats from space.

