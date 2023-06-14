# Speech Emotion Recognition

This project aims to identify the emotion present in an instance of a person speaking, using machine learning. Emotions are categorized into happy, sad, neutral, surprised, and angry. Both voice features extraction and machine learning model construction are done in Python.

## Dataset

The dataset utilized for training and evaluating the models is the [Emotional Speech Dataset (ESD)](https://github.com/HLTSingapore/Emotional-Speech-Data). The dataset consists of voice files of 20 different actors for each class (emotion), half of them being English and the other half being Chinese, also for each language there are 5 male and 5 female actors. Emotions are happy, sad, neutral, surprise, and angry.
To avoid overfitting, the actors selected for training and testing are different. Specifically, the actors selected for training are: 
* Chinese Male: 4, 6, 10
* Chinese Female: 2, 3, 7
* English Male: 11, 12, 20
* English Female: 15, 17, 18

and the remaining were selected for testing. It is notable that the choice of actors had to be random and to include both languages and genders, this is to cover more diverse aspects of the actual problem.

## Feature Engineering

Excluding the essential features, the main method for choosing features was by experimentation. *librosa* Python library was used to extract most features. One notable issue is that most audio features are dependent upon the length of the voice file, but the voice file length is variable. Therefore, different statistical measurements were taken on the set of values of the feature, and after experimenting the mean proved to the most indcative measurement, and in the case of min&max, the results dropped in quality. The chosen features can be found in `FeatureExtraction.py`, however, the most important ones are:
* **Mel Frequency Cepstral Coefficients (MFCCs)**: Probably the most indicative feature and is represented as a vector of values, it takes the number of coefficients as input which can heavily  increase the time needed for calculation. The number of coefficients is set to 40.
* **Flatness**: It had good impact on models especially in distinguishing happy and surprised. It is a single value representing the mean of all frames.
* **Pitch (F0)**: Increased the models' ability to predict male and female voices, after adding it, the misclassification of neutral females as happy/surprised males was notably lower. It was standardized on a single voice file scope then the mean was taken.
* **Mel Spectrogram**: A logarithmically scaled frequency to simulate how humans hear sound, represented as a vector, and was averaged to reduce the lengh of the vector instead of having a single value.

It is also notable that Kurtosis did not have a big impact on the models and removing it won't necessarily harm the quality. Finally, the length of the feature vector is 73.

## Data Preprocessing

Before extracting the features from a voice file, it was needed to remove the silence it had so it doesn't affect the values of the features. Next, features were extracted from each voice file (18000 voice files for training and 12000 ones for testing) forming the whole dataset. Additionally, the dataset was normalized before supplying it to some models for training, however, the same scale used for the training set was also applied on the test set. Further detail and visualization can be found in the `SpeechEmotionRecognition.ipynb`.

## Model Evaluation

Since the classes are completely balanced, accuracy will be used as the primary evaluation metric. For charts and other metrics, refer to `SpeechEmotionRecognition.ipynb`.

### Neural Network

The deep neural network model shows signs of overfitting, having an accuracy exceeding 90% on the training set while scoring an accuracy of approximately 70% on the test set (probably due to the depth of the network), which is overall acceptable. However, it returns a vector of probabilities for each class, which makes the accuracy of classification a bit misleading.

### SVM

Using the RBF kernel and accepting a normalized dataset, it yielded an accuracy of 65% indicating that the data is complex for a simple straightforward method and might need more data preprocessing and hyperparameter tuning for the model. Normalization boosted the performance of this model.

### Random Forest

Bagging seems to be suitable for this problem, as Random Forest has achieved a relatively high accuracy of 89% with 650 decision tree estimators, topping the other algorithms used. This can be due to the high dimensionality of the dataset, since bagging can reduce the variance reducing possible overfitting.

### XGBoost and AdaBoost

On the other hand, boosing seems just as effective as bagging in this case, with XGBoost and AdaBoost exceeding 87% in accuracy. It is noteworthy that although the number of estimators used in XGBoost is way lower than those used in AdaBoost, XGBoost consumed much more time training than AdaBoost probably because of the usage of 'dart' booster.

## Conclusion

The best model reached a 89% accuracy which is probably not applicable in a real example. A better approach is formulate this problem as multi-lable classification one, because emotions are not necessarily mutually exclusive. Also, adding more suitable features and constructing larger models, especially ensemble ones, can largely improve the generalization.
