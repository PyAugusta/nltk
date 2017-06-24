## Sentiment Analysis Using Tweets and the Python Natural Language Toolkit #

### Step 1 - Install Dependencies #

After pulling this repository, use pip to install the packages listed in the requirements.txt file

```
$ cd path/to/nltk/
$ pip install -r requirements.txt
```

### Step 2 - Download the nltk data sets using the get_tutorial_data.py file #

```
$ python get_tutorial_data.py
```

### Step 3 - Run the test_classifiers.py file #

This script uses the twitter data set (downloaded in step 2) which contains two JSON files that have 
already been manually classified for us. One contains positive tweets, and one contains negative tweets.

Examine the script for more specific details of what it is doing - here's a brief overview:

1. Import statements:
    - We'll need quite a few things, but the important package here is nltk. Of note are the two classifiers we'll be testing:
        1. [DecisionTreeClassifier](http://mines.humanoriented.com/classes/2010/fall/csci568/portfolio_exports/lguo/decisionTree.html)
        2. [NaiveBayesClassifier](https://stackoverflow.com/questions/10059594/a-simple-explanation-of-naive-bayes-classification)


2. Set up our logger:
    - We'll include two handlers, one for console output, and one to write to a log file.
    
3. Directories of interest:
    - This is something that I do fairly often when working with data on disk. Basically, we're just defining globals that point to the absolute path of the directories we'll be accesing.
    
4. GLOBALS:
    - TOKENS - all our words of interest
    - NEGATIVE - the label we'll use for negative tweets
    - POSITIVE - the lable we'll user for positive tweets
    
5. Classes and functions:
    - These will be used by our main function to convert our tweets to features, and pickle away the things we'll need later on.
    - Check out the comments in the code for more information
    
6. The main function:
    - We'll load in the data, featurize it, then train and test our classifiers.





