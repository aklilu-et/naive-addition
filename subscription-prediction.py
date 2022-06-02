# Databricks notebook source
# MAGIC %md
# MAGIC # Subscription Prediction with PySpark and MLlib

# COMMAND ----------

# MAGIC %md
# MAGIC ## Learning Objectives

# COMMAND ----------

# MAGIC %md
# MAGIC At the end of this session, you will be able to 
# MAGIC 
# MAGIC - Explore data with Spark DataFrames 
# MAGIC - Build a pipeline in MLlib for machine learning workflow
# MAGIC - Fit a logistic regression model, make predictions, and evaluate the model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Data Loader

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We are using a dataset from the UCI Machine Learning Repository.
# MAGIC 
# MAGIC 1. Use `wget` to download the dataset. Then use `ls` to verify that the `bank.zip` file is downloaded.

# COMMAND ----------

# MAGIC %%sh
# MAGIC wget https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip

# COMMAND ----------

ls

# COMMAND ----------

# MAGIC %md
# MAGIC 2. Unzip the file and use `ls` to see the files.

# COMMAND ----------

# MAGIC %%sh
# MAGIC unzip bank.zip

# COMMAND ----------

ls

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Exploring The Data
# MAGIC 
# MAGIC We will use the direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict whether the client will subscribe (Yes/No) to a term deposit.
# MAGIC 
# MAGIC 1. Load in the data and look at the columns.

# COMMAND ----------

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('ml-bank').getOrCreate()
df = spark.read.csv('file:/databricks/driver/bank.csv', header=True, inferSchema=True, sep=';')
df.printSchema()


# COMMAND ----------

# MAGIC %md
# MAGIC Here are the columns you should see:
# MAGIC 
# MAGIC * Input variables: age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome
# MAGIC 
# MAGIC * Output variable: y (deposit)
# MAGIC 
# MAGIC 2. Have a peek of the first five observations. Use the `.show()` method.

# COMMAND ----------

# [YOUR CODE HERE]
df.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC 3. To get a prettier result, it can be nice to use Pandas to display the smaller DataFrame. Use the Spark `.take()` method to get the first 5 rows and then convert to a pandas DataFrame. Don't forget to pass along the column names. You should see the same result as above, but in a more aesthetically appealing format.

# COMMAND ----------

import pandas as pd

# [YOUR CODE HERE]
df.take(5)

# COMMAND ----------

# MAGIC %md
# MAGIC 4. How many datapoints are there in the dataset? Use the `.count()` method.

# COMMAND ----------

# [YOUR CODE HERE]
df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC 5. Use the `.describe()` method to see summary statistics on the features.
# MAGIC 
# MAGIC     Note that the result of `.describe()` is a Spark DataFrame, so the contents won't be displayed. It only has 5 rows, so you can just convert the whole thing to a pandas DataFrame with `.toPandas()`.

# COMMAND ----------

# [YOUR CODE HERE]
y= df.describe()
y.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC 6. The above result includes the columns that are categorical, so don't have useful summary statistics. Let's limit to just the numeric features.
# MAGIC 
# MAGIC     `numeric_features` is defined below to contain the column names of the numeric features.
# MAGIC     
# MAGIC     Use the `.select()` method to select only the numeric features from the DataFrame and then get the summary statistics on the resulting DataFrame as we did above.

# COMMAND ----------

numeric_features = [name for name, dtype in df.dtypes if dtype == 'int']
# [YOUR CODE HERE]
y= df.select()
y.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC 7. Run the following code to look at correlation between the numeric features. What do you see?

# COMMAND ----------

numeric_data = df.select(numeric_features).toPandas()
axs = pd.plotting.scatter_matrix(numeric_data, figsize=(8, 8));
n = len(numeric_data.columns)

for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n - 1, i]
    h.xaxis.label.set_rotation(90)
    h.set_xticks(())

# COMMAND ----------

# MAGIC %md
# MAGIC There aren't any highly correlated variables, so we will keep them all for the model. It’s obvious that there aren’t highly correlated numeric variables. Therefore, we will keep all of them for the model. However, day and month columns are not really useful, so we will remove these two columns.
# MAGIC 
# MAGIC 8. Use the `.drop()` method to drop the `month` and `day` columns.
# MAGIC     
# MAGIC     Note that this method returns a new DataFrame, so save that result as `df`.
# MAGIC 
# MAGIC     Use the `.printSchema()` method to verify that `df` now has the correct columns.

# COMMAND ----------

# [YOUR CODE HERE]
df.drop('month','day')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3: Preparing Data for Machine Learning
# MAGIC 
# MAGIC What follows is something analagous to a dataloader pipeline in Tensorflow--we're going to chain together some transformations that will convert our categorical variables into a one-hot format more amenable to training a machine learning model. The next code cell just sets this all up, it doesn't yet run these transformations on our data.

# COMMAND ----------

#%dThe process includes Category Indexing, One-Hot Encoding and VectorAssembler — a feature transformer that merges multiple columns into a #vector column.
#df = df.select('age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'duration', 'campaign', 'pdays', #'previous', 'poutcome', 'deposit')
cols = df.columns
df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC The code is taken from [databricks’ official site](https://docs.databricks.com/applications/machine-learning/train-model/mllib/index.html#binary-classification-example) and it indexes each categorical column using the StringIndexer, then converts the indexed categories into one-hot encoded variables. The resulting output has the binary vectors appended to the end of each row. We use the StringIndexer again to encode our labels to label indices. Next, we use the VectorAssembler to combine all the feature columns into a single vector column.

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Complete the code by completing the assignment of `assembler`. Use `VectorAssembler` and pass in `assemblerInputs` as `inputCols` and name the `outputCol` `"features"`.

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder , StringIndexer, VectorAssembler

categoricalColumns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
stages = []

for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]

label_stringIdx = StringIndexer(inputCol = 'y', outputCol = 'label')
stages += [label_stringIdx]
numericCols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
#assembler = None # [YOUR CODE HERE]
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4: Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC We use Pipeline to chain multiple Transformers and Estimators together to specify our machine learning workflow. A Pipeline’s stages are specified as an ordered array.
# MAGIC 
# MAGIC 1. Fit a pipeline on df.

# COMMAND ----------

#from pyspark.ml import Pipeline
#pipeline = Pipeline(stages=stages)

#pipelineModel = None # [YOUR CODE HERE]


from pyspark.ml import Pipeline
pipeline = Pipeline(stages = stages)
#pipelineModel = pipeline.fit(df)
#df = pipelineModel.transform(df)
selectedCols = ['label', 'features'] + cols
df = df.select(selectedCols)
df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC 2. Transform `pipelineModel` on `df` and assign this to variable `transformed_df`.

# COMMAND ----------

#transformed_df = None # [YOUR CODE HERE]
#transformed_df= pipelineModel.transform(df)
#transformed_df.printSchema()



df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC From the transformation, we'd like to take the `label` and `features` columns as well as the original columns from `df.`
# MAGIC 
# MAGIC 3. Use the `.select()` method to pull these columns from the `transformed_df` and reassign the resulting DataFrame to `df`.

# COMMAND ----------

selectedCols = ['label', 'features'] + df.columns
#df = None # [YOUR CODE HERE]
#df = pipelineModel.transform(df)
df.printSchema()


from pyspark.ml import Pipeline
pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(df)

selectedCols = ['label', 'features'] + cols
df = df.select(selectedCols)

# COMMAND ----------

# MAGIC %md
# MAGIC 4. View the first five rows of the `df` DataFrame. Use either of the methods we did in Part 2:
# MAGIC     * `.show()` method
# MAGIC     * `.take()` method and convert result to a Pandas DataFrame

# COMMAND ----------

# [YOUR CODE HERE]
df.show()
df.take(5)

# COMMAND ----------

# MAGIC %md
# MAGIC 5. Randomly split the dataset in training and test sets, with 70% of the data in the training set and the remaining 30% in the test set.
# MAGIC 
# MAGIC     Hint: Call the `.randomSplit()` method.

# COMMAND ----------

train, test = None, None # [YOUR CODE HERE]

# COMMAND ----------

# MAGIC %md
# MAGIC 6. What are the sizes of the training and test sets?

# COMMAND ----------

# [YOUR CODE HERE]
train, test = df.randomSplit([0.7, 0.3], seed = 2018)
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 5: Logistic Regression Model
# MAGIC 
# MAGIC - You can build a RandomForestClassifier with : from pyspark.ml.classification import RandomForestClassifier
# MAGIC - You can build a Gradient-Boosted Tree Classifier with : from pyspark.ml.classification import GBTClassifier
# MAGIC 
# MAGIC 1. Fit a LogisticRegression with `featuresCol` as `"features"`, `labelCol` as `"label"` and a `maxIter` of 10.

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

# [YOUR CODE HERE]

# COMMAND ----------

# MAGIC %md
# MAGIC 2. We can obtain the coefficients by using LogisticRegressionModel’s attributes. Look at the following plot of the beta coefficients.

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
beta = np.sort(lrModel.coefficients)
plt.plot(beta)
plt.ylabel('Beta Coefficients')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 3. Use the `.transform()` method to make predictions and save them as `predictions`.

# COMMAND ----------

predictions = None # [YOUR CODE HERE]

# COMMAND ----------

# MAGIC %md
# MAGIC 4. View the first 10 rows of the `predictions` DataFrame.

# COMMAND ----------

# [YOUR CODE HERE]

# COMMAND ----------

# MAGIC %md
# MAGIC 5. What is the area under the curve?
# MAGIC 
# MAGIC     You can find it with the `evaluator.evaluate()` function.

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator()
# [YOUR CODE HERE]

# COMMAND ----------

# MAGIC %md
# MAGIC ## OPTIONAL: HyperParameter Tuning a Gradient-Boosted Tree Classifier
# MAGIC 
# MAGIC 1. Fit and make predictions using `GBTClassifier`. The syntax will match what we did above with `LogisticRegression`.

# COMMAND ----------

from pyspark.ml.classification import GBTClassifier

gbt = GBTClassifier(maxIter=10)
gbtModel = gbt.fit(train)
predictions = gbtModel.transform(test)
predictions.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC 2. Run some cross validation to compare different parameters.
# MAGIC 
# MAGIC     Note that it can take a while because it's training over many gradient boosted trees. Give it at least 10 minutes to complete.

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
paramGrid = (ParamGridBuilder()
             .addGrid(gbt.maxDepth, [2, 4, 6])
             .addGrid(gbt.maxBins, [20, 60])
             .addGrid(gbt.maxIter, [10, 20])
             .build())
cv = CrossValidator(estimator=gbt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
cvModel = cv.fit(train)
predictions = cvModel.transform(test)
evaluator.evaluate(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Acknowledgements

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC This notebook is adapted from [Machine Learning with PySpark and MLlib](https://towardsdatascience.com/machine-learning-with-pyspark-and-mllib-solving-a-binary-classification-problem-96396065d2aa)
