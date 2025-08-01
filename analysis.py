import pandas as pd
import joblib 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#loading the trained model
model = joblib.load("model/model.joblib")

#importing the dataset
df = pd.read_csv("dataset/final_sample.csv")

#creating the feature vector
features = df.drop(["treatment", "exposure", "visit", "conversion", "Unnamed: 0"], axis = 1)

#predicting the ITE
ite = model.predict(X = features)

#creating the final_df with consumers and their predicted ite
final_df = pd.DataFrame({
    "consumers" : [i for i in range(len(ite))],
    "ite" : ite
})

#selecting only the consumers who have ite > 0.05 (5%)
print(final_df[final_df["ite"] > 0.05])

#printing the number of consumers who have ite > 0.05
print(len(final_df[final_df["ite"] > 0.05]))

#plotting the ite values
plt.figure(figsize=(12,10))
plt.scatter([i for i in range(len(ite))], ite)
plt.xlabel("ITE")
plt.ylabel("Consumers")
plt.title("ITE values")
plt.show()