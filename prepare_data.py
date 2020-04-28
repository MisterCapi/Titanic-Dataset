import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler
import pickle

def CutFromName(name):
    if "Mr." in name:
        return "Mr"
    elif "Mrs." in name:
        return "Mrs"
    elif "Miss." in name:
        return "Miss"
    else:
        return "unknown status"

def TicketNumber(series):
    for word in series.split():
        if word.isdigit():
            return int(word)
    return 0

def TicketSpecial(series):
    if not series.isdigit():
        return 1
    else:
        return 0

def CabinLetter(cabin):
    try:
        for char in cabin:
            if char.isalpha():
                return char
        return "U"
    except:
        return "U"

def HasCabin(cabin):
    if cabin == 7:
        return 0
    return 1


df = pd.read_csv("train.csv", index_col=0)

embarked_labels = ["C", "N", "Q", "S"]
embarked_lb = LabelBinarizer()
embarked_lb.fit(embarked_labels)
embarked = embarked_lb.transform(df["Embarked"].fillna("N")).T
for embarked_col, label in zip(embarked, embarked_labels):
    df[f"Embarked {label}"] = embarked_col
df.drop("Embarked", axis=1, inplace=True)

df["Name"] = list(map(CutFromName, df["Name"]))
df["Name"].rename("Social Status")
name_labels = ["Miss", "Mr", "Mrs", "unknown status"]
name_lb = LabelBinarizer()
name_lb.fit(name_labels)
name = name_lb.transform(df["Name"]).T
for name_col, label in zip(name, name_labels):
    df[f"{label}"] = name_col
df.drop("Name", axis=1, inplace=True)

df["Age"].fillna(df["Age"].median(), inplace=True)

sex_le = LabelEncoder()
sex_le.fit(df["Sex"].unique())
df["Sex"] = sex_le.transform(df["Sex"])

df["Ticket Number"] = list(map(TicketNumber, df["Ticket"]))
df["Ticket Special"] = list(map(TicketSpecial, df["Ticket"]))
df.drop("Ticket", axis=1, inplace=True)

cabin_dict = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'U':7, 'T':8}
df["Cabin Letter"] = df["Cabin"].map(CabinLetter)
df["Cabin Letter"] = df["Cabin Letter"].map(cabin_dict)
df.drop("Cabin", axis=1, inplace=True)

df["Has Cabin"] = df["Cabin Letter"].map(HasCabin)
print(df)

y = df.iloc[:,:1].values
X = df.iloc[:,1:].values

X = StandardScaler().fit_transform(X)

with open("X.pickle", 'wb') as file:
    pickle.dump(X, file)

with open("y.pickle", 'wb') as file:
    pickle.dump(y, file)
