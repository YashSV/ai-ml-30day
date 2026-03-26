import numpy as np
import pandas as pd

s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s)

dates = pd.date_range("20130101", periods=6)
print(dates)

df= pd.DataFrame(np.random.randn(6,4),index=dates,columns=list("ABCD"))
print(df)

df2 = pd.DataFrame("ABCDEFGHIJ", index=dates, columns=list("AB"))
print(df2)

df2 = pd.DataFrame({"A": 1.,
                    "B": pd.Timestamp("20130102"),
                    "C": pd.Series(1, index=list(range(4)), dtype="float32"),
                    "D": np.array([3] * 4, dtype="int32"),
                    "E": pd.Categorical(["test", "train", "test", "train"]), 
                    "F": "foo",
    }
)
print(df2)

print(df.head())
print(df.tail(3))
print(df.columns)
print (df)
print(df.to_numpy())
print(df.describe())

print(df.T)
print("Sort_idex by columns in descending order and ascending order")
print(df)
print(df.sort_index(axis=1, ascending=False))

print("Get the individual columns of the DataFrame")
print(df["A"])

print(df.loc[dates[0]])

print(df)

print(df.iloc[1:3, 0:2])
print(df.iloc[1:3, :])
print(df.iloc[:, 0:2])

print(pd.Series(
   [1, 2, 3, 4, 5, 6],
   index=pd.date_range("20130102", periods=6))
)

print("Operations  on the DataFrame")
print(df)
print(df.mean())

s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)
print(s)
print(df.sub(s, axis="index"))

print("User defined functions")

print(df)
print(df.agg(lambda x: np.mean(x) * 5.6))

s = pd.Series(np.random.randint(5, 8, size=10))

print(s)
print(s.value_counts())

s = pd.Series(["A", "B", "C", "Aaba", "Baca", np.nan, "CABA", "dog", "cat"])
print(s.str.lower())


df = pd.DataFrame(np.random.randint(0, 5, (10, 5)))
df.to_csv("foo.csv", index=False)
print(df)
pd.read_csv("foo.csv")