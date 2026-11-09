import matplotlib.pyplot as plt
import pandas as pd
pd.options.display.width = 0

df = pd.read_csv("../data/spiral.csv", sep=";", index_col="iter")
df.iloc[0] = df.iloc[0]*2
print(df)
# df = df.iloc[:200]
plt.plot(df['real'], df['imag'], marker='o', c='b')
plt.plot(df['acc_real'], df['acc_imag'], marker='o', c='r')
for iter, row in df.iterrows():
    plt.annotate(iter, (row['real'], row['imag']))
    plt.annotate(iter, (row['acc_real'], row['acc_imag']))

plt.savefig("../data/spiral.png", dpi=300)
