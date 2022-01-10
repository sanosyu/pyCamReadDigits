import pandas as pd
import matplotlib.pyplot as plt


input_csv = pd.read_csv('./test.csv')
x = input_csv[input_csv.keys()[0]]
y = input_csv[input_csv.keys()[3]]

plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.plot(y)
plt.show()
