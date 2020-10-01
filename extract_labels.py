import pandas as pd

df = pd.read_csv('data/TamilChar.csv', header=0)
mystr = "0x0B85"
myint = int(mystr, 16)
print(chr(myint))

print(ord('é'))