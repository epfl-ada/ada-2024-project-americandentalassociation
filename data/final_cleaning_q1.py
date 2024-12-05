import pandas as pd

data = pd.read_csv("data/wars_filtered_clean-2.csv")

result = data.groupby(['WarName', 'Side']).agg(
    StartYear=('StartYear', 'min'),
    EndYear=('EndYear', 'max'),
    StateName=('StateName', list),
    BatDeath = ('BatDeath', 'sum')
)

print(result.head())
print(result.shape)
result.to_csv("data/data_Q1.csv")