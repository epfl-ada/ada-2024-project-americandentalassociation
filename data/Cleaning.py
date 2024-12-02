import pandas as pd

dataframe = pd.read_csv("data/wars.csv")
data_specific = dataframe.drop(columns = ["TransFrom", "TransTo", "WhereFought", "WarNum", "WarType", "ccode", "StartMonth1", "StartDay1", "StartYear2", "EndMonth1", "EndDay1", "StartMonth2", "StartDay2", "EndMonth2", "EndDay2", "EndYear2", "Version", "Initiator"])
data_specific = data_specific.rename(columns={"StartYear1": "StartYear", "EndYear1": "EndYear"})
print(data_specific.head())

result = data_specific.groupby(['WarName', 'Side']).agg(
    StartYear=('StartYear', 'min'),
    EndYear=('EndYear', 'max'),
    StateName=('StateName', list),
    BatDeath = ('BatDeath', 'sum')
).reset_index()

print(result.head())
print(result.shape)