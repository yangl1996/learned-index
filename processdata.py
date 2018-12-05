datas = []
poss = []
with open("random.csv") as f:
    for line in f:
        dp = line.rstrip().split(',')
        datas.append(int(dp[0]))
        poss.append(int(dp[1]))

ndata = len(datas)
print ndata
for i in range(ndata):
    print datas[i]
    print poss[i]



"""
Format
num of datapoints: int

data: int
location: int
"""

