'''
block_number|from_address|to_address|value|gas|gas_price|block_timestamp
address|is_erc20|is_erc721|block_number|block_timestamp
'''

import pyspark

#Check for valid transactions
def valtran(line):
    try:
        fields = line.split(',')
        if len(fields) != 7:
            return False
        int(fields[3])
        return True
    except:
        return False

#Check for valid contracts
def valcon(line):
    try:
        fields = line.split(',')
        if len(fields) != 5:
            return False
        return True
    except:
        return False

sc = pyspark.SparkContext()

#Aggregating transactions
transactions = sc.textFile("/data/ethereum/transactions")
val_transactions = transactions.filter(valtran)
map_transactions = val_transactions.map(lambda l: (l.split(',')[2], int(l.split(',')[3])))
agg_transactions = map_transactions.reduceByKey(lambda a, b: a+b)

#Getting contracts and keeping the mapped output in memory
contracts = sc.textFile("/data/ethereum/contracts")
val_contracts = contracts.filter(valcon)
map_contracts = val_contracts.map(lambda l: (l.split(',')[0], None))

#Performing join on aggregared transactions and contracts
joined = agg_transactions.join(map_contracts)

#Sorting to get top10
top10 = joined.takeOrdered(10, key=lambda k: -k[1][0])

for address in top10:
    wei = address[1][0]
    eth = wei / 10e+17
    print("{} - {}".format(address[0], eth))

#1' 39"
