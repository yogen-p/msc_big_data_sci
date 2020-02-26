'''
block_number|        from_address|          to_address|              value|   gas|  gas_price|block_timestamp
'''


import time
from mrjob.job import MRJob

class PartOne(MRJob):

    def mapper(self, _, line):
        fields = line.split(',')
        try:
            #Mapping month and year with 1 transaction
            epoch = time.gmtime(int(fields[6]))
            year = epoch.tm_year
            month = epoch.tm_mon
            yield((year, month), 1)
        except:
            pass

    def combiner(self, date, count):
        #Aggregating all the transactions for one month
        yield(date, sum(count))

    def reducer(self, date, count):
        #Aggregating all the transactions for one month
        yield(date, sum(count))

if __name__ == '__main__':
    #Configuring 1 reducer to get one sorted output
    PartOne.JOBCONF= { 'mapreduce.job.reduces': '1' }
    PartOne.run()
