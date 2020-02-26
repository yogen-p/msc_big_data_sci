'''
block_number|        from_address|          to_address|              value|   gas|  gas_price|block_timestamp
'''

from mrjob.job import MRJob

class Job1(MRJob):

    def mapper(self, _, line):
        try:
            #Mapping the address
            #with its corresponding transaction
            fields = line.split(',')
            address = fields[2]
            value = int(fields[3])
            #Filtering out 0s to speed up
            if value == 0:
                pass
            else:
                yield(address, value)
        except:
            pass

    def combiner(self, address, value):
        #Adding up the value of transaction
        yield(address, sum(value))

    def reducer(self, address, value):
        #Adding up the value of transaction
        yield(address, sum(value))

if __name__ == '__main__':
    Job1.run()

#26' 29"
