from mrjob.job import MRJob

class Job2(MRJob):
    def mapper(self, _, line):
        #Differentiating between both the files
        try:
            if len(line.split(','))==5:
                fields = line.split(',')
                join_key1 = fields[0]
                join_value = fields[3]
                yield (join_key1, (join_value, 1))

            if len(line.split('\t'))==2:
                fields = line.split('\t')
                join_key2 = fields[0]
                #Removing unnecessary double quotes
                join_key2 = join_key2[1:-1]
                join_value = int(fields[1])
                yield (join_key2, (join_value, 2))
        except:
            pass

    def reducer(self, address, values):
        block = None
        amt = None
        for value in values:
            if value[1] == 1:
                block = value[0]
            if value[1] == 2:
                amt = value[0]
        #Validating the values before yielding
        if block is not None and amt is not None:
            yield((address, block), amt)


if __name__ == '__main__':
    Job2.run()

#7' 35"
