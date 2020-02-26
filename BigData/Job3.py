from mrjob.job import MRJob

class Job3(MRJob):

    def mapper(self, _, line):
        #Mapping and removing unnecessary quotes
        try:
            fields = line.split()
            address = fields[0][2:-2]
            block = fields[1][1:-2]
            address = address + ' - ' + block
            value = int(fields[2])
            yield(None, (address, value))
        except:
            pass

    def reducer(self, _, values):
        #Sorting process and yielding
        values = sorted(values, reverse=True, key=lambda l: l[1])
        values = values[:10]
        for value in values:
            address = value[0]
            wei = value[1]
            yield(address, val)

if __name__ == '__main__':
    Job3.run()


