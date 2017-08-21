"""This module processes CSV file into list"""
import csv
import os

class csvutil(object):
    """class csv provide staticmethods"""
    @staticmethod
    def importcsv(myfile, mylist):
        """import csv file from local into list"""
        if os.path.exists(myfile):
            try:
                with open(myfile, 'rb') as fin:
                    reader = csv.reader(fin)
                    mylist = list(reader)
            except: #handle other exceptions such as attribute errors
                print("Unexpected error:"), sys.exc_info()[0]
        elif IOError:
            print("I/O error({0}): {1}").format(e.errno, e.strerror)
