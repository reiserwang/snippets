import csv
import os

'''

'''
class csv(object):
    """description of class"""
    @staticmethod
    def importcsv(myfile,mylist):
        if os.path.exists(myfile):
            try:
                with open(myfile,'rb') as f:
                    reader = csv.reader(f)
                    mylist=list(reader)
            except: #handle other exceptions such as attribute errors
                print ("Unexpected error:"), sys.exc_info()[0]
        elif IOError:
            print ("I/O error({0}): {1}").format(e.errno, e.strerror)
