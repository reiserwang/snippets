from datetime import datetime
import logging
'''
if __debug__:
    logger = logging.getLogger("timeproc")
    logger.setLevel(logging.DEBUG)
    logger.debug("Debug on")
'''
def string_to_time(s, timeformat):
    """
    @param s the string to parse
    @param format the format to attempt parsing of the given string
    @return the parsed datetime or None on failure to parse 
    @see datetime.datetime.strptime
    """
    try:
        datetime_obj = datetime.strptime(s, timeformat)
    except ValueError:
        datetime_obj = None
    return datetime_obj

if __name__ == "__main__":
    timestr = None
    while True:
        try:
            timestr =raw_input()
            if not timestr.strip(): 
                timestr = str(datetime.now())
                print timestr
            print string_to_time(timestr, "%y-%m-%d")
        except Exception:
            '''logger.debug("Exception occured.")'''
            print("Exception occurred.")
            pass
