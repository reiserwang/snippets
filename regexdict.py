import re


class RegExDict(object):
    """
    https://djangosnippets.org/snippets/309/
    
    A dictionary-like object for use with regular expression keys.
    Setting a key will map all strings matching a certain regex to the
    set value.
    
    One caveat: the order of the iteration over items is unspecified,
    thus if a lookup matches multiple keys, it is unspecified which
    value will be returned - still, one such value will be returned.
    
    >>> d = RegExDict()
    >>> d[r'moo.*haha'] = 7
    >>> d[r'holler.*fool'] = 2
    >>> d['mooWORDhaha']
    7
    """
    def __init__(self):
        self._regexes = {}
        
    def __getitem__(self, name):
        for regex, value in self._regexes.items():
            m = regex.match(name)
            if m is not None:
                return value
        raise KeyError('Key does not match any regex')
    
    def __setitem__(self, regex, value):
        self._regexes[re.compile(regex)] = value