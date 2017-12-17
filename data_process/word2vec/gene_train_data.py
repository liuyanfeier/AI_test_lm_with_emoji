# coding: utf-8
import os
import codecs
import re
import sys

with codecs.open("data", 'w', 'utf-8') as out_text:
    with codecs.open(sys.argv[1], 'r', 'utf-8') as text:
        for line in text:
            line=line.strip('\n')
            line = ' '.join(line.split())
            out_text.write(line + ' ')
        out_text.write("\n")




