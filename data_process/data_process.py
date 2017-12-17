#coding: utf-8
from __future__ import division

import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize

import os
import codecs
import re
import collections
import sys
import emoji
import string

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def has_numbers(inputString):
    return bool(re.search(r'\d', inputString))

def untokenize(line):
    return line.replace("`", "'").replace("‘", "'").replace("’","'").replace("“","\"").replace("”","\"").replace("^", "")


def process_line(line):

    tokens = word_tokenize(line)
    output_tokens = []

    for token in tokens:
        # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
        if token in string.punctuation or token == "...":
            continue       #去除punc
            #output_tokens.append("<punc>")
        elif has_numbers(token) or is_number(token):
            #print(token)
            output_tokens.append("<num>")
        else:
            #token = token.lower()
            output_tokens.append(token)

    return untokenize(" ".join(output_tokens) + " ")

def clean_raw_text():
    ignore_emoji = {}
    with open("ignore_emoji", "r") as vocab:
        for line in vocab:
            words = line.split()
            for word in words:
                ignore_emoji.update({word:1})

    with codecs.open("text_with_emoji", 'w', 'utf-8') as out_txt:
        with codecs.open(sys.argv[1], 'r', 'utf-8') as text:

            for line in text:
                line = line.strip()

                skip = False
                for i in range(len(line)):
                    if line[i] not in emoji.UNICODE_EMOJI and ord(line[i]) > 128:
                        #print(line[i], ord(line[i]))
                        skip = True
                if skip == True:
                    print(line)
                    continue
                raw_line=line
                line = process_line(line)
                line = line.replace(" n't", "n't").replace(" '", "'")
                token=''
                for i in range(len(line)):
                    #print(line[i])
                    if line[i] in ignore_emoji:      #5 ignore emoji
                        print(line[i])
                    elif line[i] in emoji.UNICODE_EMOJI:
                        token=token+'   '+line[i]+'   '
                    elif line[i] == '\.':
                        print(line[i])
                    else:
                        token=token+line[i]
                token = ' '.join(token.split())  #去除多余的空格

                words = token.split()
                if len(words) < 2:
                    skip = True
                if skip == True:
                    print(line)
                    continue
                out_txt.write(token+'\n')

def gene_train_data():
    text_vocab = {}
    with open("my_vocab.txt", "r") as vocab:
        for line in vocab:
            words = line.split()
            for word in words:
                text_vocab.update({word:1})

    with codecs.open("text_with_emoji", "r", encoding="utf-8") as f:
        data_1 = f.read()

    x_text = data_1.split()
    word_counts = collections.Counter(x_text)
    input_vocab = [x[0] for x in word_counts.most_common(20000) if x[1] > 10 and '.' not in x[0]]
    #input_vocab = [x[0] for x in word_counts.most_common(2000)]
    print("len(input_vocab): ", len(input_vocab))
    for word in input_vocab:
        if word in emoji.UNICODE_EMOJI:
            continue
        if '*' in word or '.' in word or '\\' in word or '/' in word or '-' in word or '=' in word or ',' in word or '|' in word or word[0] == '\'' or word[len(word)-1] == '\'' or word[0] == '.' or word[len(word)-1] == '.':
            input_vocab.remove(word)
            print(word)
            continue
        for i in range(len(word)):
            if word[i] != '\'':
                if ord(word[i]) > 122 or ord(word[i]) < 40:
                    input_vocab.remove(word)
                    print(word)
                    break
    input_vocab.append("<num>")
    print("len(input_vocab): ", len(input_vocab))

    with open("text_with_emoji", "r") as text:
        with open("input_data", "w") as in_text:
            with open("output_data", "w") as out_text:
                for line in text:
                    line = line.strip()
                    words = line.split()
                    out_line=""
                    in_line=""
                    unk_num = 0
                    word_num = 0
                    for word in words:
                        if '*' in word or '\.' in word or '\\' in word or '\/' in word or '-' in word or '=' in word or ',' in word or '|' in word or word[0] == '\'' or word[len(word)-1] == '\'' or word[0] == '.' or word[len(word)-1] == '.':
                            unk_num += 1
                            out_line=out_line+"<unk>"+" "
                            in_line=in_line+"<unk>"+" " 
                        elif word not in input_vocab:
                            unk_num += 1
                            out_line=out_line+"<unk>"+" "
                            if word not in emoji.UNICODE_EMOJI:
                                in_line=in_line+"<unk>"+" "
                            else:
                                in_line=in_line+"<emoji>"+" "
                        else:
                            in_line=in_line+word+" "
                            if word == "<num>":
                                unk_num += 1
                                out_line=out_line+"<unk>"+" "
                            elif text_vocab.get(word.lower()) == None and word not in emoji.UNICODE_EMOJI:
                                unk_num += 1
                                out_line=out_line+"<unk>"+" "
                            else:
                                out_line=out_line+word+" "          #输出需要是合法的单词
                                word_num += 1
                    if unk_num != 0:
                        if word_num/unk_num < 4 or len(words) > 128:
                            continue
                    in_text.write(in_line+"\n")
                    out_text.write(out_line+"\n")
     
    
def do_analyze():
    input_text_vocab = {}
    with codecs.open("input_data", "r", encoding="utf-8") as f:
        input_data = f.read()
    input_text = input_data.split()
    word_counts = collections.Counter(input_text)
    with open("input_vocab", "w") as text:
        for x in word_counts.most_common():
            if '\.' not in x[0] and x[1] > 4:
                text.write(x[0] + " " + str(x[1]) +"\n")
                input_text_vocab.update({x[0]:1})
    
    output_text_vocab = {}
    with codecs.open("output_data", "r", encoding="utf-8") as f:
        output_data = f.read()
    output_text = output_data.split()
    word_counts = collections.Counter(output_text)
    with open("output_vocab", "w") as text:
        for x in word_counts.most_common():
            if '\.' not in x[0] and x[1] > 10:
                text.write(x[0] + " " + str(x[1]) +"\n")
                output_text_vocab.update({x[0]:1})
    
    return input_text_vocab, output_text_vocab
   
def gene_train_data_again(input_vocab, output_vocab):
    with open("input_data", "r") as in_text:
        with open("input_data_again", "w") as out_text:
            for line in in_text:
                line = line.strip()
                words = line.split()
                text = ""
                for word in words:
                    if input_vocab.get(word) == None:
                        text = text+"<unk>" + " "
                    else:
                        text = text+word + " "
                out_text.write(text+"\n")
 
    with open("output_data", "r") as in_text:
        with open("output_data_again", "w") as out_text:
            for line in in_text:
                line = line.strip()
                words = line.split()
                text = ""
                for word in words:
                    if output_vocab.get(word) == None:
                        text = text+"<unk>" + " "
                    else:
                        text = text+word + " "
                out_text.write(text+"\n")
 
    word_num = 0
    line_num = 0
    emoji_num = 0
    unk_num = 0
    emoji_line = 0
    with open("output_data", "r") as text:
        for line in text:
            line = line.strip()
            words = line.split()
            flag = False
            line_num += 1
            for word in words:
                word_num += 1
                if word == "<unk>":
                    unk_num += 1
                if word in emoji.UNICODE_EMOJI:
                    emoji_num += 1
                    if flag == False:
                        emoji_line += 1
                        flag = True

    print("word_num: ", word_num)
    print("line_num: ", line_num)
    print("emoji_num: ", emoji_num)
    print("emoji_line: ", emoji_line)
    print("unk_num: ", unk_num)

if __name__=="__main__":
    if len(sys.argv) != 2:
        print("Usage: python data_process.py filename")
        exit(0)

    print("Cleaning Text!")
    clean_raw_text()
    print("generate training data!")
    gene_train_data()
    print("do analyze!")
    input_vocab, output_vocab = do_analyze()
    print("generate training data again!")
    gene_train_data_again(input_vocab, output_vocab)
    print("Done!")
