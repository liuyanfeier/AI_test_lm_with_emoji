# -*- coding: utf-8 -*-  
  
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import word2vec 
import logging  
  
# 主程序  
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)  
sentences = word2vec.Text8Corpus("data")  # 加载语料 

model = word2vec.Word2Vec(sentences, size=512, min_count=5, window=5)  # 训练skip-gram模型 
  
model.save("my_word2vec_512.model")  
  
vocab = model.wv.vocab.keys()

vocab_len = len(vocab)

print(vocab_len)

with open("my_vocab_512", 'w') as f:
    for i in range(vocab_len):
        f.write(list(vocab)[i] + '\n') 

num = 0
with open("my_vocab_512", "r") as in_text:
    with open("word2feat_512", "w") as out_text:
        for line in in_text:
            words = line.split()
            for word in words:
                if word in model.wv.vocab:
                    #if word in  m_vocab:
                    out_text.write(word + " ");
                    for i in range(512):
                        out_text.write(str(model[word][i]) + " ")
                    out_text.write("\n")
                    #else :
                        #y2 = model.most_similar("good", topn=20)  # 20个最相关的
                        #print (u"和good最相关的词有：\n")
                        #for item in y2:
                            #print (item[0], item[1])
                else:
                    num = num + 1
                    #print ("Wrong!\n")

print ("num: ",num) 
