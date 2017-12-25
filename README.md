# AI_test_lm_with_emoji

训练了一个带emoji的语言模型，可以在预测单词的同时下预测出可能出现的emoji。

data_process文件是对原始语料进行清洗以及word2vec提取特征等。

主程序使用方法如下：
python3 train.py --input_data data/input --output_data output --save_path save/emoji_512

各项参数可以进行更改，主程序根据官方ptb例子修改而来。

测试方法为：
python3 test.py --input_data input/ --save_path save

训练好的word2vec和lm模型放在 https://drive.google.com/drive/folders/1paCCqa9VKYUH7xrhnd36jZxqzP3yxo1T?usp=sharing 上。

把其中的lm模型放进save目录，将word2vec模型放进input目录
将测试的文本命名为test.txt放进input目录
最终input目录中有文件input_vocab    output_vocab    word2feat_512    test.txt 
save目录中的文件为checkpoint    model.ckpt.data-00000-of-00001    model.ckpt.index

最终结果为：

Top1：Emoji_recall:  0.48 Wrong:  0.81

Top3：Emoji_recall:  0.57 Wrong:  0.69

