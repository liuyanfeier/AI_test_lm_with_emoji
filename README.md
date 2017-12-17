# AI_test_lm_with_emoji

训练了一个带emoji的语言模型，可以在整袋预测单词的境况下预测可能出现的emoji。

data_process文件是对原始语料进行清洗以及word2vec提取特征。

主程序使用方法如下：
python3 train.py --input_data data/input --output_data output --save_path save/emoji_512

各项参数可以进行更改，主程序根据官方ptb例子修改而来。

最终结果为：
Top1：Emoji_recall:  0.48 Wrong:  0.81
Top3：Emoji_recall:  0.57 Wrong:  0.69

