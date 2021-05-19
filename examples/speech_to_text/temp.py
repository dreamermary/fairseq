import re
EN_STOP_CHAR = '[\.。,，、/\?？:：“”\"‘’\'\{\}\[\]|\\<>\|《》~`·！!@#\$%\^&\*\(\)￥……（）—=_-【】]'

EN_STOP_CHAR="[+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）:;：；‘’“”《》<>\?]"
ZH_STOP_CHAR="[+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）:;：；‘’“”《》<>ZXCVBNMASDFGHJKLQWERTYUIOPzxcvbnmasdfghjklqwertyuiop\?]"
src = "This boy is a nuisance, you might say."
trg = "我吃lL，饭了?。"
src = re.sub(EN_STOP_CHAR, "", src)
trg = re.sub(ZH_STOP_CHAR, "", trg)

from string import punctuation
punc = punctuation + u'.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：'

print(src)
print(trg)
111