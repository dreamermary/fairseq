

# merge [train.L test.L valid.L] to learn bpe

import os

root = '/content/drive/Shareddrives/mahouli249@gmail.com/dataset/wmt/en_de/phone'
slist = ['phone.train.de','phone.valid.de','phone.test.de']
dfile = 'phone.data.de'

# root = '/content/drive/Shareddrives/mahouli249@gmail.com/dataset/wmt/en_de/phone'
# slist = ['phone.train.en','phone.valid.en','phone.test.en']
# dfile = 'phone.data.en'

# slist = ['train.fr','valid.de','test.de']
# dfile = 'data.fr'



def merge_file(slist,dfile):
    lines = []
    for s in slist:
        with open(os.path.join(root,s)) as fr:
            lines += fr.readlines()
    
    with open(os.path.join(root,dfile),'w') as fw:
        fw.writelines(lines)

merge_file(slist,dfile)
print('ok')

