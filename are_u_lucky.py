# make a test if u are lucky today
# if u get an odd number, it means a lucky day. Otherwise, even number.

import random
import time

fwrite = open('./are_u_lucky','a+')
localtime = time.localtime(time.time())
num = random.randint(1,50000)
print(num)
_localtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
print('localtime:\t',_localtime)

fwrite.write(str(num)+'\t'+str(_localtime)+'\n')