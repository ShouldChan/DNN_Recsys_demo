# step 1 -----------------count nums of per user rated movies-------
uid_set = set()
u_count_dict = {}
u_count = 1
with open('./ratings.txt','rb') as fread:
	lines = fread.readlines()
	for i in range(len(lines)):
		temp = lines[i].strip().split('\t')
		uid, mid, rate, ts = temp[0],temp[1],temp[2],temp[3]
		if len(uid_set) == 0:
			j = i+1
			while lines[j].strip().split('\t')[0] == uid:
				u_count += 1
				j += 1
			i = u_count + 1
		if uid not in uid_set:
			u_count_dict[uid] = u_count
			uid_set.add(uid)
			u_count = 1
		else:
			u_count += 1

with open('./user_rate_count.txt','wb') as fwrite:
	for key,value in u_count_dict.iteritems():
		fwrite.write(str(key)+'\t'+str(value)+'\n')
