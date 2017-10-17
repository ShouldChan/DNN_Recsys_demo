# step 1 -----------------count nums of per user rated movies-------
'''
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
'''
# step 2------------------count nums of per movies rated-----
'''
mid_set = set()
m_count_dict = {}
with open('./ratings.txt','rb') as fread:
    lines = fread.readlines()
    for i in range(len(lines)):
        temp = lines[i].strip().split('\t')
        uid, mid, rate, ts = temp[0],temp[1],temp[2],temp[3]
        # if mid shows at the first time, initialize
        if mid not in mid_set:
            m_count_dict[mid] = 1
            mid_set.add(mid)
        else:
            m_count_dict[mid] += 1

with open('./movie_per_count.txt','wb') as fwrite:
    for key,value in m_count_dict.iteritems():
        fwrite.write(str(key)+'\t'+str(value)+'\n')

'''
# step 3-------use count_file to filter the rating history------
# users who per rated movies <= 20 is out
uid_not_need = []
with open('./user_rate_count.txt','rb') as fread:
    lines = fread.readlines()
    for line in lines:
        uid, u_count = line.strip().split('\t')[0],int(line.strip().split('\t')[1])
        if u_count <= 20:
            uid_not_need.append(uid)

print 'we kick out %d users'%len(uid_not_need)
n_users = int(2113-len(uid_not_need))
print 'valid user: %d'%(n_users)

# movies that per rated <=30
mid_not_need = []
with open('./movie_per_count.txt','rb') as fread:
    lines = fread.readlines()
    for line in lines:
        mid, m_count = line.strip().split('\t')[0],int(line.strip().split('\t')[1])
        if m_count <= 10:
            mid_not_need.append(mid)
print 'we kick out %d movies'%len(mid_not_need)
n_movies = int(7258-len(mid_not_need))
print 'valid movie: %d'%(n_movies)

# fwrite = open('./filter_ratings.txt','wb')
count = 0
with open('./filter_ratings.txt','wb') as fwrite:
	with open('./ratings.txt','rb') as fread:
	    lines = fread.readlines()
	    for i in range(len(lines)):
	        temp = lines[i].strip().split('\t')
	        uid, mid, rate, ts = temp[0],temp[1],temp[2],temp[3]
	        if (uid not in uid_not_need) and (mid not in mid_not_need):
	        	count += 1
	        	fwrite.write(str(lines[i]))
# fwrite.close()
sparsity = round(1.0 - count / float(n_users * n_movies), 3)
spar_2_str = str(sparsity * 100) + '%'
print sparsity 