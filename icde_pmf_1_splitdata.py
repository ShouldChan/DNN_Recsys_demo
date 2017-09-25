# Step 1
# split train and test
user_count = {}
ucount = 0
line_mark = set()
user_set = set()

with open('./Foursquare_final.txt','rb') as fread:
	lines = fread.readlines()
	for i in range(len(lines)):
		temp = lines[i].strip().split('\t')
		uid,pid,time,lat,lon = int(temp[0]),temp[1],temp[2],temp[3],temp[4]
		user_set.add(uid)
		if i != len(lines)-1:
			temp_nx = lines[i+1].strip().split('\t')
			uid_nx,pid_nx,time_nx,lat_nx,lon_nx \
			= int(temp_nx[0]),temp_nx[1],temp_nx[2],temp_nx[3],temp_nx[4]
		else:
			break
		if uid == uid_nx:
			ucount += 1
		else:
			user_count[uid] = ucount
			ucount = 0
	user_count[761] = 16
	print 'step 1\tcount user checkins over...'
# print len(user_set) #762 users


# Step 2
# best codes in 2017-09-24
	N = int(0.7*user_count[0])
	M = int(user_count[0])
	j = int(0)
	for i in range(len(lines)):
		temp = lines[i].strip().split('\t')
		uid,pid,time,lat,lon = int(temp[0]),temp[1],temp[2],temp[3],temp[4]
		if i != len(lines)-1:
			temp_nx = lines[i+1].strip().split('\t')
			uid_nx,pid_nx,time_nx,lat_nx,lon_nx \
			= int(temp_nx[0]),temp_nx[1],temp_nx[2],temp_nx[3],temp_nx[4]
		else:
			break
		if i >= N and i <= M:
			line_mark.add(i)
		if uid != uid_nx:
			j += 1
			N = int(i+1+0.7*user_count[j])
			M = int(i+1+user_count[j])
	print 'step 2\tset test over...'

	ftrain = open('./foursquare_train.txt','wb')
	ftest = open('./foursquare_test.txt','wb')
	for i in range(len(lines)):
		# temp = lines[i].strip().split('\t')
		# uid,pid,time,lat,lon = int(temp[0]),temp[1],temp[2],temp[3],temp[4]
		if i not in line_mark:
			ftrain.write(lines[i])
		else:
			ftest.write(lines[i])
	ftrain.close()
	ftest.close()
	print 'step 3\tsplit train test over...'