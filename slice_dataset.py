# slice dataset
max_users = 264

data_slice = []

count = 1
fwrite = open('./slice_dataset.txt','wb')
with open('./v2_filter_user_ratedmovies.txt','rb') as fread:

    while count < max_users:
        line = fread.readline()
        temp = line.strip().split('\t')
        uid, mid, rate = temp[0],temp[1],temp[2]

        line_nx = fread.readline()
        temp_nx = line_nx.strip().split('\t')
        uid_nx, mid_nx, rate_nx = temp_nx[0],temp_nx[1],temp_nx[2]

        fwrite.write(str(uid)+'\t'+str(mid)+'\t'+str(rate)+'\n')
        if uid != uid_nx:
            count += 1
        print count

# fwrite=open('./moviejpgid_imdbid.txt','wb')
# with open('./valid_movieid_imdbid.txt','rb') as fread:
#     lines = fread.readlines()
#     for line in lines:
#         temp = line.strip().split('\t')
#         jpgid, imdbid = temp[0],temp[1]
#         fwrite.write(str(jpgid)+'.jpg\t'+str(imdbid)+'\n')
# fwrite.close()