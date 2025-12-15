import numpy as np
import torch
import Constants
import pickle
import os


class Options(object):

    def __init__(self, data_name='poli'):
        #文件路径初始化（构造了多个文件路径，指向不同的数据文件，包括新闻数据、用户数据、标签和新闻列表等）
        self.nretweet = 'data/' + data_name + '/news_centered_data.txt'
        self.uretweet = 'data/' + data_name + '/user_centered_data.txt'
        self.label = 'data/' + data_name + '/label.txt'
        self.news_list = 'data/' + data_name + '/' + data_name + '_news_list.txt'

        self.news_centered = 'data/' + data_name + '/Processed/news_centered.pickle'
        self.user_centered = 'data/' + data_name + '/Processed/user_centered.pickle'
        #加载索引（使用 NumPy 加载训练、验证和测试的索引文件，并将其转换为 PyTorch 的张量格式。）
        self.train_idx = torch.from_numpy(np.load('data/' + data_name +'/train_idx.npy'))
        self.valid_idx = torch.from_numpy(np.load('data/' + data_name +'/val_idx.npy'))
        self.test_idx = torch.from_numpy(np.load('data/' + data_name +'/test_idx.npy'))
        #处理后的数据文件路径（定义了处理后的训练、验证和测试数据的文件路径）
        self.train = 'data/' + data_name + '/Processed/train_processed.pickle'
        self.valid = 'data/' + data_name + '/Processed/valid_processed.pickle'
        self.test = 'data/' + data_name + '/Processed/test_processed.pickle'
        #用户和新闻映射文件路径
        self.user_mapping = 'data/' + data_name + '/user_mapping.pickle'
        self.news_mapping = 'data/' + data_name + '/news_mapping.pickle'
        self.save_path = ''#初始化了保存路径save_path（默认为空）和嵌入维度embed_dim（设置为64）
        self.embed_dim = 64


def buildIndex(user_set, news_set):
    n2idx = {}#用于存储新闻到索引的映射
    u2idx = {}#用于存储用户到索引的映射

    pos = 0#用于跟踪当前索引的位置
    u2idx['<blank>'] = pos
    pos += 1#首先，将一个占位符添加到用户字典中，索引为 0
    for user in user_set:#遍历user_set中的每个用户，将用户和其对应的索引添加到u2idx字典中，索引值依次递增
        u2idx[user] = pos
        pos += 1

    pos = 0
    n2idx['<blank>'] = pos
    pos += 1
    for news in news_set:
        n2idx[news] = pos
        pos += 1

    user_size = len(user_set)#计算用户和新闻的数量
    news_size = len(news_set)
    return user_size, news_size, u2idx, n2idx#返回用户集合和新闻集合的大小，以及对应的索引字典u2idx和n2idx

def Pre_data(data_name, early_type, early, max_len=200):#用于预处理推文的传播数据
    options = Options(data_name)
    cascades = {}#这个字典将用于存储推文的传播信息

    '''load news-centered retweet data'''#加载以新闻为中心的转推数据
    for line in open(options.nretweet):#逐行打开并读取新闻中心的转发数据文件
        userlist = []
        timestamps = []
        levels = []
        infs = []

        chunks = line.strip().split(',')#将当前行按逗号分割，并将第一个元素作为推文的标识符，初始化cascades字典中的对应值。
        cascades[chunks[0]] = []

        for chunk in chunks[1:]:#遍历每个转发记录，尝试解析用户、时间戳、级别和信息。如果解析失败，则将其视为根推文，并将默认值添加到相应的列表中
            try:
                user, timestamp, level, inf = chunk.split()
                userlist.append(user)
                timestamps.append(float(timestamp)/3600/24)
                levels.append(int(level)+1)
                infs.append(inf)
            except:
                user = chunk
                userlist.append(user)
                timestamps.append(float(0.0))
                infs.append(1)
                levels.append(1)
                print('tweet root', chunk)
        cascades[chunks[0]] = [userlist, timestamps, levels, infs]#将解析后的数据列表存储到cascades字典中

    news_list = []
    for line in open(options.news_list):
            news_list.append(line.strip())#读取新闻列表文件，将每一行添加到news_list列表中
    cascades = {key: value for key, value in cascades.items() if key in news_list}#只保留在news_list中存在的推文

    if early:#处理早期决策
        if early_type == 'engage':
            max_len = early#如果early_type为'engage'，则将max_len设置为early的值
        elif early_type == 'time':
            mint = []#如果early_type为'time'，则计算时间戳的最小和最大值，确定满足条件的传播记录。
            for v in cascades.values():
                times = v[1]
                if max(times)-min(times) < early:
                    mint.append(len(times))
                else:
                    for t in times:
                        if t - min(times) >= early:
                            mint.append(times.index(t))
                            break


    '''ordered by timestamps'''#按时间戳排序
    for idx, cas in enumerate(cascades.keys()):#使用enumerate函数遍历cascades字典的键（即每个推文的标识符），并获取每个键的索引idx和键名cas
        max_ = mint[idx] if early and early_type == 'time' and mint[idx] < max_len else max_len#据early和early_type的值来决定max_的值。如果early为真且early_type为'time'，并且mint[idx]小于max_len，则将max_设置为mint[idx]；否则，将max_设置为max_len 。
        cascades[cas] = [i[:max_] for i in cascades[cas]]#对cascades[cas]中的每个列表进行切片，只保留前max_个元素。

        order = [i[0] for i in sorted(enumerate(cascades[cas][1]), key=lambda x: float(x[1]))]
        #首先使用enumerate函数为时间戳列表cascades[cas][1]中的每个时间戳生成索引
        #使用sorted函数根据时间戳的值对这些索引进行排序。key=lambda x: float(x[1])指定了排序的依据是时间戳的值
        #最终order列表包含了按时间戳排序后的索引

        #print(cascades[cas].shape)
        cascades[cas] = [[x[i] for i in order] for x in cascades[cas]]#根据之前计算出的order列表对cascades[cas]中的每个列表进行重新排序。这样，用户列表、时间戳、级别和信息都将按照时间戳的顺序排列。
        #cascades[cas] = cascades[cas][:,order]
        #cascades[cas][1][:] = [cascades[cas][1][i] for i in order]
        #cascades[cas][0][:] = [cascades[cas][0][i] for i in order]
        #cascades[cas][2][:] = [cascades[cas][2][i] for i in order]
        #cascades[cas][3][:] = [cascades[cas][3][i] for i in order]

    ucascades = {}
    '''load user-centered retweet data'''
    for line in open(options.uretweet):#逐行打开并读取指定的用户转发数据文件
        newslist = []
        userinf = []

        chunks = line.strip().split(',')#对当前行进行处理，首先去掉行首尾的空白字符，然后按逗号分割成多个部分，存储在chunks列表中

        ucascades[chunks[0]] = []#将chunks列表的第一个元素作为键，初始化ucascades字典中的对应值为一个空列表

        for chunk in chunks[1:]:#从chunks列表的第二个元素开始遍历
            news, timestamp, inf= chunk.split()#将当前的chunk按空格分割成三个部分
            newslist.append(news)#将提取到的news和inf分别添加到newslist和userinf列表中
            userinf.append(inf)

        ucascades[chunks[0]] = np.array([newslist, userinf])#将chunks[0]作为键，将包含新闻列表和用户信息的NumPy数组作为值存储到ucascades字典中

    '''ordered by timestamps'''
    for cas in list(ucascades.keys()):#使用list(ucascades.keys())获取ucascades字典中的所有用户ID，并逐个遍历
        order = [i[0] for i in sorted(enumerate(ucascades[cas][1]), key=lambda x: float(x[1]))]
        #对当前用户的转发信息（ ucascades[cas][1] ）进行排序。
        #enumerate函数用于生成每个时间戳的索引和对应的值。
        #sorted函数根据时间戳的值对这些索引进行排序， key=lambda x: float(x[1])  指定了排序的依据是时间戳的值。
        #最终，order列表包含了按时间戳排序后的索引。

        #ucascades[cas] = cascades[cas][:, order]
        ucascades[cas] = [[x[i] for i in order] for x in ucascades[cas]]#根据之前计算出的order列表对ucascades[cas]中的每个列表进行重新排序
        #ucascades[cas][1][:] = [ucascades[cas][1][i] for i in order]
        #ucascades[cas][0][:] = [ucascades[cas][0][i] for i in order]
    user_set = ucascades.keys()#将ucascades字典中的所有用户ID 存储在user_set变量中


    if os.path.exists(options.user_mapping):#检查指定的用户映射文件 ( options.user_mapping ) 是否存在。
        with open(options.user_mapping, 'rb') as handle:#如果文件存在，使用pickle模块以二进制模式打开用户映射文件，并将其内容加载到u2idx字典中
            u2idx = pickle.load(handle)
            user_size = len(list(user_set))#user_size变量存储用户集合的大小
        with open(options.news_mapping, 'rb') as handle:#打开新闻映射文件并将其内容加载到n2idx字典中
            n2idx = pickle.load(handle)
            news_size = len(news_list)#news_size变量存储新闻列表的大小
    else:
        user_size, news_size, u2idx, n2idx = buildIndex(user_set, news_list)#如果用户映射文件不存在，调用buildIndex函数来构建用户和新闻的索引，并返回用户数量、新闻数量、用户索引和新闻索引。
        with open(options.user_mapping, 'wb') as handle:
            pickle.dump(u2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(options.news_mapping, 'wb') as handle:
            pickle.dump(n2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
            #将构建好的用户索引 ( u2idx ) 和新闻索引 ( n2idx ) 保存到对应的映射文件中，以便后续使用。
            #使用pickle.dump并指定protocol=pickle.HIGHEST_PROTOCOL来确保以最高效的方式保存数据。

    for cas in cascades:#遍历  cascades  字典中的每个推文（ cas ）
        cascades[cas][0] = [u2idx[u] for u in cascades[cas][0]]#使用u2idx字典将用户列表（ cascades[cas][0] ）中的用户 ID 转换为对应的索引
    t_cascades = dict([(n2idx[key], cascades[key]) for key in cascades])#使用字典推导式创建一个新的字典t_cascades，将cascades中的每个推文的键（推文ID）转换为对应的新闻索引（使用n2idx ）

    for cas in ucascades:#遍历  ucascades  字典中的每个用户（ cas ）
        ucascades[cas][0] = [n2idx[n] for n in ucascades[cas][0]]#将其新闻列表（ ucascades[cas][0] ）中的新闻 ID 转换为对应的索引（使用n2idx ）
    u_cascades = dict([(u2idx[key], ucascades[key]) for key in ucascades])#使用字典推导式创建一个新的字典u_cascades，将ucascades中的每个用户的键（用户ID）转换为对应的用户索引（使用u2idx ）

    '''load labels'''
    labels = np.zeros((news_size + 1, 1))#使用NumPy创建一个大小为(news_size + 1, 1)的数组labels,并用零初始化。
    for line in open(options.label):#打开指定的标签文件（ options.label ），逐行读取内容
        news, label = line.strip().split(' ')#对每行进行处理，去掉首尾空白字符，并按空格分割成两个部分:news和label 。
        if news in n2idx:#检查news是否在n2idx字典中（即该新闻是否在已知的新闻索引中）
            labels[n2idx[news]] = label#如果存在，则根据n2idx[news]获取对应的索引，并将label填充到labels数组中

    seq = np.zeros((news_size + 1, max_len))
    timestamps = np.zeros((news_size + 1, max_len))
    user_level = np.zeros((news_size + 1, max_len))
    user_inf = np.zeros((news_size + 1, max_len))
    news_list = [0] + news_list
    for n, s in cascades.items():#遍历cascades字典的每一项，其中n是字典的键，而s是对应的值
        news_list[n2idx[n]] = n#将当前的键n存储到news_list列表中，索引由n2idx[n]提供
        #se_data通过将s[0]与一个填充数组拼接而成，填充数组的长度为max_len - len(s[0])，填充的值为Constants.PAD
        se_data = np.hstack((s[0], np.array([Constants.PAD] * (max_len - len(s[0])))))
        seq[n2idx[n]] = se_data#将处理后的序列数据存储到  seq  数组中

        t_data = np.hstack((s[1], np.array([Constants.PAD] * (max_len - len(s[1])))))
        timestamps[n2idx[n]] = t_data

        lv_data = np.hstack((s[2], np.array([Constants.PAD] * (max_len - len(s[2])))))
        user_level[n2idx[n]] = lv_data

        inf_data = np.hstack((s[3], np.array([Constants.PAD] * (max_len - len(s[3])))))
        user_inf[n2idx[n]] = inf_data

    useq = np.zeros((user_size + 1, max_len))
    uinfs = np.zeros((user_size + 1, max_len))

    for n, s in ucascades.items():
        if len(s[0])<max_len:
            #如果s[0]的长度小于max_len，则创建一个填充数组se_data，其内容为s[0]和若干个Constants.PAD（填充值），以使得最终的长度为max_len
            se_data = np.hstack((s[0], np.array([Constants.PAD] * (max_len - len(s[0])))))
            useq[u2idx[n]] = se_data

            tinf_data = np.hstack((s[1], np.array([Constants.PAD] * (max_len - len(s[1])))))#将s[1]与填充数组拼接而成
            uinfs[u2idx[n]] = tinf_data
        else:
            useq[u2idx[n]] = s[0][:max_len]#直接将  s[0]  的前  max_len  个元素存储到  useq  数组中
            #utimestamps[u2idx[n]] = s[1][:max_len]
            uinfs[u2idx[n]] = s[1][:max_len]#将  s[1]  的前  max_len  个元素存储到  uinfs  数组中

    total_len = sum(len(t_cascades[i][0]) for i in t_cascades)#计算t_cascades中所有新闻级联的总长度。这里假设t_cascades[i][0]是每个新闻级联的序列数据。
    total_ulen = sum(len(u_cascades[i][0]) for i in u_cascades)#计算u_cascades中所有用户参与的总长度。这里假设u_cascades[i][0]是每个用户参与的序列数据。
    print("total size:%d " % (len(seq) - 1))#新闻级联的总数
    print('spread size',(total_len))#输出所有新闻级联的总长度
    print("average news cascades length:%f" % (total_len / (len(seq) - 1)))#计算并输出平均新闻级联的长度
    print("average user participant length:%f" % (total_ulen / (len(useq) - 1)))#计算并输出平均用户参与的长度
    print("user size:%d" % (user_size))#输出用户的数量
    news_cascades = [seq, timestamps, user_level, user_inf]#将新闻级联相关的数据（序列、时间戳、用户等级和用户信息）打包成一个列表
    user_parti = [useq, uinfs]#将用户参与相关的数据（用户序列和用户信息）打包成一个列表。

    return news_cascades, user_parti, labels, user_size, news_list

if __name__ == "__main__":
    data_name = 'poli'
    options = Options(data_name)
    #调用Pre_data函数，传入数据名称和其他参数，获取预处理后的新闻级联、用户参与、标签、用户数量和新闻列表。
    news_cascades, user_parti, labels, user_size, news_list = Pre_data(data_name, early_type = Constants.early_type, early = 100)
    #准备训练、验证和测试数据
    train_news = np.array([i+1 for i in options.train_idx])
    valid_news = np.array([i+1 for i in options.valid_idx])
    test_news = np.array([i+1 for i in options.test_idx])
    #创建训练、验证和测试数据集，每个数据集包含新闻索引和对应的标签
    train_data = [train_news, labels[train_news]]
    valid_data = [valid_news,labels[valid_news]]
    test_data = [test_news,labels[test_news]]
    #使用  pickle  库将预处理后的数据保存到文件中。具体来说：
    #news_cascades  被保存到  options.news_centered  指定的文件中。
    #user_parti  被保存到  options.user_centered  指定的文件中。
    #train_data  被保存到  options.train  指定的文件中。
    #valid_data  被保存到  options.valid  指定的文件中。
    #test_data  被保存到  options.test  指定的文件中。
    with open(options.news_centered, 'wb') as handle:
        pickle.dump(news_cascades, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(options.user_centered, 'wb') as handle:
        pickle.dump(user_parti, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(options.train, 'wb') as handle:
        pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(options.valid, 'wb') as handle:
        pickle.dump(valid_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(options.test, 'wb') as handle:
        pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Data Preprocessing Done!')#输出提示信息，表示数据预处理完成。

