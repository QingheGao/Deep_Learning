import wget, os, gzip, pickle, random, re, sys
import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms, utils
import random
import matplotlib.pyplot as plt

IMDB_URL = 'http://dlvu.github.io/data/imdb.{}.pkl.gz'
IMDB_FILE = 'imdb.{}.pkl.gz'

PAD, START, END, UNK = '.pad', '.start', '.end', '.unk'

def load_imdb(final=False, val=5000, seed=0, voc=None, char=False):

    cst = 'char' if char else 'word'

    imdb_url = IMDB_URL.format(cst)
    imdb_file = IMDB_FILE.format(cst)

    if not os.path.exists(imdb_file):
        wget.download(imdb_url)

    with gzip.open(imdb_file) as file:
        sequences, labels, i2w, w2i = pickle.load(file)

    if voc is not None and voc < len(i2w):
        nw_sequences = {}

        i2w = i2w[:voc]
        w2i = {w: i for i, w in enumerate(i2w)}

        mx, unk = voc, w2i['.unk']
        for key, seqs in sequences.items():
            nw_sequences[key] = []
            for seq in seqs:
                seq = [s if s < mx else unk for s in seq]
                nw_sequences[key].append(seq)

        sequences = nw_sequences

    if final:
        return (sequences['train'], labels['train']), (sequences['test'], labels['test']), (i2w, w2i), 2

    # Make a validation split
    random.seed(seed)

    x_train, y_train = [], []
    x_val, y_val = [], []

    val_ind = set( random.sample(range(len(sequences['train'])), k=val) )
    for i, (s, l) in enumerate(zip(sequences['train'], labels['train'])):
        if i in val_ind:
            x_val.append(s)
            y_val.append(l)
        else:
            x_train.append(s)
            y_train.append(l)

    return (x_train, y_train), \
           (x_val, y_val), \
           (i2w, w2i), 2


def gen_sentence(sent, g):

    symb = '_[a-z]*'

    while True:

        match = re.search(symb, sent)
        if match is None:
            return sent

        s = match.span()
        sent = sent[:s[0]] + random.choice(g[sent[s[0]:s[1]]]) + sent[s[1]:]

def gen_dyck(p):
    open = 1
    sent = '('
    while open > 0:
        if random.random() < p:
            sent += '('
            open += 1
        else:
            sent += ')'
            open -= 1

    return sent

def gen_ndfa(p):

    word = random.choice(['abc!', 'uvw!', 'klm!'])

    s = ''
    while True:
        if random.random() < p:
            return 's' + s + 's'
        else:
            s+= word

def load_brackets(n=50_000, seed=0):
    return load_toy(n, char=True, seed=seed, name='dyck')

def load_ndfa(n=50_000, seed=0):
    return load_toy(n, char=True, seed=seed, name='ndfa')

def load_toy(n=50_000, char=True, seed=0, name='lang'):

    random.seed(0)

    if name == 'lang':
        sent = '_s'

        toy = {
            '_s': ['_s _adv', '_np _vp', '_np _vp _prep _np', '_np _vp ( _prep _np )', '_np _vp _con _s' , '_np _vp ( _con _s )'],
            '_adv': ['briefly', 'quickly', 'impatiently'],
            '_np': ['a _noun', 'the _noun', 'a _adj _noun', 'the _adj _noun'],
            '_prep': ['on', 'with', 'to'],
            '_con' : ['while', 'but'],
            '_noun': ['mouse', 'bunny', 'cat', 'dog', 'man', 'woman', 'person'],
            '_vp': ['walked', 'walks', 'ran', 'runs', 'goes', 'went'],
            '_adj': ['short', 'quick', 'busy', 'nice', 'gorgeous']
        }

        sentences = [ gen_sentence(sent, toy) for _ in range(n)]
        sentences.sort(key=lambda s : len(s))

    elif name == 'dyck':

        sentences = [gen_dyck(7./16.) for _ in range(n)]
        sentences.sort(key=lambda s: len(s))

    elif name == 'ndfa':

        sentences = [gen_ndfa(1./4.) for _ in range(n)]
        sentences.sort(key=lambda s: len(s))

    else:
        raise Exception(name)

    tokens = set()
    for s in sentences:

        if char:
            for c in s:
                tokens.add(c)
        else:
            for w in s.split():
                tokens.add(w)

    i2t = [PAD, START, END, UNK] + list(tokens)
    t2i = {t:i for i, t in enumerate(i2t)}

    sequences = []
    for s in sentences:
        if char:
            tok = list(s)
        else:
            tok = s.split()
        sequences.append([t2i[t] for t in tok])

    return sequences, (i2t, t2i)

def batch_change(train, maxium_token):
    x_batch = []
    x_batch_second=[]
    for i in train:
        if len(i) < maxium_token:
            k = i.copy()
            k.extend((maxium_token-len(k))*[0])
            x_batch.append(k)
    
        elif len(i) > maxium_token:
            number = int(np.ceil(len(i)/maxium_token))
            for k in range(number):
                x_batch.append(i[k*maxium_token:(k+1)*maxium_token])
        else:
            x_batch.append(i)
            
    for i in x_batch:
        if len(i) < maxium_token:
            k = i.copy()
            k.extend((maxium_token-len(k))*[0])
            x_batch_second.append(k)
        else:
            x_batch_second.append(i)
            
    return x_batch_second

def create_batch(x, maxium = 300):
    number = 0
    batchtotal = []
    batch = []
    for i in range(len(x)):
        batch.append(x[i])
        number += len(x[i])
        if number <= maxium:
            if i+1 == len(x):
                batchtotal.append(batch)
                break
            continue
        else:
            batchtotal.append(batch)
            number = 0
            batch = []

    return batchtotal
          


def propcess( maxium = 300, choose = 'ndfa'):
    if choose == 'ndfa':
        x_train, (i2w, w2i) = load_ndfa(n =150000)
    if choose == 'dyck':
        x_train, (i2w, w2i) = load_brackets(n =150000)
    if choose == 'toy':
        x_train, (i2w, w2i) = load_toy(n = 50000)
    if choose == 'imdb':
        (x_train, _), (_, _), (i2w, w2i), numcls = load_imdb(final=True, char=True)
        
    x_try = []
    for i in x_train:
        i.insert(0, 1)
        i.append(2)
        x_try.append(i)
    
    y_train = []
    for i in x_try:
        k = i[1:]
        k.append(0)
        y_train.append(k)
    
    
    ##batch 等长
#     x_final = batch_change(x_try, 5)
#     y_final = batch_change(y_train, 5)
#     k = np.array(list(zip(x_final, y_final)))
#     np.random.shuffle(k)
#     k = torch.tensor(k, dtype=torch.long)

#     ###batch 不等长
    x_batch = create_batch(x_try,maxium)
    y_batch = create_batch(y_train,maxium)
    
    
    ##padiing
    length = []
    for i in x_batch:
        zzz = max([len(k) for k in i])
        length.append(zzz)
    x_final = []
    y_final = []
    
    for i in range(len(x_batch)):
        x_trans = batch_change(x_batch[i], length[i])
        y_trans = batch_change(y_batch[i], length[i])
        
        x_final.append(x_trans)
        y_final.append(y_trans)
    k = list(zip(x_final, y_final))
    random.shuffle(k)

    
    return k,(i2w, w2i)

class LSTM(nn.Module):
    def __init__(self, char = 15, in_size =32, hidden = 16, num_layer = 2):
        super(LSTM, self).__init__()
        
        self.hidden = hidden
        self.char = char
        self.num_layer=num_layer
        self.embd = nn.Embedding(char,in_size)
        self.model = nn.LSTM(in_size, self.hidden, self.num_layer, batch_first=True)
        self.fc = nn.Linear(hidden, self.char)
        self.softmax = nn.Softmax(dim=1)
        
        # Initialize the weights
    
    def forward(self,x,hidden,cell):
        x = self.embd(x)
        x,(h,c) = self.model(x,(hidden,cell))
        x = x.reshape(x.shape[0]*x.shape[1], self.hidden)
        x = self.fc(x)
        x = self.softmax(x)
        return x, (h, c)

    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layer, batch_size, self.hidden).to(device)
        cell = torch.zeros(self.num_layer, batch_size, self.hidden).to(device)
        return hidden, cell
    
    
def generate(net, initial_str, temperature):
    predicted = initial_str.copy()
    
    initial_str = torch.tensor(initial_str, dtype=torch.long).reshape(3,1,-1)
    
    hidden, cell = net.init_hidden(batch_size=1)
    initial_input = initial_str

    for p in range(len(initial_str) - 1):
        _, (hidden, cell) = net(
            initial_input[p].to(device), hidden, cell
        )

    last_char = initial_input[-1]

    while True:
        output, (hidden, cell) = net(
            last_char.to(device), hidden, cell
        )
        if temperature.item() != 0:
            output_dist = output.data.view(-1).div(temperature).exp()
            top_char = torch.multinomial(output_dist, 1)[0]
        else:
            top_char = output.argmax()
        
        predicted.append(top_char.item())
            
        if top_char.int() == 2 or len(predicted)>200:
            break
        last_char = top_char.view(1,1)
    return predicted


def find_model(net,  combine, criterion, optimizer, device, initial_str, i2w,epochsize=10,t = 0.1):
    trainloss_final = []
    norm_final = []
    array1 = []
    for epoch in range(epochsize):
        print(epoch)
        trainloss_epoch = []
        norm = []
        
        for batch in combine:
    
            inputs, labels = batch[0], batch[1]
            inputs, labels = torch.tensor(inputs, dtype=torch.long).to(device), torch.tensor(labels, dtype=torch.long).to(device)
            
            hidden, cell = net.init_hidden(batch_size=inputs.shape[0])
            
            optimizer.zero_grad()
            
            labels = labels.reshape(labels.shape[0]*labels.shape[1],)
    
            outputs, (hidden, cell) = net(inputs,hidden,cell)
            loss = criterion(outputs, labels)
            
            hidden, cell = hidden.detach(), cell.detach()
            
            loss.backward()
        
            nn.utils.clip_grad_norm_(net.parameters(), 5)
            trainloss_epoch.append(loss.item()/len(labels))
        
            optimizer.step()
        
            with torch.no_grad():
                total_norm = 0
                for p in net.parameters():
                    sqsum = p.grad.data.pow(2).sum()
                    total_norm += sqsum
                total_norm = total_norm.sqrt().item()
                norm.append(total_norm)
        k=[]
        print('[%d] loss: %.4f' %(epoch + 1, np.mean(trainloss_epoch)))
        for i in range(10):
            predict = generate(net, initial_str,temperature = torch.tensor(t))
            k.append([i2w[i] for i in predict])

        array1.append(k)
        norm_final.append(np.mean(norm))
        trainloss_final.append(np.mean(trainloss_epoch))
        
        np.save('./result/array_final_im_300_1',array1)
        np.save('./result/norm_final_im_300_1',norm_final)
        np.save('./result/trainloss_final_im_300_1',trainloss_final)

    return  trainloss_final, norm_final

combine,(i2w, w2i) = propcess( maxium = 131000, choose = 'imdb')
net = LSTM(char = 210,in_size =300, hidden = 300,  num_layer = 1)
criterion = nn.CrossEntropyLoss(reduction ='sum')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
initial_str = [w2i['.start'],w2i['T'],w2i['h']]
optimizer = optim.Adam(net.parameters(), lr= 0.01)
trainloss_batch  = find_model(net,combine,
                            criterion,optimizer,device,initial_str, i2w,epochsize=20)
