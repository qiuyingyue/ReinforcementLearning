#Make sure run this with python3

import pickle
f=open('Silvia_expert100.pkl', 'rb')
d=f.read()
data = pickle.loads(d)
obs = data['obs']
act = data['act']
l_o = len(obs[1])
l_a = len(act[1])
print(l_o)
print(l_a)
