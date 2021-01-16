import ipyvolume as ipv
import numpy as np

N = 10000
x,y,z = np.random.normal(0,1,(3,N))

fig = ipv.figure()
scatter = ipv.scatter(x,y,z, marker="sphere")
ipv.save("here.html")



