from pool import Pool, write
from layer_graph import Layer_graph, LAYERS

P = Pool()
G = Layer_graph(57784)
G.add_node(LAYERS.ip)
G.append(LAYERS.conv3, 64, 1)
G.append(LAYERS.maxpool)
G.append(LAYERS.conv3, 128, 1)
G.append(LAYERS.maxpool)
# G.append(LAYERS.conv3, 128, 1)
# G.append(LAYERS.maxpool)
# G.append(LAYERS.conv3, 256, 1)
# G.append(LAYERS.maxpool)
# G.append(LAYERS.conv3, 512, 1)
# G.append(LAYERS.maxpool)
G.append(LAYERS.fc, 128)
# G.append(LAYERS.fc, 256)
# G.append(LAYERS.fc, 512)
G.append(LAYERS.softmax)
G.append(LAYERS.op)
G.finish()
P.append(G, 0.6751)
G = Layer_graph(92111)
G.add_node(LAYERS.ip)
G.append(LAYERS.conv3, 64, 1)
G.append(LAYERS.batchnorm)
G.append(LAYERS.maxpool)
G.append(LAYERS.resnet, 64)
G.append(LAYERS.batchnorm)
G.append(LAYERS.conv3, 128, 2)
G.append(LAYERS.resnet, 128)
G.append(LAYERS.batchnorm)
G.append(LAYERS.fc, 1024)
G.append(LAYERS.softmax)
G.append(LAYERS.op)
G.finish()
P.append(G, 0.7109)
G = Layer_graph(126517)
G.add_node(LAYERS.ip)
G.append(LAYERS.conv3, 64, 1)
G.append(LAYERS.conv3, 64, 1)
G.append(LAYERS.maxpool)
G.append(LAYERS.conv3, 128, 1)
G.append(LAYERS.conv3, 128, 1)
G.append(LAYERS.maxpool)
G.append(LAYERS.conv3, 128, 1)
G.append(LAYERS.conv3, 128, 1)
G.append(LAYERS.conv3, 128, 1)
G.append(LAYERS.conv3, 128, 1)
G.append(LAYERS.maxpool)
G.append(LAYERS.conv3, 256, 1)
G.append(LAYERS.conv3, 256, 1)
G.append(LAYERS.conv3, 256, 1)
G.append(LAYERS.conv3, 256, 1)
G.append(LAYERS.maxpool)
G.append(LAYERS.conv3, 512, 1)
G.append(LAYERS.conv3, 512, 1)
G.append(LAYERS.conv3, 512, 1)
G.append(LAYERS.conv3, 512, 1)
G.append(LAYERS.maxpool)
G.append(LAYERS.fc, 128)
G.append(LAYERS.fc, 256)
G.append(LAYERS.fc, 512)
G.append(LAYERS.softmax)
G.append(LAYERS.op)
G.finish()
P.append(G, 0.1)
'''
G = Layer_graph(57735)
G.add_node(LAYERS.ip)
G.append(LAYERS.conv7, 64, 1)
G.append(LAYERS.maxpool)
G.append(LAYERS.conv3, 64) #conv 3 / 2
G.append(LAYERS.conv3, 64, 1)
G.append(LAYERS.conv3, 128) #conv 3 / 2
G.append(LAYERS.conv3, 128, 1)
G.append(LAYERS.conv3, 256) #conv 3 / 2
G.append(LAYERS.conv3, 256, 1)
G.append(LAYERS.conv3, 512) #conv 3 / 2
G.append(LAYERS.conv3, 512, 1)
G.append(LAYERS.avgpool)
G.append(LAYERS.fc, 1024)
G.append(LAYERS.softmax)
G.append(LAYERS.op)
G.finish()
P.append(G, 0.6)
G = Layer_graph(92552)
G.add_node(LAYERS.ip)
G.append(LAYERS.conv7, 64, 1)
G.append(LAYERS.maxpool)
G.append(LAYERS.conv3, 64) #conv 3 / 2
G.append(LAYERS.conv3, 64, 1)
G.append(LAYERS.conv3, 64, 1)
G.append(LAYERS.conv3, 128) #conv 3 / 2
G.append(LAYERS.conv3, 128, 1)
G.append(LAYERS.conv3, 128, 1)
G.append(LAYERS.conv3, 256) #conv 3 / 2
G.append(LAYERS.conv3, 256, 1)
G.append(LAYERS.conv3, 256, 1)
G.append(LAYERS.conv3, 512) #conv 3 / 2
G.append(LAYERS.conv3, 512, 1)
G.append(LAYERS.conv3, 512, 1)
G.append(LAYERS.avgpool)
G.append(LAYERS.fc, 1024)
G.append(LAYERS.softmax)
G.append(LAYERS.op)
G.finish()
P.append(G, 0.85)
G = Layer_graph(31659)
G.add_node(LAYERS.ip)
G.append(LAYERS.conv7, 64, 1)
G.append(LAYERS.maxpool)
G.append(LAYERS.conv3, 64) #conv 3 / 2
G.append(LAYERS.conv3, 64, 1)
G.append(LAYERS.conv3, 64, 1)
G.append(LAYERS.conv3, 64, 1)
G.append(LAYERS.conv3, 128) #conv 3 / 2
G.append(LAYERS.conv3, 128, 1)
G.append(LAYERS.conv3, 128, 1)
G.append(LAYERS.conv3, 128, 1)
G.append(LAYERS.conv3, 256) #conv 3 / 2
G.append(LAYERS.conv3, 256, 1)
G.append(LAYERS.conv3, 256, 1)
G.append(LAYERS.conv3, 256, 1)
G.append(LAYERS.avgpool)
G.append(LAYERS.fc, 512)
G.append(LAYERS.softmax)
G.append(LAYERS.op)
G.finish()
P.append(G, 0.9)
G = Layer_graph(127367)
G.add_node(LAYERS.ip)
G.append(LAYERS.conv7, 64, 1)
G.append(LAYERS.maxpool)
G.append(LAYERS.conv3, 64) #conv 3 / 2
G.append(LAYERS.conv3, 64, 1)
G.append(LAYERS.conv3, 64, 1)
G.append(LAYERS.conv3, 64, 1)
G.append(LAYERS.conv3, 128) #conv 3 / 2
G.append(LAYERS.conv3, 128, 1)
G.append(LAYERS.conv3, 128, 1)
G.append(LAYERS.conv3, 128, 1)
G.append(LAYERS.conv3, 256) #conv 3 / 2
G.append(LAYERS.conv3, 256, 1)
G.append(LAYERS.conv3, 256, 1)
G.append(LAYERS.conv3, 256, 1)
G.append(LAYERS.conv3, 512) #conv 3 / 2
G.append(LAYERS.conv3, 512, 1)
G.append(LAYERS.conv3, 512, 1)
G.append(LAYERS.conv3, 512, 1)
G.append(LAYERS.avgpool)
G.append(LAYERS.fc, 1024)
G.append(LAYERS.softmax)
G.append(LAYERS.op)
G.finish()
P.append(G, 0.95)
'''
write(P, 'models/pool0')