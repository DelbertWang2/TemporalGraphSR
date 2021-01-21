
sizes = [
    [64, 128],
    [128, 256],
    [256, 512],
    [512, 256],
    [256, 128],
    [128, 64]
]
nodes = 33

params = 0
for ilayer in range(len(sizes)):
    size = sizes[ilayer]
    params += size[0] * size[1] + size[1] * nodes

print(params/1e6)


FLOPs = 0
for ilayer in range(len(sizes)):
    size = sizes[ilayer]
    FLOPs += nodes * nodes * size[0] + nodes * size[0] * size[1]

print(FLOPs)