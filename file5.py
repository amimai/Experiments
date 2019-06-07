import file3


test_set = []

bs = 40
lr = 0.0002000
ts =120


test_set.append(['a1', bs, 40, lr, ts, 120])
test_set.append(['a2', bs, 40, lr, ts, 240])
test_set.append(['a3', bs, 40, lr, ts, 360])#best
test_set.append(['a4', bs, 40, lr, ts, 480])
test_set.append(['a5', bs, 40, lr, ts, 600])

for ea in test_set:
    file3.testRun(ea[0], ea[1], ea[2], ea[3], ea[4], ea[5])
