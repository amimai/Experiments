import file3


test_set = []


test_set.append(['a1',5,40,0.0001500,30,240])
test_set.append(['a2',10,40,0.0001500,30,240])
test_set.append(['a3',20,40,0.0001500,30,240])
test_set.append(['a4',30,40,0.0001500,30,240])
test_set.append(['a5',40,40,0.0001500,30,240])

test_set.append(['a6',30,40,0.0000500,30,240])
test_set.append(['a7',30,40,0.0001000,30,240])
test_set.append(['a8',30,40,0.0001500,30,240])
test_set.append(['a9',30,40,0.0002000,30,240])
test_set.append(['aa',30,40,0.0002500,30,240])

test_set.append(['ab',30,40,0.0001500,30,240])
test_set.append(['ac',30,40,0.0001500,60,240])
test_set.append(['ad',30,40,0.0001500,90,240])
test_set.append(['ae',30,40,0.0001500,120,240])
test_set.append(['af',30,40,0.0001500,150,240])

for ea in test_set:
    file3.testRun(ea[0],ea[1],ea[2],ea[3],ea[4],ea[5])
