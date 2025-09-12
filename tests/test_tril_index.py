
import math

N = 16

count = 0
ixs = set()
for i in range(N):
    for j in range(i):
        ix = (i * (i-1)) // 2 + j
        print(f"{i=}, {j=}, {ix=}")
        
            # =================================================
        a = int((1 + math.sqrt(8*ix + 1)) / 2)
        b = ix - (a * (a-1)) // 2
        assert a == i, a
        assert b == j, b
        
        # =================================================
        a = 0
        s = 0
        while s <= ix:
            a += 1
            s += a
        assert a == i, a
        assert s - a == (a * (a-1)) // 2
        b = ix - (s - a)
        assert b == j, b
        
        # =================================================
        
        s1 = 0
        s2 = 0
        while s2 <= ix:
            s1 += 1
            s2 += s1
        a = s1
        assert s1 == i, s1
        assert s2 - s1 == (a * (a-1)) // 2
        b = ix - (s2 - s1)
        assert b == j, b
        
        
        ixs.add(ix)
        count += 1
        
print("Check:", count, (N-1)*N//2, len(ixs))
# exit(0)

N = 16

count = 0
ixs = set()
for i in range(N):
    for j in range(i):
        for k in range(j):
            ix = (i*(i-1)*(i-2))//6 + (j*(j-1)) // 2 + k
            print(f"{i=}, {j=}, {k=} {ix=}")
            ixs.add(ix)
            count += 1
            
            # =================================================
            ix = (i*(i-1)*(i-2))//6 + (j*(j-1)) // 2 + k
            
            a = 0
            s = 0
            while s <= ix:
                a += 1
                s += (a*(a-1))//2
            assert a == i, a
            assert s - (a*(a-1))//2 == (a*(a-1)*(a-2))//6
            ix -= (s - (a*(a-1))//2)
            
            b = 0
            s = 0
            while s <= ix:
                b += 1
                s += b
            assert b == j, b
            assert s - b == (b * (b-1)) // 2
            c = ix - (s - b)
            assert c == k, c
            
            # =================================================
            ix = (i*(i-1)*(i-2))//6 + (j*(j-1)) // 2 + k
                
            s1 = 0
            s2 = 0
            s3 = 0
            s = 0
            while s <= ix:
                # s1 += 1
                
                # s2 += (s1 - 1)
                # assert s2 == (s1*(s1-1))//2
                
                # s += s2
                
                # s3 += (s2 - (s1 - 1))
                # assert s3 == (s1*(s1-1)*(s1-2))//6
                
                s3 += s2                
                s2 += s1
                s1 += 1
                
                assert s2 == (s1*(s1-1))//2
                assert s3 == (s1*(s1-1)*(s1-2))//6
                
                s += s2
                
                
            a = s1
            assert a == i, a
            
            s1 = 0
            s2 = 0
            s = 0
            while s <= ix - s3:
                s2 += s1
                s1 += 1
                
                s += s1
                
                
            b = s1
            assert b == j, b
            
            
            c = ix - s3 - s2
            assert c == k, c
            
        
print("Check:", count, (N*(N-1)*(N-2))//6, len(ixs))
exit()


N = 16

count = 0
ixs = set()
for i in range(N):
    for j in range(i):
        for k in range(j):
            for l in range(k):
                ix = (i*(i-1)*(i-2)*(i-3))//24  +  (j*(j-1)*(j-2))//6 + (k*(k-1)) // 2 + l
                print(f"{i=}, {j=}, {k=} {l=} {ix=}")
                ixs.add(ix)
                count += 1
                                    
                s1 = 0
                s2 = 0
                s3 = 0
                s4 = 0
                s = 0
                while s <= ix:
                    s4 += s3
                    s3 += s2
                    s2 += s1
                    s1 += 1
                
                    assert s2 == (s1*(s1-1))//2
                    assert s3 == (s1*(s1-1)*(s1-2))//6
                    assert s4 == (s1*(s1-1)*(s1-2)*(s1-3))//24
                    
                    s += s3
                    
                a = s1
                assert a == i, a
                
                s1 = 0
                s2 = 0
                s3 = 0
                s = 0
                while s <= ix - s4:
                    s3 += s2
                    s2 += s1
                    s1 += 1                
                    s += s2
                b = s1
                assert b == j, b
                
                
                s1 = 0
                s2 = 0
                s = 0
                while s <= ix - s4 - s3:
                    s2 += s1
                    s1 += 1
                    s += s1
                c = s1
                assert c == k, c
                
                d = ix - s4 - s3 - s2
                assert d == l, d
            
        
print("Check:", count, (N*(N-1)*(N-2)*(N-3))//24, len(ixs))