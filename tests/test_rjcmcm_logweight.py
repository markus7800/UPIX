import numpy as np
import scipy


Z = np.zeros((11,11))
# Z[j,i] move from i to j

Z[1,0] = 0.073009

Z[2,1] = 0.055523
Z[0,1] = 0.000542

Z[3,2] = 0.062438
Z[1,2] = 0.000289

Z[4,3] = 0.033668
Z[2,3] = 0.000447

Z[5,4] = 0.020631
Z[3,4] = 0.024580

Z[6,5] = 0.015525
Z[4,5] = 0.057171

Z[7,6] = 0.013637
Z[5,6] = 0.079494

Z[8,7] = 0.012420
Z[6,7] = 0.092142

Z[9,8] = 0.011297
Z[7,8] = 0.103132

Z[10,9] = 0.010433
Z[8,9] = 0.112220

# Z[11,10] = 0.009782
Z[9,10] = 0.119647

print(Z[:5,:5])

A = np.zeros((11,11))
b = np.zeros(11)

for i in range(11):
    for j in range(11):
        if Z[j,i] != 0:
            A[i,i] += 1
            A[i,j] -= 1
            b[i] += np.log(Z[i,j]) - np.log(Z[j,i])
            
print(A[:5,:5])
print(b[:5])
print(np.linalg.det(A))
A[1,1] += 1
print(np.linalg.det(A))
logw = np.linalg.solve(A, b)
logw -= scipy.special.logsumexp(logw)
w = np.exp(logw)
for i in range(11):
    print(f"{i}. {w[i]:.8f}")


# MC-DCCResult {
#         #clusters=1: StackedSampleValues(dict, 25,000 x 16) with prob=0.000000, log_Z=-16.193192
#         #clusters=2: StackedSampleValues(dict, 25,000 x 16) with prob=0.000012, log_Z=-11.290582
#         #clusters=3: StackedSampleValues(dict, 25,000 x 16) with prob=0.002400, log_Z=-6.032095
#         #clusters=4: StackedSampleValues(dict, 25,000 x 16) with prob=0.335129, log_Z=-1.093240
#         #clusters=5: StackedSampleValues(dict, 25,000 x 16) with prob=0.459034, log_Z=-0.778631
#         #clusters=6: StackedSampleValues(dict, 25,000 x 16) with prob=0.165647, log_Z=-1.797898
#         #clusters=7: StackedSampleValues(dict, 25,000 x 16) with prob=0.032350, log_Z=-3.431136
#         #clusters=8: StackedSampleValues(dict, 25,000 x 16) with prob=0.004788, log_Z=-5.341713
#         #clusters=9: StackedSampleValues(dict, 25,000 x 16) with prob=0.000577, log_Z=-7.458423
#         #clusters=10: StackedSampleValues(dict, 25,000 x 16) with prob=0.000058, log_Z=-9.754378
#         #clusters=11: StackedSampleValues(dict, 25,000 x 16) with prob=0.000005, log_Z=-12.193935
# }

Z = np.zeros((11,11))
# Z[j,i] move from i to j

Z[1,0] = 0.073009
Z[0,0] = 1 - 0.073009

Z[2,1] = 0.055523 * 0.5
Z[0,1] = 0.000542 * 0.5
Z[1,1] = 1 - (0.055523 + 0.000542) * 0.5

Z[3,2] = 0.062438 * 0.5
Z[1,2] = 0.000289 * 0.5
Z[2,2] = 1 - (0.062438 + 0.000289) * 0.5

Z[4,3] = 0.033668 * 0.5
Z[2,3] = 0.000447 * 0.5
Z[3,3] = 1 - (0.033668 + 0.000447) * 0.5

Z[5,4] = 0.020631 * 0.5
Z[3,4] = 0.024580 * 0.5
Z[4,4] = 1 - (0.020631 + 0.024580) * 0.5

Z[6,5] = 0.015525 * 0.5
Z[4,5] = 0.057171 * 0.5
Z[5,5] = 1 - (0.015525 + 0.057171) * 0.5

Z[7,6] = 0.013637 * 0.5
Z[5,6] = 0.079494 * 0.5
Z[6,6] = 1 - (0.013637 + 0.079494) * 0.5

Z[8,7] = 0.012420 * 0.5
Z[6,7] = 0.092142 * 0.5
Z[7,7] = 1 - (0.012420 + 0.092142) * 0.5

Z[9,8] = 0.011297 * 0.5
Z[7,8] = 0.103132 * 0.5
Z[8,8] = 1 - (0.011297 + 0.103132) * 0.5

Z[10,9] = 0.010433 * 0.5
Z[8,9] = 0.112220 * 0.5
Z[9,9] = 1 - (0.010433 + 0.112220) * 0.5

# Z[11,10] = 0.009782 * 0.5
# Z[9,10] = 0.119647 * 0.5
# Z[10,10] = 1 - (0.009782 + 0.119647) * 0.5
Z[9,10] = 0.119647
Z[10,10] = 1 - 0.119647
print(Z[:5,:5])

A = np.zeros((11,11))
b = np.zeros(11)

for i in range(11):
    for j in range(11):
        if Z[j,i] != 0:
            A[i,i] += 1
            A[i,j] -= 1
            b[i] += np.log(Z[i,j]) - np.log(Z[j,i])
            
print(A[:5,:5])
print(b[:5])
print(np.linalg.det(A))
A[1,1] += 1
print(np.linalg.det(A))
logw = np.linalg.solve(A, b)
logw -= scipy.special.logsumexp(logw)
w = np.exp(logw)
for i in range(11):
    print(f"{i}. {w[i]:.8f}")
    
    
    
    
Z = np.zeros((11,11))
# Z[j,i] move from i to j

Z[1,0] = 0.071878
Z[0,0] = 0.928122

Z[2,1] = 0.056125
Z[0,1] = 0.000542
Z[1,1] = 0.943332

Z[3,2] = 0.062641
Z[1,2] = 0.000277
Z[2,2] = 0.937081

Z[4,3] = 0.033815
Z[2,3] = 0.000476
Z[3,3] = 0.965709

Z[5,4] = 0.020584
Z[3,4] = 0.024599
Z[4,4] = 0.954817

Z[6,5] = 0.015924
Z[4,5] = 0.057074
Z[5,5] = 0.927002

Z[7,6] = 0.013532 
Z[5,6] = 0.079446
Z[6,6] = 0.907021

Z[8,7] = 0.012426
Z[6,7] = 0.092407
Z[7,7] = 0.895166

Z[9,8] = 0.011226
Z[7,8] = 0.102071
Z[8,8] = 0.886702

Z[10,9] = 0.010315
Z[8,9] = 0.112250
Z[9,9] = 0.877435

# Z[11,10] = 0.009729 
# Z[9,10] = 0.120249
# Z[10,10] = 0.870022
Z[9,10] = 0.120249
Z[10,10] = 0.870022
print(Z[:5,:5])

A = np.zeros((11,11))
b = np.zeros(11)

for i in range(11):
    for j in range(11):
        if Z[j,i] != 0:
            A[i,i] += 1
            A[i,j] -= 1
            b[i] += np.log(Z[i,j]) - np.log(Z[j,i])
            
print(A[:5,:5])
print(b[:5])
print(np.linalg.det(A))
A[0,0] += 1
print(np.linalg.det(A))
logw = np.linalg.solve(A, b)
logw -= scipy.special.logsumexp(logw)
w = np.exp(logw)
for i in range(11):
    print(f"{i}. {w[i]:.8f}")