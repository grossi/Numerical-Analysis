import numpy as np

def gauss_solution(A, b):
    print("b ", b)
    n = np.shape(A)[0]
    x = np.zeros((n), dtype=float)
    for i in reversed(range(n)):
        print("b e A", b[i], A[i][i])
        x[i] = b[i]/A[i][i]
    return x

def gauss(A, b):
    n = np.shape(A)[0]
    L = np.identity(3, dtype=float)
    for k in range(n-1):
        # Swap lines to find the largest pivot
        for j in range(k+1, n):
            #print("if a[k][k]", A[k][k], "< A[j][k]", A[j][k])
            if( np.abs(A[k][k]) < np.abs(A[j][k])):
                #print("A before swap\n", A)
                A[[k, j]] = A[[j, k]]
                #print("A after swap\n", A)
                #print("b before", b)
                temp = b[k]
                b[k] = b[j]
                b[j] = temp
                #print("b after", b)
        pivot = A[k][k]
        if( A[k][k] == 0 ):
            return "ERROR DIVIDE BY 0"
        for i in range(k+1, n):
            m = A[i][k]/pivot
            #print(m)
            L[i][k] = m
            b[i] = b[i] - m * b[k]
            for j in range(k, n):
                A[i][j] = A[i][j] - m*A[k][j]
        #print("A: \n", A)
    return A, b, L


#C = np.array([[1, 2, 2], [2, 3, 5], [7, 8, 9]], dtype=float)

b1 = np.array([6, 11, 24], dtype=float)

A = np.array([[1, -2, 3], [4, 2, -1], [2, -1, 3]], dtype=float)

print(" TESTE 1 \n\n")
print(A, b1)
print("Result\n")
RA, Rb1, RL = gauss(A, b1)
print("RA:\n", RA)
print("Rb1:\n", Rb1)
print("RL :\n", RL)
print("RL*RA: \n", RL @ RA)

print(gauss_solution(RA, Rb1))
