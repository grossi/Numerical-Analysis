import numpy as np
npa = np.linalg

def getDLU(A):
    D = np.zeros_like(A)
    L = np.zeros_like(A)
    U = np.zeros_like(A)
    n = np.shape(A)[0]
    for k in range(n):
        D[k][k] = A[k][k]
        for j in range(n):
            if(k > j):
                L[k][j] = -1 * A[k][j]
            if(k < j):
                U[k][j] = -1 * A[k][j]
    return D, L, U

def iterativeMethod(A, b, P, N, tol):
    x = np.ones_like(A[0])
    r0 =  b - A.dot(x)[0]
    # print("r0", r0)
    r = r0
    Pinverse = npa.inv(P)
    # print("p\n", P, "\np^-1\n", Pinverse)
    e = npa.norm(r)/npa.norm(r0)
    iterations = 0
    while (e > tol):
        iterations+=1
        # print("e", e)
        # print("x", x)
        # print("P^-1 * r\n", Pinverse.dot(r))
        x = x + Pinverse.dot(r)
        r = b - A.dot(x)
        e = npa.norm(r)/npa.norm(r0)
    return x, iterations

# p é o vetor de direção
# 
# alpha é o tamanho do passo

def conjugateGradient(A, b):
    x = np.zeros_like(A[0])
    r = b - A.dot(x)
    # print("r", r)
    p = r
    peta = r.T.dot(r)
    # print("peta", peta)
    for _ in range(len(b)):
        a_dot_p = A.dot(p)
        # print("p", np.transpose(p))
        # print("Scalar", p.T.dot(a_dot_p))
        alpha = peta / p.T.dot(a_dot_p)
        # print("alpha", alpha)
        x = x + alpha * p
        # print("x", x)
        # print("r.dot r +! ", r.dot(r - alpha * a_dot_p))
        r = r - alpha * a_dot_p
        beta = peta
        peta = r.T.dot(r)
        beta = peta / beta
        p = r + beta * p
    return x

b = np.array([6, 11, 24], dtype=float)

A = np.array([[140, 23, 3], [23, 7, -1], [3, -1, 83]], dtype=float)

print(" \n\nTESTE 1 CG \n\n")
print("A:\n", A)
print("Result\n")
x = conjugateGradient(A, b)
print("x\n", x)
print("x*A\n", A.dot(x), "\nb\n", b)

N = 15
C_temp = np.random.random_integers(-2000,2000,size=(N,N))
C = (C_temp + C_temp.T)/2
b = np.random.random_integers(-2000, 2000, size=(N))

print(" \n\nTESTE 2 CG \n\n")
print("A:\n", C)
print("Result\n")
x = conjugateGradient(C, b)
print("x\n", x)
print("x*A\n", C.dot(x), "\nb\n", b)

