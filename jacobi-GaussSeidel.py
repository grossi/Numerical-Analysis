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

#C = np.array([[1, 2, 2], [2, 3, 5], [7, 8, 9]], dtype=float)

b = np.array([6, 11, 24], dtype=float)

A = np.array([[104, -23, 3], [4, 23, -13], [56, -1, 83]], dtype=float)

# A = P - N
#
# A = - L + D - U
# D = Matrix Diagonal
# L = Lower part times -1
# U = Upper part times -1

print(" TESTE 1 DLU \n\n")
print(A)
print("Result\n")
D, L, U = getDLU(A)
print("D:\n", D)
print("L:\n", L)
print("U :\n", U)
print("D+L+U\n", D+L+U)


# Jacobi: P = D
#         N = (U + L)
#
P = D
# print("p\n", P, "\np^-1\n", npa.inv(np.matrix(P)), "\nP * P^-1\n", P.dot(npa.inv(np.matrix(P))))
N = U + L
Bj = npa.inv(np.matrix(P)) * N
# print("Bj\n", Bj)
print("Inf Norm of Bj: ", npa.norm(Bj, np.inf))

xJacobi, iterations = iterativeMethod(A, b, P, N, 0.0001)

print("Jacobi solution:", xJacobi, "\n Iterations: ", iterations)


# Gauss-Seidel: P = D - L
#               N = U

P = D - L
N = U
Bgs = npa.inv(np.matrix(P)) * N
# print("Bgs\n", Bgs)
print("Inf Norm of Bgs: ", npa.norm(Bgs, np.inf))

xGS, iterations = iterativeMethod(A, b, P, N, 0.0001)

print("Gauss-Seidel solution:\n", xGS, "\n Iterations: ", iterations)

N = 4
C_temp = np.random.random_integers(-2000,2000,size=(N,N))
# Makes the matrix diagonal dominant
for i in range (0, N):
    C_temp[i][i]+= 4000
A = (C_temp + C_temp.T)/2
b = np.random.random_integers(-2000, 2000, size=(N))

print(" TESTE 2 DLU \n\n")
print("A:\n", A)
print("Result\n")
D, L, U = getDLU(A)
print("D:\n", D)
print("L:\n", L)
print("U :\n", U)
print("D+L+U\n", D+L+U)


# Jacobi: P = D
#         N = (U + L)
#
P = D
# print("p\n", P, "\np^-1\n", npa.inv(np.matrix(P)), "\nP * P^-1\n", P.dot(npa.inv(np.matrix(P))))
N = U + L
Bj = npa.inv(np.matrix(P)) * N
# print("Bj\n", Bj)
print("Inf Norm of Bj: ", npa.norm(Bj, np.inf))

xJacobi, iterations = iterativeMethod(A, b, P, N, 0.0001)

print("Jacobi solution:", xJacobi, "\n Iterations: ", iterations)


# Gauss-Seidel: P = D - L
#               N = U

P = D - L
N = U
Bgs = npa.inv(np.matrix(P)) * N
# print("Bgs\n", Bgs)
print("Inf Norm of Bgs: ", npa.norm(Bgs, np.inf))

xGS, iterations = iterativeMethod(A, b, P, N, 0.0001)

print("Gauss-Seidel solution:\n", xGS, "\n Iterations: ", iterations)