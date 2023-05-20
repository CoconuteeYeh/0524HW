# -*- coding: utf-8 -*-
"""
Created on Sat May 20 20:52:19 2023

@author: Chih Tung
"""

import sys

# Brute-Force Algorithm
def matrix_chain_multiplication_bf(P):
    n = len(P) - 1
    return matrix_chain_bf(P, 1, n)

def matrix_chain_bf(P, i, j):
    if i == j:
        return 0, f"A{i}"
    
    min_cost = sys.maxsize
    optimal_ordering = ""

    for k in range(i, j):
        left_cost, left_ordering = matrix_chain_bf(P, i, k)
        right_cost, right_ordering = matrix_chain_bf(P, k+1, j)
        
        cost = left_cost + right_cost + P[i-1] * P[k] * P[j]
        
        if cost < min_cost:
            min_cost = cost
            optimal_ordering = f"({left_ordering}) x ({right_ordering})"
    
    return min_cost, optimal_ordering


# Dynamic Programming Algorithm
def matrix_chain_multiplication_dp(P):
    n = len(P) - 1
    m = [[0] * (n+1) for _ in range(n+1)]
    s = [[0] * (n+1) for _ in range(n+1)]
    
    for l in range(2, n+1):
        for i in range(1, n-l+2):
            j = i + l - 1
            m[i][j] = sys.maxsize
            
            for k in range(i, j):
                temp_cost = m[i][k] + m[k+1][j] + P[i-1] * P[k] * P[j]
                
                if temp_cost < m[i][j]:
                    m[i][j] = temp_cost
                    s[i][j] = k
    
    return m[1][n], construct_optimal_parenthesization(s, 1, n)

def construct_optimal_parenthesization(s, i, j):
    if i == j:
        return f"A{i}"
    
    k = s[i][j]
    left = construct_optimal_parenthesization(s, i, k)
    right = construct_optimal_parenthesization(s, k+1, j)
    
    return f"({left} x {right})"


# Test the implementations and compare running times
import timeit
import matplotlib.pyplot as plt

def test_algorithm(algorithm, sizes):
    running_times = []
    
    for size in sizes:
        P = generate_random_dimensions(size)
        
        start_time = timeit.default_timer()
        result = algorithm(P)
        end_time = timeit.default_timer()
        
        running_time = end_time - start_time
        running_times.append(running_time)
        
        print(f"Size: {size}")
        print(f"Running Time: {running_time:.6f} sec")
        print(f"Optimal Scalar Multiplications: {result[0]}")
        print(f"Optimal Matrix Multiplication Ordering: {result[1]}\n")
    
    return running_times

def generate_random_dimensions(size):
    import random
    return [random.randint(1, 100) for _ in range(size+1)]

input_sizes = [5, 10, 15, 20, 21, 22]
bf_running_times = test_algorithm(matrix_chain_multiplication_bf, input_sizes)
dp_running_times = test_algorithm(matrix_chain_multiplication_dp, input_sizes)

plt.plot(input_sizes, bf_running_times, label='Brute-Force')
plt.plot(input_sizes, dp_running_times, label='Dynamic Programming')
plt.xlabel('Input Size')
plt.ylabel('Running Time (seconds)')
plt.legend()
plt.show()