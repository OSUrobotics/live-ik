test_arr = [[0,1]]

"""

for i, val in enumerate(test_arr):
    if isinstance(val, list):
        for k, inner_val in enumerate(val):
            test_arr[i][k] = inner_val + 1
            print(inner_val)
    else:
        test_arr[i] = val + 1
        print(inner_val)
print(test_arr)
"""
print(test_arr.count(list))
        