'''
Example 6.3 (pg. 360) from Zurada 1992 book:
Four 16-pixel bit maps of letter characters associated to
 7-bit binary vectors.

NN-Team-2
'''

import numpy as np

class BAM(object):
    def __init__(self, data):
        self.AB = []
        # store associations in bipolar form to the array
        for item in data:
            self.AB.append(
                [self.__l_make_bipolar(item[0]),
                 self.__l_make_bipolar(item[1])]
            )
        self.len_x = len(self.AB[0][1])
        self.len_y = len(self.AB[0][0])
        # create empty BAM matrix
        self.M = [[0 for x in range(self.len_x)] for x in range(self.len_y)]
        # compute BAM matrix from associations
        self.__create_bam()
        
    def __create_bam(self):
        '''Bidirectional associative memory'''
        #print("Constructing Weight Matrix")
        for assoc_pair in self.AB:
            X = np.asarray(assoc_pair[0]).T
            Y = np.asarray(assoc_pair[1])
            #print("A = ", X)
            #print("B = ", Y)
            # calculate M
            for idx, xi in enumerate(X):
                for idy, yi in enumerate(Y):
                    self.M[idx][idy] += xi * yi

    def get_assoc(self, A):
        '''Return association for input vector A'''
        # Assuming we are given input vector a at k = 1 (first forward pass)
        maxiter = 10 # Max attempts to associate for the given vector

        forward = np.asarray(A)
        backward = forward
        weights = np.asarray(self.M)

        # Need to remember the previous forward & backward passes to check if converged
        prev_backward = forward
        prev_forward = forward
        match = 0

        i = 0 # Current iteration number
        b = 2 
        for i in range(maxiter+1):
            # Compute W^T * A to get b^k (kth time)
            # First time through, we are solving for b at k = 2
            prev_backward = backward # Save the old backward pass
            
            backward = np.matmul(weights.T, forward) 
            print("b^{:d} = ".format(b), backward)
            backward = self.__threshold(backward)
            
                        
            # Compute W * b to get a^k
            prev_forward = forward # Save the old forward pass
            
            forward = np.matmul(weights, backward)
            print("a^{:d} = ".format(b+1), forward)
            forward = self.__threshold(forward)
            
            # Stop if forward & backward passes are identical to their previous value
            if np.array_equal(backward, prev_backward) and np.array_equal(forward, prev_forward):
                match = 1
                break

            b = b + 2
            
        if match == 0:
            print("WARN: Could not resolve association after {:d} iterations!".format(i))
                
        return self.__threshold(forward)
    
    def get_bam_matrix(self):
        '''Return BAM matrix'''
        return self.M

    def __mult_mat_vec(self, vec):
        '''Multiply input vector with BAM matrix'''
        v_res = [0] * self.len_x
        for x in range(self.len_x):
            for y in range(self.len_y):
                v_res[x] += vec[y] * self.M[y][x]
        return v_res

    def __threshold(self, vec):
        '''Transform vector to [-1, 1]'''
        ret_vec = []
        for i in vec:
            if i < 0:
                ret_vec.append(-1)
            elif i > 0:
                ret_vec.append(1)
            else:
                ret_vec.append(0)
        return ret_vec

    def __l_make_bipolar(self, vec):
        '''Transform vector to bipolar form [-1, 1]'''
        ret_vec = []
        for item in vec:
            if item == 0:
                ret_vec.append(-1)
            if item == -1:
                ret_vec.append(-1)
            else:
                ret_vec.append(1)
        return ret_vec

    def print_matrix(self):
        print("Weight Matrix ({:d}x{:d}):".format(self.len_x, self.len_y))
        print(np.matrix(self.M))
        print('')

if __name__ == "__main__":

    data_pairs = []

    # Below are the training vectors from Zurada92, pg 361
    a_1 = np.array([1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1])
    b_1 = np.array([1, -1, -1, -1, -1, 1, 1])

    a_2 = np.array([1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1])
    b_2 = np.array([1, -1, -1, 1, 1, 1, -1])

    a_3 = np.array([1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1])
    b_3 = np.array([1, -1, 1, 1, -1, 1, -1])

    a_4 = np.array([-1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1])
    b_4 = np.array([-1, 1, 1, -1, 1, -1, 1])
       
    print("Starting Association Pairs:")
    print("---------------------")
    print("A1 = ", a_1)
    print("A2 = ", a_2)
    print("A3 = ", a_3)
    print("A4 = ", a_4)
    print("B1 = ", b_1)
    print("B2 = ", b_2)
    print("B3 = ", b_3)
    print("B4 = ", b_4)
    print("---------------------")

    # Add associations to a matrix to pass into the BAM:
    data_pairs.append([a_1, b_1])
    data_pairs.append([a_2, b_2])
    data_pairs.append([a_3, b_3])
    data_pairs.append([a_4, b_4])
    
    b = BAM(data_pairs)

    b.print_matrix()

    # Distorted a_2 vector to feed to NN
    distort = np.array([-1, -1, -1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, 1])

    print("Distorted A2 Input = ", distort)
    print("")
    print("Memory response from distored A2 vector:")
    res = b.get_assoc(distort)
    print("")
    print("Response vector = ", res)
    print("")

