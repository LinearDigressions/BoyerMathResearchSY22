import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class RBF():

    def __init__(self, signal_variance=1, length_scale=1):
        self.signal_variance = signal_variance
        self.length_scale = length_scale 
        self.num_parameters = 2

    def covariance_function(self, x1, x2):
        return self.signal_variance * np.exp(-0.5 * (x1 - x2)**2 / self.length_scale**2)
    
    def gradient_signal_variance(self, x1, x2):
        return np.exp(-0.5 * (x1 - x2)**2 / self.length_scale**2)
    
    def gradient_length_scale(self, x1, x2):
        return 2 * (x1 - x2)**2 * self.length_scale**-3 * self.covariance_function(x1,x2)
    
    def update_signal_variance(self, new_signal_variance):
        self.signal_variance = new_signal_variance
    
    def update_length_scale(self, new_length_scale):
        self.length_scale = new_length_scale

    def get_gradient_functions(self):
        signal_variance_dict = {'name':"signal_variance", 
                                'value': self.signal_variance,
                                "partial_derivative":self.gradient_signal_variance,
                                "update": self.update_signal_variance}
    
        length_scale_dict = {'name':"length_scale", 
                             'value': self.length_scale,
                            "partial_derivative":self.gradient_length_scale,
                            "update": self.update_length_scale}
        
        return [signal_variance_dict, length_scale_dict]


        
class GaussianProcessesRegression():

    def __init__(self, kernel):

        self.covariance_matrix = None
        self.kernel = kernel
        self.alpha = None
        self.K = None
       
    
 
    def generate_covariance_matrix(self, x, y, covariance_function):
        
        n1 = x.size
        n2 = y.size

        covariance_matrix = np.zeros((n1,n2))

        for i in range(n1):
            for j in range(n2):
                covariance_matrix[i][j] = covariance_function(x[i], y[j])
        return covariance_matrix
    

    def fit(self, x_train, y_train, noise_level=None):
        self.x_train = x_train
        self.y_train = y_train
        self.n = x_train.shape[0]

        self.K = self.generate_covariance_matrix(x_train,x_train, self.kernel.covariance_function)

        if noise_level != None:
            self.K = self.K + noise_level * np.eye(self.n)

        self.L = np.linalg.cholesky(self.K)
        self.alpha = np.linalg.solve(np.transpose(self.L), np.linalg.solve(self.L, y_train))

        self.K_inverse = np.linalg.inv(self.K)
        self.gradient_pre = np.outer(self.alpha,self.alpha) - self.K_inverse


        return(-0.5 * y_train.dot(self.alpha) - np.trace(self.L) - 0.5 * self.n * np.log(2 * np.pi))

    def predict(self, x_test):

        predictions = np.zeros((x_test.shape[0], 2))

        for i in range (x_test.shape[0]):
            x_test_i = np.array([x_test[i]])
            k  = self.generate_covariance_matrix(self.x_train, x_test_i, self.kernel.covariance_function).flatten()

            predictions[i,0]
            
            predictions[i,0] = k.dot(self.alpha)

            v = np.linalg.solve(self.L, k)
            

            predictions[i,1] = self.kernel.covariance_function(x_test_i, x_test_i) - v.dot(v)

        return predictions
    
    def gradient(self):

        gradient = np.zeros(self.kernel.num_parameters)
        kernel_parameters = self.kernel.get_gradient_functions()
        
        for i in range(self.kernel.num_parameters):

            ith_parameter_partial_derivative = kernel_parameters[i]["partial_derivative"]
            
            K_gradient = self.generate_covariance_matrix(self.x_train, self.x_train, ith_parameter_partial_derivative)

            ith_gradient = 0.5 * np.trace(np.matmul(self.gradient_pre, K_gradient))

            gradient[i] = ith_gradient

        return gradient
    
    # def plot_mle(self):
    #     x = np.linspace(0,1,50)
    #     y = np.linspace(-.5,.5,50)
    #     z = np.zeros((x.shape[0], y.shape[0]))

    #     for i in range(x.shape[0]):
    #         for j in range(y.shape[0]):
                
    #             self.kernel.update_length_scale(x[i])
    #             self.kernel.update_signal_variance(y[j])
              
    #             z[i][j] = self.fit(self.x_train, self.y_train)
            

    #     plt.contour(x, y, z)
    #     plt.show()
    
    def optimize(self, learning_rate = 0.01, max_iterations = 100):
                
        kernel_parameters = self.kernel.get_gradient_functions()

        for i in range(max_iterations):

            new_gradient = self.gradient()

            j = 0
            for parameter in kernel_parameters:
                new_value = parameter["value"] - learning_rate * new_gradient[j]
                parameter["update"](new_value)
                j += 1

            if np.linalg.norm(new_gradient) < 0.005:
                break
        
                



    def plot_gradient_field(self):
        
        x,y = np.meshgrid(np.linspace(0,1,50),np.linspace(-.5,.5,50))
        
        u = np.zeros(x.shape)
        v = np.zeros(x.shape)

        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                
                self.kernel.update_length_scale(x[i][j])
                self.kernel.update_signal_variance(y[i][j])
              
                result = self.gradient()
                u[i][j] = result[0]
                v[i][j] = result[1]

        plt.quiver(x, y, u, v)
        plt.show()







n = 8

x_test = np.linspace(start=0, stop=1, num=n)
x_train = np.array([0.25, 0.5, 0.75])

y_train = (x_train)**2

rbf_kernel = RBF()
gp_model = GaussianProcessesRegression(rbf_kernel)


gp_model.fit(x_train, y_train)

# gp_model.plot_mle()
print(rbf_kernel.signal_variance)
print(rbf_kernel.length_scale)
gp_model.optimize()
print(rbf_kernel.signal_variance)
print(rbf_kernel.length_scale)


# print(gp_model.gradient())


# log_marginal_likelihood= gp_model.fit(x_train, y_train)
# print("LML: " + str(log_marginal_likelihood))
# predictions = gp_model.predict(x_test)
# print(predictions)
# print(predictions[:,0])


# plt.scatter(x_test, predictions[:,0])
# plt.show()




# First draft regression
# def gp_regression(X, y, x_test):

#     n = X.shape[0]

#     K = generate_covariance_matrix(X,X)
#     k = generate_covariance_matrix(x_test, X)

#     L = np.linalg.cholesky(K)

#     alpha = np.linalg.solve(np.transpose(L), np.linalg.solve(L, y))

#     f_test_mean = k.dot(alpha)

#     v = np.linalg.solve(L, k)

#     f_test_var = covariance(x_test, x_test) - v.dot(v)

#     log_marginal_liklihood = -0.5 * y.dot(alpha) - np.trace(L) - 0.5 * n * np.log(2 * np.pi)

#     return f_test_mean, f_test_var

