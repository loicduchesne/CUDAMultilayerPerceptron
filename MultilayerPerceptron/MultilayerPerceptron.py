import cupy as cp

class MLP:

    def __init__(self, activation_function_hidden : str, activation_function_output : str, weight_initialization_fn='zero',
                 leaky_coeff=None, hidden_layers=None, learning_rate_hidden=0.01, learning_rate_output=0.01, max_epochs=60,
                 batch_size=1, term_condition=1e-8, L1_reg = 0, L2_reg = 0, add_bias=True, random_state=None,
                 record_updates=False, verbose=False, dtype='float64'):
        ## Activation functions (& their derivatives)
        self.activation_functions = {'identity': self._identity, 'relu': self._ReLU, 'leaky': self._leaky_ReLU,
                                     'logistic': self._logistic, 'softmax': self._softmax, 'tanh': self._tanh}
        self.d_activation_functions = {'identity': self._d_identity, 'relu': self._d_ReLU, 'leaky': self._d_leaky_ReLU,
                                       'logistic': self._d_logistic, 'softmax': self._d_softmax, 'tanh': self._d_tanh}

        self.h = self.activation_functions[activation_function_hidden.lower()]
        self.g = self.activation_functions[activation_function_output.lower()]

        self.d_h = self.d_activation_functions[activation_function_hidden.lower()]
        self.d_g = self.d_activation_functions[activation_function_output.lower()]

        # Activation specific config
        if activation_function_hidden.lower() == 'leaky':
            if leaky_coeff == None:
                raise ValueError('A coefficient needs to be specified for the leaky ReLU algorithm')
            else:
                self.a = leaky_coeff

        # Loss initialization
        if self.g == self._logistic:
            self.L = self._CE
            self.d_L = self._d_CE

        elif self.g == self._softmax: # If True: classification problem
            self.L = self._CE
            self.d_L = self._d_CE

        else: # Regression
            self.L = self._MSE
            self.d_L = self._d_MSE

        ## Weight initialization
        self.weight_initialization_functions = {'zero' : self._Zero, 'uniform' : self._Uniform, 'normal' : self._Normal,
                                                'xavier': self._Xavier, 'kaiming': self._Kaiming}
        self.wv_init_fn = self.weight_initialization_functions[weight_initialization_fn.lower()]

        ## Hidden layers
        self.hidden_layers = hidden_layers

        ## Config
        self.b_s = batch_size
        self.lr_V = learning_rate_hidden
        self.lr_W = learning_rate_output
        self.max_epochs = max_epochs
        self.term_condition = term_condition
        self.add_bias = add_bias
        self.record_updates = record_updates

        # Random state
        self.rng = cp.random.RandomState(random_state)

        # Regularization
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg

        # Verbose
        self.verbose = verbose

        # Loss history
        if record_updates:
            self.w_hist = [] # records the weight
            self.loss_hist = []
            self.loss_hist_avg = []
        else:
            self.w_hist = None
            self.loss_hist = None
            self.loss_hist_avg = None

        # dtype
        self.dtype = dtype


    def fit(self, X, Y):
        # Numpy utilities
        try:
            X = X.to_numpy()
            Y = Y.to_numpy()
        except AttributeError:
            pass
        # Cuda
        try:
            X = cp.asarray(X, dtype=self.dtype)
            Y = cp.asarray(Y, dtype=self.dtype)
        except AttributeError:
            pass

        if Y.ndim == 1:
            Y = Y[:, None]

        # Initialize history
        if self.record_updates:
            self.w_hist = []
            self.loss_hist = []
            self.loss_hist_avg = []

        # Initialize weights
        self._initialize(X, Y)

        for _ in range(self.max_epochs): # Epoch
            # Stochastic
            X, Y = self.shuffle(X, Y)

            ## BATCH
            L_sum = 0

            for i in range(0, X.shape[0], self.b_s):
                X_i = X[i: i+self.b_s, :] # X for this batch
                Y_i = Y[i: i+self.b_s] # Y for this batch

                # Forward pass
                Yh_i = self._forwardprop(X_i)

                # Loss
                L = self.L(Y_i, Yh_i)
                L_sum += L # Add batch loss

                if self.record_updates:
                    self.loss_hist.append(L)
                    if self.record_updates and _ == 0 and i == 0:
                        self.loss_hist_avg.append(L_sum)


                # Backpropagation
                dW, dV, dB = self._backprop(X_i, Y_i, Yh_i)

                # Update weights & biases
                self.W -= self.lr_W * dW
                if self.add_bias:
                    self.B[-1] -= self.lr_W * dB[-1]

                for j in range(len(self.V)):
                    self.V[j] -= self.lr_V * dV[j]
                    if self.add_bias:
                        self.B[j] -= self.lr_V * dB[j]

                # Regularization
                m = X_i.shape[0]
                self.W -= (self.lr_W * self.L1_reg/m) * cp.sign(self.W)
                self.W *= 1 - self.lr_W * self.L2_reg/m

                for j in range(len(self.V)):
                    self.V[j] -= (self.lr_V * self.L1_reg/m) * cp.sign(self.V[j])
                    self.V[j] *= 1 - self.lr_V * self.L2_reg/m

            # History
            if self.record_updates:
                self.w_hist.append(self.W)
                self.loss_hist_avg.append(L_sum/X.shape[0]*self.b_s)

            if self.verbose == True:
                print(str(_+1) + " epoch")

        # History
        return self.w_hist, self.loss_hist_avg, self.loss_hist


    def predict(self, X):
        # Numpy utilities
        try:
            X = X.to_numpy()
        except AttributeError:
            pass
        # Cuda
        try:
            X = cp.asarray(X)
        except AttributeError:
            pass

        A_i = X # Activations

        for i in range(len(self.hidden_layers)):
            # Propagation
            if self.add_bias:
                A_i = self.h(A_i @ self.V[i] + self.B[i])
            else:
                A_i = self.h(A_i @ self.V[i])

        if self.add_bias:
            if (self.g == self._logistic or self.g == self._softmax):
                return self.prob_to_max(self.g(A_i @ self.W + self.B[-1])).get()
            else:
                return self.prob_to_class(self.g(A_i @ self.W + self.B[-1])).get()
        else:
            if (self.g == self._logistic or self.g == self._softmax):
                return self.prob_to_max(self.g(A_i @ self.W)).get()
            else:
                return self.prob_to_class(self.g(A_i @ self.W)).get()

    def proba(self, X):
        # Numpy utilities
        try:
            X = X.to_numpy()
        except AttributeError:
            pass
        # Cuda
        try:
            X = cp.asarray(X)
        except AttributeError:
            pass

        A_i = X # Activations

        for i in range(len(self.hidden_layers)):
            # Propagation
            if self.add_bias:
                A_i = self.h(A_i @ self.V[i] + self.B[i])
            else:
                A_i = self.h(A_i @ self.V[i])

        if self.add_bias:
            return self.g(A_i @ self.W + self.B[-1])
        else:
            return self.g(A_i @ self.W)

    def _forwardprop(self, X):
        # Initializations for forward prop.
        self.Z = [] # Z: PRE ACTIVATED VALUES
        self.A = [] # A: POST ACTIVATED VALUES

        A_i = X # Activations
        self.A.append(A_i)

        for i in range(len(self.hidden_layers)):
            # Propagation
            if self.add_bias:
                Z_i = A_i @ self.V[i] + self.B[i]
            else:
                Z_i = A_i @ self.V[i]

            self.Z.append(Z_i) # Pre-activated
            A_i = self.h(Z_i)
            self.A.append(A_i) # Post-activated

        if self.add_bias:
            return self.g(A_i @ self.W + self.B[-1])
        else:
            return self.g(A_i @ self.W)

    def _backprop(self, X, Y, Yh):
        N = X.shape[0]
        dB = []
        dV = []

        # Output error
        dY = Yh - Y

        if self.add_bias:
            dB.append(cp.mean(dY, axis=0, keepdims=True))
        else:
            pass

        # Weight error
        if self.hidden_layers in [[], None]:
            dW = 1/N*(X.T @ dY)
        else:
            dW = 1/N*(self.A[-1].T @ dY)

        # Hidden neuron errors
        dZ = dY
        # Hidden errors
        for i in reversed(range(len(self.hidden_layers))):
            if i == len(self.hidden_layers)-1:
                dZ = (dZ @ self.W.T) * self.d_h(self.Z[i])
            else:
                dZ = (dZ @ self.V[i+1].T) * self.d_h(self.Z[i])
            if self.add_bias: # Bias Error
                dB.insert(0, cp.mean(dZ, axis=0, keepdims=True))

            dV.insert(0, 1/N * (self.A[i].T @ dZ))

        return dW, dV, dB


    def _initialize(self, X, Y):
        self.V = []
        self.B = [] # Biases

        ## X (N x D)
        ## Y (N x C)
        ## V (D x M)
        ## W (M x C)

        # V[0]: (D x hidden_layer[0])
        # V[1]: (hidden_layer [0] x hidden_layer[1])
        # V[2]: (hidden_layer[1] x hidden_layer[2])
        # V[n]: (hidden_layer[n-1] x hidden_layer[n]) where n=len(hidden_layer)

        # W: (hidden_layer[n] x C)
        #input layer
        if self.hidden_layers in [[], None]:
            self.W = self.wv_init_fn(X.shape[1], Y.shape[1])

            if self.add_bias:
                self.B.append(cp.zeros((1, Y.shape[1])))
            return None

        self.V.append(self.wv_init_fn(X.shape[1], self.hidden_layers[0]))

        if self.add_bias:
            self.B.append(cp.zeros((1, self.hidden_layers[0])))

        #hidden layers
        for i in range(len(self.hidden_layers) - 1): # -1 to ignore hidden -> output layer
            self.V.append(self.wv_init_fn(self.hidden_layers[i], self.hidden_layers[i+1]))

            if self.add_bias:
                self.B.append(cp.zeros((1, self.hidden_layers[i+1])))

        ## W
        self.W = self.wv_init_fn(self.hidden_layers[-1], Y.shape[1])

        if self.add_bias:
            self.B.append(cp.zeros((1, Y.shape[1])))


    ## WEIGHT INITIALIZATION
    def _Zero(self, row, col):
        return cp.zeros((row, col), dtype=self.dtype)

    def _Uniform(self, row, col):
        return self.rng.uniform(low=-1, high=1, size=[row, col], dtype=self.dtype)

    def _Normal(self, row, col):
        return self.rng.normal(loc= 0, size = [row, col], dtype=self.dtype)

    def _Xavier(self, row, col):
        a = cp.sqrt(2 / (row + col))
        return self.rng.normal(loc=0, scale=a, size=[row, col], dtype=self.dtype)

    def _Kaiming(self, row, col):
        a = cp.sqrt(2 / col)
        return self.rng.normal(loc=0, scale=a, size=[row, col], dtype=self.dtype)


    ## ACTIVATION FUNCTIONS
    def _identity(self, X):
        return X

    def _d_identity(self, X):
        return cp.ones(X.shape, dtype=self.dtype)

    def _ReLU(self, X):
        return cp.maximum(0, X, dtype=self.dtype)

    def _d_ReLU(self, X):
        return cp.heaviside(X, 0, dtype=self.dtype)

    def _leaky_ReLU(self, X):
        return cp.maximum(self.a * X, X, dtype=self.dtype)

    def _d_leaky_ReLU(self, X):
        return cp.heaviside(X, self.a, dtype=self.dtype)

    def _logistic(self, X):
        return 1.0/(1.0 + cp.exp(-1.0 * X))

    def _d_logistic(self, X):
        return self._logistic(X)*(1.0 - self._logistic(X))

    def _softmax(self, X):
        e_X = cp.exp(X - cp.max(X), dtype=self.dtype)
        return e_X / e_X.sum(axis=1, keepdims=True, dtype=self.dtype)

    def _d_softmax(self, X):
        return X # *Note: We never use softmax in the middle layers, therefore this should never be called.

    def _tanh(self, X):
        return cp.tanh(X)

    def _d_tanh(self, X):
        return 1 - cp.tanh(X)**2


    ## LOSS FUNCTIONS
    def _MSE(self, Y, Yh):
        N = Y.shape[0]
        return (1 / N * cp.sum((Y - Yh) ** 2))

    def _d_MSE(self, Y, Yh):
        N = Y.shape[0]
        return -2/N * (Y - Yh)

    def _MSElog(self, Y, Yh):
        # Numpy utilities
        # try:
        #     X = cp.asarray(X)
        #     Y = cp.asarray(Y)
        # except AttributeError:
        #     pass
        pass

    def _CE(self, Y, Yh):
        N = Y.shape[0]
        return -cp.sum(Y * cp.log(Yh + 1e-15))/N # Add small value to avoid log(0)

    def _d_CE(self, Y, Yh):
        return Yh - Y


    ## UTILITY
    def shuffle(self, data, target):
        assert len(data) == len(target)
        p = self.rng.permutation(len(data))
        return data[p], target[p]

    def prob_to_max(self, arr):
        result = cp.zeros_like(arr)

        for i, row in enumerate(arr):
            max_index = cp.argmax(row)
            result[i][max_index] = 1

        return result

    def prob_to_class(self, arr):
        return cp.rint(arr)


# Evaluate model accuracy
def evaluate_acc(Y, Yh) -> float:
    Y.shape = Yh.shape
    accuracy = cp.mean(cp.all(Y == Yh, axis=1))
    return accuracy
