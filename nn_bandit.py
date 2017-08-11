# coding:utf-8
import time
import numpy as np
import tensorflow as tf
import pandas as pd

# keras
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import regularizers

data_dir = './hetrec2011-lastfm-2k/'

'''
Algorithm 1: Neural Bandit 1


initialize a mulitlayer perceptron A_k for each action in action set K
choose exploration parameter epsilon
for t = 1, 2, ..., T:
    observe context x_t 
    for k in K:
        predict y_k from x_t using A_k
    perform a Bernoulli trail with success probability epsilon
    if success:
        play arm with the highest predicted reward
    else
        play a random arm 
    perform a training step on the arm played
'''

class Bandit_SGD(SGD):
    """Stochastic gradient descent optimizer for contextual bandits.
    Includes support for momentum, learning rate decay, and Nesterov momentum.
    # Arguments
        n_arms: int >0. Number of arms.
        explore: float [0., .5] Exploration parameter.
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter updates momentum.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """
    
    def __init__(self, n_arms=2., explore=.1, **kwargs):
        super(Bandit_SGD, self).__init__(**kwargs)
        self.n_arms = K.variable(n_arms, name='n_arms')
        self.explore = K.variable(explore, name='explore')
    
    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = []

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))
            self.updates .append(K.update_add(self.iterations, 1))
        
        # weight scaling for bandits
        P = (1. - self.explore) + self.explore / self.n_arms
        # momentum
        shapes = [K.get_variable_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            # apply bandit scaling
            g =  g/P
            v = self.momentum * m - lr * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = (p + self.momentum * v - lr * g)
            else:
                new_p = (p + v)

            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

class Bandit_Adam(Adam):
    """Adam optimizer for contextual bandits
    Default parameters follow those provided in the original paper.
    # Arguments
        n_arms: int >0. Number of arms.
        explore: float [0., .5] Exploration parameter.
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    """

    def __init__(self, n_arms=2., explore=.1, **kwargs):
        super(Bandit_Adam, self).__init__(**kwargs)
        self.n_arms = K.variable(n_arms, name='n_arms')
        self.explore = K.variable(explore, name='explore')
        

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        t = self.iterations + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        shapes = [K.get_variable_shape(p) for p in params]
        ms = [K.zeros(shape) for shape in shapes]
        vs = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + ms + vs
        # weight scaling for bandits
        P = (1. - self.explore)*(loss > -0.6931471805599453) + self.explore / self.n_arms
        
        for p, g, m, v in zip(params, grads, ms, vs):
            # apply bandit scaling
            g = g/P
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))

            new_p = p_t
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append(K.update(p, new_p))
        return self.updates

def read_dataset():
    # step1----------read dataset
    header = ['userID', 'artistID', 'weight']
    df = pd.read_csv(data_dir + 'user_artists.dat', sep = '\t', names = header)

    users = df.userID.unique()
    items = df.artistID.unique()

    n_users = users.shape[0]
    n_items = items.shape[0]

    # print type(users)
    # print users
    # print items
    # print df.head(3)
    # print df.describe()
    print n_users, n_items
    return df, users, items, n_users, n_items

def build_experts(n, input_shape, n_hidden, n_layers):
    # builds a committee of experts
    def build_expert():
        model = Sequential()
        # add hidden layers
        for layer in range(n_layers):
            # Keras 1.2 doesn't support separate kernel/bias initializers, layers only take a single init parameter
            # model.add(Dense(n_hidden,
            #                 kernel_initializer='glorot_uniform',
            #                 activation='relu',
            #                 input_dim=input_shape,
            #                 kernel_regularizer=regularizers.l2(0.01)))
            model.add(Dense(n_hidden,
                            init='glorot_uniform',
                            activation='relu',
                            input_dim=input_shape))
        # output layer
        model.add(Dense(1,
                        init='glorot_normal',
                        activation='sigmoid'))
        return model
    experts = [build_expert() for i in range(n)]
    return experts

def compile_experts(experts, optimizer, loss, **kwargs):
    # compiles a commitee of experts
    n_arms = len(experts)
    def compile_expert(expert, **kwargs):
        expert.compile(optimizer=optimizer,
                      loss=loss)
        return expert
    compiled_experts = [compile_expert(expert) for expert in experts]
    return compiled_experts

# chooses an arm as in Algorithm 1
def choose_arm(x, experts, explore):
    n_arms = len(experts)
    # make predictions
    preds = [expert.predict(x) for expert in experts]
    # get best arm
    arm_max = np.nanargmax(preds)
    # create arm selection probabilities
    P = [(1-explore)*(arm==arm_max) + explore/n_arms for arm in range(n_arms)]
    # select an arm
    chosen_arm = np.random.choice(np.arange(n_arms), p=P)
    pred = preds[chosen_arm]
    return chosen_arm, pred

def run_bandit_1(X, Y, explore, exp_annealing_rate=1, min_explore=.005, **kwargs):
    n, n_arms = Y.shape
    input_shape = X.shape[1]
    experts = build_experts(n_arms, input_shape, 32, 1)
    experts = compile_experts(experts, **kwargs)
    # trace for arm choices
    chosen_arms = []
    # trace for regrets
    regrets = []
    true_rewards = []
    
    start_time = time.time()
    message_iteration = 10
    print 'Starting bandit\n----------\nN_arms: %d \n----------\n'%n_arms

    for i in range(n):
        context = X[[i]]
        chosen_arm, pred = choose_arm(context, experts, explore)
        reward = Y[i, chosen_arm]
        max_reward = np.max(Y[i])
        max_arm = np.argmax(Y[i])
        true_rewards.append(max_arm)
        expert = experts[chosen_arm]
        expert.fit(context, np.expand_dims(reward, axis=0), verbose=0)
        experts[chosen_arm] = expert
        chosen_arms.append(chosen_arm)
        regret = max_reward - reward
        regrets.append(regret)
        if explore > min_explore:
            explore *= exp_annealing_rate
        if (i % message_iteration == 0) and (i > 0):
            if message_iteration <= 1e4:
                message_iteration *= 10
            elapsed = time.time() - start_time
            remaining = (n*elapsed/i - elapsed)/60
            print '''Completed iteration: %d
            Elapsed time: %.2f seconds
            Estimated time remaining: %.2f minutes
            --------------------'''%(i,elapsed,remaining)
    elapsed = (time.time() - start_time)/60
    print 'Finished in: %.2f minutes'%elapsed
    return experts, chosen_arms, true_rewards, regrets


def run_bandit_2(X, Y):
    # neural bandit 2
    # get shapes and number of arms
    n, n_arms = Y.shape
    input_shape = X.shape[1]

    # init models
    # 32 hidden units, 1 hidden layer, explore = .005
    model_1 = build_experts(n_arms, input_shape, n_hidden=32, n_layers=1)
    model_1 = compile_experts(model_1, loss='binary_crossentropy', optimizer='adam')

    # 64 hidden units, 1 hidden layer, explore = .005
    model_2 = build_experts(n_arms, input_shape, n_hidden=64, n_layers=1)
    model_2 = compile_experts(model_2, loss='binary_crossentropy', optimizer='adam')

    # 128 hidden units, 1 hidden layer, explore = .005
    model_3 = build_experts(n_arms, input_shape, n_hidden=128, n_layers=1)
    model_3 = compile_experts(model_3, loss='binary_crossentropy', optimizer='adam')


    # 64 hidden units, 2 hidden layers, explore = .005
    model_4 = build_experts(n_arms, input_shape, n_hidden=64, n_layers=2)
    model_4 = compile_experts(model_4, loss='binary_crossentropy', optimizer='adam')


    # 64 hidden units, 2 hidden layers, explore = .005
    model_5 = build_experts(n_arms, input_shape, n_hidden=128, n_layers=2)
    model_5 = compile_experts(model_5, loss='binary_crossentropy', optimizer='adam')


    # 32 hidden units, 1 hidden layer, annealing_explore
    model_6 = build_experts(n_arms, input_shape, n_hidden=32, n_layers=1)
    model_6 = compile_experts(model_6, loss='binary_crossentropy', optimizer='adam')

    # 64 hidden units, 1 hidden layer, annealing_explore
    model_7 = build_experts(n_arms, input_shape, n_hidden=64, n_layers=1)
    model_7 = compile_experts(model_7, loss='binary_crossentropy', optimizer='adam')

    # 128 hidden units, 1 hidden layer, annealing_explore
    model_8 = build_experts(n_arms, input_shape, n_hidden=128, n_layers=1)
    model_8 = compile_experts(model_8, loss='binary_crossentropy', optimizer='adam')


    # 64 hidden units, 2 hidden layers, annealing_explore
    model_9 = build_experts(n_arms, input_shape, n_hidden=64, n_layers=2)
    model_9 = compile_experts(model_9, loss='binary_crossentropy', optimizer='adam')


    # 64 hidden units, 2 hidden layers, annealing_explore
    model_10 = build_experts(n_arms, input_shape, n_hidden=128, n_layers=2)
    model_10 = compile_experts(model_10, loss='binary_crossentropy', optimizer='adam')

def get_model_probabilities(weights, gamma_model):
    # get probabilites of choosing each model
    p = np.array([(1-gamma_model)*weight/sum(weights) + gamma_model/n_models for weight in weights])
    return p

def choose_model(weights, gamma_model, model_probabilities):
    # choose a model based on weights
    n_models = len(weights)
    model = np.random.choice(np.arange(n_models), p=model_probabilities)
    return model

if __name__ == "__main__":
    df, users, items, n_users, n_items = read_dataset()
    print len(df)
    X = df.values[-20:,:]  #remove the first line(the colume description)
    
    print X.shape[1]
    print X.shape
    
    n_round = 20
    n_arms = 7
    Y = np.zeros((n_round, n_arms))

    # experts = build_experts(4, X.shape[1], 32, 1)
    # experts[0].summary()

    # sample run bandit_1
    # n_points = 100

    # fit_models_1, arm_hist_1, true_reward_hist_1, regret_hist_1 = run_bandit_1(X[:n_points], Y[:n_points], optimizer='adam', loss='binary_crossentropy', explore=.005, exp_annealing_rate=1, clipnorm=1.)

    # smaple run bandit_2
        # neural bandit 2
    # get shapes and number of arms
    n, n_arms = Y.shape
    input_shape = X.shape[1]

    # init models
    # 32 hidden units, 1 hidden layer, explore = .005
    model_1 = build_experts(n_arms, input_shape, n_hidden=32, n_layers=1)
    model_1 = compile_experts(model_1, loss='binary_crossentropy', optimizer='adam')

    # 64 hidden units, 1 hidden layer, explore = .005
    model_2 = build_experts(n_arms, input_shape, n_hidden=64, n_layers=1)
    model_2 = compile_experts(model_2, loss='binary_crossentropy', optimizer='adam')

    # 128 hidden units, 1 hidden layer, explore = .005
    model_3 = build_experts(n_arms, input_shape, n_hidden=128, n_layers=1)
    model_3 = compile_experts(model_3, loss='binary_crossentropy', optimizer='adam')


    # 64 hidden units, 2 hidden layers, explore = .005
    model_4 = build_experts(n_arms, input_shape, n_hidden=64, n_layers=2)
    model_4 = compile_experts(model_4, loss='binary_crossentropy', optimizer='adam')


    # 64 hidden units, 2 hidden layers, explore = .005
    model_5 = build_experts(n_arms, input_shape, n_hidden=128, n_layers=2)
    model_5 = compile_experts(model_5, loss='binary_crossentropy', optimizer='adam')


    # 32 hidden units, 1 hidden layer, annealing_explore
    model_6 = build_experts(n_arms, input_shape, n_hidden=32, n_layers=1)
    model_6 = compile_experts(model_6, loss='binary_crossentropy', optimizer='adam')

    # 64 hidden units, 1 hidden layer, annealing_explore
    model_7 = build_experts(n_arms, input_shape, n_hidden=64, n_layers=1)
    model_7 = compile_experts(model_7, loss='binary_crossentropy', optimizer='adam')

    # 128 hidden units, 1 hidden layer, annealing_explore
    model_8 = build_experts(n_arms, input_shape, n_hidden=128, n_layers=1)
    model_8 = compile_experts(model_8, loss='binary_crossentropy', optimizer='adam')


    # 64 hidden units, 2 hidden layers, annealing_explore
    model_9 = build_experts(n_arms, input_shape, n_hidden=64, n_layers=2)
    model_9 = compile_experts(model_9, loss='binary_crossentropy', optimizer='adam')


    # 64 hidden units, 2 hidden layers, annealing_explore
    model_10 = build_experts(n_arms, input_shape, n_hidden=128, n_layers=2)
    model_10 = compile_experts(model_10, loss='binary_crossentropy', optimizer='adam')

    # set n_steps and model exploration parameter
    n_steps = 100
    gamma_model=.1

    # collect models, only 4 in the interest of time
    models = [model_2, model_4, model_7, model_9]
    n_models = len(models)
    # init weight vector
    weights = np.ones(n_models)
    # init model explore parameters
    explores = np.array([.005]*2 + [.5]*2)
    anneal = np.array([False]*2 + [True]*2)
    annealing_rate = .99995
    min_explore = .005
    # init histories
    arm_hist_2 = []
    model_hist_2 = []
    regret_hist_2 = []
    weight_hist_2 = []

    # init timing vars
    start_time = time.time()
    next_check = 1

    # train the models
    for step in range(n_steps):
        # store weights
        weight_hist_2.append(weights)
        # get probs and choose a model
        p = get_model_probabilities(weights, gamma_model)
        chosen_model = choose_model(weights, gamma_model, p)
        # store model choice
        model_hist_2.append(chosen_model)
        # get a random data point
        i = np.random.randint(X.shape[0])
        context = X[[i]]
        # choose an arm
        chosen_arm, pred = choose_arm(context, models[chosen_model], explores[chosen_model])
        # store arm selection
        arm_hist_2.append(chosen_arm)
        # observe reward and max reward
        reward = Y[i, chosen_arm]
        max_reward = np.max(Y[i])
        # calculate and store regret
        regret = max_reward - reward
        regret_hist_2.append(regret)
        # update the chosen arm for each model
        for m, model in enumerate(models):
            expert = model[chosen_arm]
            expert.fit(context, np.expand_dims(reward, axis=0), verbose=0)
            model[chosen_arm] = expert
            # anneal explore param if necessary
            if (anneal[m]) and (explores[m] > min_explore):
                explores[m] *= annealing_rate
        # update weights
        weights[chosen_model] = weights[chosen_model]*np.exp((gamma_model*reward/(p[chosen_model]*n_models)))
        # print progress
        if step == next_check:
            elapsed = time.time()-start_time
            print 'Step %d complete in %.2f seconds.'%(step,elapsed)
            next_check *= 2