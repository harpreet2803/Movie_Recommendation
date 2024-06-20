import pandas as pd, numpy as np, random, csv, time
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import keras.backend as K
from keras import Sequential, Input
from keras.layers import Dense, Dropout

class DataGenerator():
  def __init__(self, datapath, itempath):
    '''
    data: Load data from the DB MovieLens with movietitles
    users & items: List the users and the items
    histo: List all the users historic ratings in timeseries order
    '''
    self.data  = self.load_datas(datapath, itempath)
    self.users = self.data['userId'].unique()   #list of all users
    self.items = self.data['itemId'].unique()   #list of all items
    self.histo = self.gen_histo()
    self.train = []
    self.test  = []

  def load_datas(self, datapath, itempath):
    '''
    Load the data and merge the name of each movie. 
    A row corresponds to a rate given by a user to a movie.
    '''
    data = pd.read_csv(datapath, sep = '\t', names = ['userId', 'itemId', 'rating', 'timestamp'])
    movie_titles = pd.read_csv(itempath, sep='|', names = ['itemId', 'itemName'], usecols = range(2), encoding = 'latin-1')
    return data.merge(movie_titles, on = 'itemId', how = 'left')


  def gen_histo(self):
    '''
    Group all rates given by users and store them from older to most recent.
    '''
    historic_users = []
    for i, u in enumerate(self.users):
      temp = self.data[self.data['userId'] == u]
      temp = temp.sort_values('timestamp').reset_index()
      temp.drop('index', axis = 1, inplace = True)
      historic_users.append(temp)
    return historic_users

  def sample_histo(self, user_histo, action_ratio = 0.8, max_samp_by_user = 5,  max_state = 100, max_action = 50, nb_states = [], nb_actions = []):
    '''
    For a given historic, make one or multiple sampling.
    If no optional argument given for nb_states and nb_actions, then the sampling
    is random and each sample can have differents size for action and state.
    To normalize sampling we need to give list of the numbers of states and actions
    to be sampled.
    '''
    n = len(user_histo)
    sep = int(action_ratio * n)
    nb_sample = random.randint(1, max_samp_by_user)
    if not nb_states:
      nb_states = [min(random.randint(1, sep), max_state) for i in range(nb_sample)]
    if not nb_actions:
      nb_actions = [min(random.randint(1, n - sep), max_action) for i in range(nb_sample)]
    assert len(nb_states) == len(nb_actions), 'Given array must have the same size'
    
    states  = []
    actions = []
    for i in range(len(nb_states)):
      sample_states = user_histo.iloc[0:sep].sample(nb_states[i])
      sample_actions = user_histo.iloc[-(n - sep):].sample(nb_actions[i])
      
      sample_state = []
      sample_action = []
      for j in range(nb_states[i]):
        row = sample_states.iloc[j]
        state = str(row.loc['itemId']) + '&' + str(row.loc['rating'])
        sample_state.append(state)
      
      for j in range(nb_actions[i]):
        row = sample_actions.iloc[j]
        action = str(row.loc['itemId']) + '&' + str(row.loc['rating'])
        sample_action.append(action)

      states.append(sample_state)
      actions.append(sample_action)
    return states, actions

  def gen_train_test(self, test_ratio, seed = None):
    '''
    Shuffle the historic of users and separate it in a train and a test set.
    Store the ids for each set. A user can't be in both set.
    '''
    n = len(self.histo)

    if seed is not None:
      random.Random(seed).shuffle(self.histo)
    else:
      random.shuffle(self.histo)

    self.train = self.histo[:int((test_ratio * n))]
    self.test  = self.histo[int((test_ratio * n)):]
    self.user_train = [h.iloc[0,0] for h in self.train]
    self.user_test  = [h.iloc[0,0] for h in self.test]
    

  def write_csv(self, filename, histo_to_write, delimiter=';', action_ratio=0.8, max_samp_by_user=5, max_state=100, max_action=50, nb_states=[], nb_actions=[]):
    '''
    From  a given historic, create a csv file with the format:
    columns : state;action_reward;n_state
    rows    : itemid&rating1 | itemid&rating2 | ... ; itemid&rating3 | ... | itemid&rating4; itemid&rating1 | itemid&rating2 | itemid&rating3 | ... | item&rating4
    at filename location.
    '''
    with open(filename, mode = 'w') as file:
      f_writer = csv.writer(file, delimiter = delimiter)
      f_writer.writerow(['state', 'action_reward', 'n_state'])
      for user_histo in histo_to_write:
        states, actions = self.sample_histo(user_histo, action_ratio, max_samp_by_user, max_state, max_action, nb_states, nb_actions)
        for i in range(len(states)):
          # FORMAT STATE
          state_str   = '|'.join(states[i])
          # FORMAT ACTION
          action_str  = '|'.join(actions[i])
          # FORMAT N_STATE
          n_state_str = state_str + '|' + action_str
          f_writer.writerow([state_str, action_str, n_state_str])

class EmbeddingsGenerator:
  def  __init__(self, train_users, data):
    self.train_users = train_users
    self.data = data.sort_values(by=['timestamp'])
    self.data['userId'] = self.data['userId'] - 1
    self.data['itemId'] = self.data['itemId'] - 1
    self.user_count = self.data['userId'].max() + 1
    self.movie_count = self.data['itemId'].max() + 1
    self.user_movies = {} #list of rated movies by each user
    for userId in range(self.user_count):
      self.user_movies[userId] = self.data[self.data.userId == userId]['itemId'].tolist()
    self.m = self.model()

  def model(self, hidden_layer_size=100):
    m = Sequential()
    m.add(Dense(hidden_layer_size, input_shape=(1, self.movie_count)))
    m.add(Dropout(0.2))
    m.add(Dense(self.movie_count, activation='softmax'))
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return m
  
  def generate_input(self, user_id):
    '''
    Returns a context and a target for the user_id
    context: user's history with one random movie removed
    target: id of random removed movie
    '''
    user_movies_count = len(self.user_movies[user_id])
    random_index = np.random.randint(0, user_movies_count - 1) # -1 avoids taking the last movie
    target = np.zeros((1, self.movie_count))
    target[0][self.user_movies[user_id][random_index]] = 1
    context = np.zeros((1, self.movie_count))
    context[0][self.user_movies[user_id][:random_index] + self.user_movies[user_id][random_index + 1:]] = 1
    return context, target

  def train(self, nb_epochs = 300, batch_size = 10000):
    '''
    Trains the model from train_users's history
    '''
    for i in range(nb_epochs):
      print('%d/%d' % (i+1, nb_epochs))
      batch = [self.generate_input(user_id = np.random.choice(self.train_users) - 1) for _ in range(batch_size)]
      X_train = np.array([b[0] for b in batch])
      y_train = np.array([b[1] for b in batch])
      self.m.fit(X_train, y_train, epochs=1, validation_split=0.5)

  def test(self, test_users, batch_size = 100000):
    '''
    Returns [loss, accuracy] on the test set
    '''
    batch_test = [self.generate_input(user_id = np.random.choice(test_users) - 1) for _ in range(batch_size)]
    X_test = np.array([b[0] for b in batch_test])
    y_test = np.array([b[1] for b in batch_test])
    return self.m.evaluate(X_test, y_test)

  def save_embeddings(self, file_name):
    '''
    Generates a csv file containg the vector embedding for each movie.
    '''
    inp = self.m.layers[0].input                                 # input placeholder
    outputs = [layer.output for layer in self.m.layers]          # all layer outputs
    functor = K.function([inp, K.learning_phase()], outputs)     # evaluation function

    vectors = []
    for movie_id in range(self.movie_count):
      movie = np.zeros((1, 1, self.movie_count))
      movie[0][0][movie_id] = 1
      layer_outs = functor([movie])
      vector = [str(v) for v in layer_outs[0][0][0]]
      vector = '|'.join(vector)
      vectors.append([movie_id, vector])

    embeddings = pd.DataFrame(vectors, columns=['item_id', 'vectors']).astype({'item_id': 'int32'})
    embeddings.to_csv(file_name, sep=';', index=False)

class Embeddings:
  def __init__(self, item_embeddings):
    self.item_embeddings = item_embeddings
  
  def size(self):
    return self.item_embeddings.shape[1]
  
  def get_embedding_vector(self):
    return self.item_embeddings
  
  def get_embedding(self, item_index):
    return self.item_embeddings[item_index]

  def embed(self, item_list):
    return np.array([self.get_embedding(item - 1) for item in item_list])

def read_file(data_path):
  ''' Load data from train.csv or test.csv. '''

  data = pd.read_csv(data_path, sep = ';')
  for col in ['state', 'n_state', 'action_reward']:
    data[col] = [np.array([[np.int(k) for k in ee.split('&')] for ee in e.split('|')]) for e in data[col]]
  for col in ['state', 'n_state']:
    data[col] = [np.array([e[0] for e in l]) for l in data[col]]

  data['action'] = [[e[0] for e in l] for l in data['action_reward']]
  data['reward'] = [tuple(e[1] for e in l) for l in data['action_reward']]
  data.drop(columns = ['action_reward'], inplace = True)

  return data

def read_embeddings(embeddings_path):
  ''' Load embeddings (a vector for each item). '''
  
  embeddings = pd.read_csv(embeddings_path, sep=';')

  return np.array([[np.float64(k) for k in e.split('|')] for e in embeddings['vectors']])

class Environment():
  def __init__(self, data, embeddings, alpha, gamma, fixed_length):
    self.embedded_data = pd.DataFrame()
    self.embedded_data['state'] = [np.array([embeddings.get_embedding(item_id) for item_id in row['state']]) for _, row in data.iterrows()]
    self.embedded_data['action'] = [np.array([embeddings.get_embedding(item_id) for item_id in row['action']]) for _, row in data.iterrows()]
    self.embedded_data['reward'] = data['reward']

    self.alpha = alpha # α (alpha) in Equation (1)
    self.gamma = gamma # Γ (Gamma) in Equation (4)
    self.fixed_length = fixed_length
    self.current_state = self.reset()
    self.groups = self.get_groups()

  def reset(self):
    self.init_state = self.embedded_data['state'].sample(1).values[0]
    return self.init_state

  def step(self, actions):
    '''
    Compute reward and update state.
    '''
    simulated_rewards, cumulated_reward = self.simulate_rewards(self.current_state.reshape((1, -1)), actions.reshape((1, -1)))

    for k in range(len(simulated_rewards)):
      if simulated_rewards[k] > 0:
        self.current_state = np.append(self.current_state, [actions[k]], axis=0)
        if self.fixed_length:
          self.current_state = np.delete(self.current_state, 0, axis=0)

    return cumulated_reward, self.current_state

  def get_groups(self):
    ''' Calculate average state/action value for each group. Equation (3). '''

    groups = []
    for rewards, group in self.embedded_data.groupby(['reward']):
      size = group.shape[0]
      states = np.array(list(group['state'].values))
      actions = np.array(list(group['action'].values))
      groups.append({
        'size': size,
        'rewards': rewards,
        'average state': (np.sum(states / np.linalg.norm(states, 2, axis=1)[:, np.newaxis], axis=0) / size).reshape((1, -1)), 
        'average action': (np.sum(actions / np.linalg.norm(actions, 2, axis=1)[:, np.newaxis], axis=0) / size).reshape((1, -1))
      })
    return groups

  def simulate_rewards(self, current_state, chosen_actions):
    '''
    Calculate simulated rewards.
    '''
    def cosine_state_action(s_t, a_t, s_i, a_i):
      cosine_state = np.dot(s_t, s_i.T) / (np.linalg.norm(s_t, 2) * np.linalg.norm(s_i, 2))
      cosine_action = np.dot(a_t, a_i.T) / (np.linalg.norm(a_t, 2) * np.linalg.norm(a_i, 2))
      return (self.alpha * cosine_state + (1 - self.alpha) * cosine_action).reshape((1,))

    probabilities = [cosine_state_action(current_state, chosen_actions, g['average state'], g['average action']) for g in self.groups]
    probabilities = np.array(probabilities) / sum(probabilities)
    returned_rewards = self.groups[np.argmax(probabilities)]['rewards']

    def overall_reward(rewards, gamma):
      return np.sum([gamma**k * reward for k, reward in enumerate(rewards)])

    cumulated_reward = np.sum([p * overall_reward(g['rewards'], self.gamma) for p, g in zip(probabilities, self.groups)])
    return returned_rewards, cumulated_reward

class Actor():
  ''' Policy function approximator. '''
  
  def __init__(self, sess, state_space_size, action_space_size, batch_size, ra_length, history_length, embedding_size, tau, learning_rate, scope='actor'):
    self.sess = sess
    self.state_space_size = state_space_size
    self.action_space_size = action_space_size
    self.batch_size = batch_size
    self.ra_length = ra_length
    self.history_length = history_length
    self.embedding_size = embedding_size
    self.tau = tau
    self.learning_rate = learning_rate
    self.scope = scope

    with tf.compat.v1.variable_scope(self.scope):
      # Build Actor network
      self.action_weights, self.state, self.sequence_length = self._build_net('estimator_actor')
      self.network_params = tf.compat.v1.trainable_variables()

      # Build target Actor network
      self.target_action_weights, self.target_state, self.target_sequence_length = self._build_net('target_actor')
      self.target_network_params = tf.compat.v1.trainable_variables()[len(self.network_params):] 

      # Initialize target network weights with network weights (θ^π′ ← θ^π)
      self.init_target_network_params = [self.target_network_params[i].assign(self.network_params[i]) for i in range(len(self.target_network_params))]
        
      # Update target network weights (θ^π′ ← τθ^π + (1 − τ)θ^π′)
      self.update_target_network_params = [self.target_network_params[i].assign(tf.multiply(self.tau, self.network_params[i]) + tf.multiply(1 - self.tau, self.target_network_params[i])) for i in range(len(self.target_network_params))]

      # Gradient computation from Critic's action_gradients
      self.action_gradients = tf.compat.v1.placeholder(tf.float32, [None, self.action_space_size])
      gradients = tf.gradients(ys=tf.reshape(self.action_weights, [self.batch_size, self.action_space_size], name='42222222222'), xs=self.network_params, grad_ys=self.action_gradients)
      params_gradients = list(map(lambda x: tf.compat.v1.div(x, self.batch_size * self.action_space_size), gradients))
      
      # Compute ∇_a.Q(s, a|θ^µ).∇_θ^π.f_θ^π(s)
      self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(params_gradients, self.network_params))

  def _build_net(self, scope):
    ''' Build the (target) Actor network. '''

    def gather_last_output(data, seq_lens):
      def cli_value(x, v):
        y = tf.constant(v, shape=x.get_shape(), dtype=tf.int64)
        x = tf.cast(x, tf.int64)
        return tf.compat.v1.where(tf.greater(x, y), x, y)

      batch_range = tf.range(tf.cast(tf.shape(input=data)[0], dtype=tf.int64), dtype=tf.int64)
      tmp_end = tf.map_fn(lambda x: cli_value(x, 0), seq_lens - 1, dtype=tf.int64)
      indices = tf.stack([batch_range, tmp_end], axis=1)
      return tf.gather_nd(data, indices)

    with tf.compat.v1.variable_scope(scope):
      # Inputs: current state, sequence_length
      # Outputs: action weights to compute the score Equation (6)
      state = tf.compat.v1.placeholder(tf.float32, [None, self.state_space_size], 'state')
      state_ = tf.reshape(state, [-1, self.history_length, self.embedding_size])
      sequence_length = tf.compat.v1.placeholder(tf.int32, [None], 'sequence_length')
      cell = tf.compat.v1.nn.rnn_cell.GRUCell(self.embedding_size, activation = tf.nn.relu, kernel_initializer = tf.compat.v1.initializers.random_normal(), bias_initializer = tf.compat.v1.zeros_initializer())
      outputs, _ = tf.compat.v1.nn.dynamic_rnn(cell, state_, dtype=tf.float32, sequence_length=sequence_length)
      last_output = gather_last_output(outputs, sequence_length) # TODO: replace by h
      x = tf.keras.layers.Dense(self.ra_length * self.embedding_size)(last_output)
      action_weights = tf.reshape(x, [-1, self.ra_length, self.embedding_size])

    return action_weights, state, sequence_length

  def train(self, state, sequence_length, action_gradients):
    '''  Compute ∇_a.Q(s, a|θ^µ).∇_θ^π.f_θ^π(s). '''
    self.sess.run(self.optimizer, feed_dict = {self.state: state, self.sequence_length: sequence_length, self.action_gradients: action_gradients})

  def predict(self, state, sequence_length):
    return self.sess.run(self.action_weights, feed_dict = {self.state: state, self.sequence_length: sequence_length})

  def predict_target(self, state, sequence_length):
    return self.sess.run(self.target_action_weights, feed_dict = {self.target_state: state, self.target_sequence_length: sequence_length})

  def init_target_network(self):
    self.sess.run(self.init_target_network_params)

  def update_target_network(self):
    self.sess.run(self.update_target_network_params)
    
  def get_recommendation_list(self, ra_length, noisy_state, embeddings, target=False):
    '''
    returns list of embedded items as future actions.
    '''

    def get_score(weights, embedding, batch_size):
      ret = np.dot(weights, embedding.T)
      return ret

    batch_size = noisy_state.shape[0]
    method = self.predict_target if target else self.predict
    weights = method(noisy_state, [ra_length] * batch_size)
    scores = np.array([[[get_score(weights[i][k], embedding, batch_size) for embedding in embeddings.get_embedding_vector()] for k in range(ra_length)] for i in range(batch_size)])
    return np.array([[embeddings.get_embedding(np.argmax(scores[i][k])) for k in range(ra_length)] for i in range(batch_size)])

class Critic():
  ''' Value function approximator. '''
  
  def __init__(self, sess, state_space_size, action_space_size, history_length, embedding_size, tau, learning_rate, scope='critic'):
    self.sess = sess
    self.state_space_size = state_space_size
    self.action_space_size = action_space_size
    self.history_length = history_length
    self.embedding_size = embedding_size
    self.tau = tau
    self.learning_rate = learning_rate
    self.scope = scope

    with tf.compat.v1.variable_scope(self.scope):
      # Build Critic network
      self.critic_Q_value, self.state, self.action, self.sequence_length = self._build_net('estimator_critic')
      self.network_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='estimator_critic')

      # Build target Critic network
      self.target_Q_value, self.target_state, self.target_action, self.target_sequence_length = self._build_net('target_critic')
      self.target_network_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='target_critic')

      # Initialize target network weights with network weights (θ^µ′ ← θ^µ)
      self.init_target_network_params = [self.target_network_params[i].assign(self.network_params[i]) for i in range(len(self.target_network_params))]

      # Update target network weights (θ^µ′ ← τθ^µ + (1 − τ)θ^µ′)
      self.update_target_network_params = [self.target_network_params[i].assign(tf.multiply(self.tau, self.network_params[i]) + tf.multiply(1 - self.tau, self.target_network_params[i])) for i in range(len(self.target_network_params))]

      # Minimize MSE between Critic's and target Critic's outputed Q-values
      self.expected_reward = tf.compat.v1.placeholder(tf.float32, [None, 1])
      self.loss = tf.reduce_mean(input_tensor=tf.math.squared_difference(self.expected_reward, self.critic_Q_value))
      self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

      # Compute ∇_a.Q(s, a|θ^µ)
      self.action_gradients = tf.gradients(ys=self.critic_Q_value, xs=self.action)

  def _build_net(self, scope):
    ''' Build the (target) Critic network. '''

    def gather_last_output(data, seq_lens):
      def cli_value(x, v):
        y = tf.constant(v, shape=x.get_shape(), dtype=tf.int64)
        return tf.compat.v1.where(tf.greater(x, y), x, y)

      this_range = tf.range(tf.cast(tf.shape(input=seq_lens)[0], dtype = tf.int64), dtype = tf.int64)
      tmp_end = tf.map_fn(lambda x: cli_value(x, 0), seq_lens - 1, dtype = tf.int64)
      indices = tf.stack([this_range, tmp_end], axis = 1)
      return tf.gather_nd(data, indices)

    with tf.compat.v1.variable_scope(scope):
      # Inputs: current state, current action
      # Outputs: predicted Q-value
      state = tf.compat.v1.placeholder(tf.float32, [None, self.state_space_size], 'state')
      state_ = tf.reshape(state, [-1, self.history_length, self.embedding_size])
      action = tf.compat.v1.placeholder(tf.float32, [None, self.action_space_size], 'action')
      sequence_length = tf.compat.v1.placeholder(tf.int64, [None], name = 'critic_sequence_length')
      cell = tf.compat.v1.nn.rnn_cell.GRUCell(self.history_length, activation = tf.nn.relu, kernel_initializer = tf.compat.v1.initializers.random_normal(), bias_initializer = tf.compat.v1.zeros_initializer())
      predicted_state, _ = tf.compat.v1.nn.dynamic_rnn(cell, state_, dtype = tf.float32, sequence_length = sequence_length)
      predicted_state = gather_last_output(predicted_state, sequence_length)

      inputs = tf.concat([predicted_state, action], axis=-1)
      layer1 = tf.compat.v1.layers.Dense(32, activation=tf.nn.relu)(inputs)
      layer2 = tf.compat.v1.layers.Dense(16, activation=tf.nn.relu)(layer1)
      critic_Q_value = tf.compat.v1.layers.Dense(1)(layer2)
      return critic_Q_value, state, action, sequence_length

  def train(self, state, action, sequence_length, expected_reward):
    ''' Minimize MSE between expected reward and target Critic's Q-value. '''
    return self.sess.run([self.critic_Q_value, self.loss, self.optimizer], feed_dict = {self.state: state, self.action: action, self.sequence_length: sequence_length, self.expected_reward: expected_reward})

  def predict(self, state, action, sequence_length):
    ''' Returns Critic's predicted Q-value. '''
    return self.sess.run(self.critic_Q_value, feed_dict={self.state: state, self.action: action, self.sequence_length: sequence_length})

  def predict_target(self, state, action, sequence_length):
    ''' Returns target Critic's predicted Q-value. '''
    return self.sess.run(self.target_Q_value, feed_dict={self.target_state: state, self.target_action: action, self.target_sequence_length: sequence_length})

  def get_action_gradients(self, state, action, sequence_length):
    ''' Returns ∇_a.Q(s, a|θ^µ). '''
    return np.array(self.sess.run(self.action_gradients, feed_dict={self.state: state, self.action: action, self.sequence_length: sequence_length})[0])

  def init_target_network(self):
    self.sess.run(self.init_target_network_params)

  def update_target_network(self):
    self.sess.run(self.update_target_network_params)

class ReplayMemory():
  ''' Replay memory D in article. '''
  
  def __init__(self, buffer_size):
    self.buffer_size = buffer_size
    self.buffer = []

  def add(self, state, action, reward, n_state):
    self.buffer.append([state, action, reward, n_state])
    if len(self.buffer) > self.buffer_size:
      self.buffer.pop(0)

  def size(self):
    return len(self.buffer)

  def sample_batch(self, batch_size):
    return random.sample(self.buffer, batch_size)

def experience_replay(replay_memory, batch_size, actor, critic, embeddings, ra_length, state_space_size, action_space_size, discount_factor):
  '''
  Experience replay.
  '''
  samples = replay_memory.sample_batch(batch_size)
  states = np.array([s[0] for s in samples])
  actions = np.array([s[1] for s in samples])
  rewards = np.array([s[2] for s in samples])
  n_states = np.array([s[3] for s in samples]).reshape(-1, state_space_size)
  n_actions = actor.get_recommendation_list(ra_length, states, embeddings, target = True).reshape(-1, action_space_size)
  target_Q_value = critic.predict_target(n_states, n_actions, [ra_length] * batch_size)
  expected_rewards = rewards + discount_factor * target_Q_value
  critic_Q_value, critic_loss, _ = critic.train(states, actions, [ra_length] * batch_size, expected_rewards)
  action_gradients = critic.get_action_gradients(states, n_actions, [ra_length] * batch_size)
  actor.train(states, [ra_length] * batch_size, action_gradients)
  critic.update_target_network()
  actor.update_target_network()
  return np.amax(critic_Q_value), critic_loss

def train(sess, environment, actor, critic, embeddings, history_length, ra_length, buffer_size, batch_size, discount_factor, nb_episodes, filename_summary):
  def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.compat.v1.summary.scalar('reward', episode_reward)
    episode_max_Q = tf.Variable(0.)
    tf.compat.v1.summary.scalar('max_Q_value', episode_max_Q)
    critic_loss = tf.Variable(0.)
    tf.compat.v1.summary.scalar('critic_loss', critic_loss)

    summary_vars = [episode_reward, episode_max_Q, critic_loss]
    summary_ops = tf.compat.v1.summary.merge_all()
    return summary_ops, summary_vars

  summary_ops, summary_vars = build_summaries()
  sess.run(tf.compat.v1.global_variables_initializer())
  writer = tf.compat.v1.summary.FileWriter(filename_summary, sess.graph)

  actor.init_target_network()
  critic.init_target_network()

  replay_memory = ReplayMemory(buffer_size)
  replay = False


  start_time = time.time()
  for i_session in range(nb_episodes):
    session_reward = 0
    session_Q_value = 0
    session_critic_loss = 0

    states = environment.reset()
    
    if (i_session + 1) % 10 == 0:
      environment.groups = environment.get_groups()

    for t in range(nb_rounds):
      actions = actor.get_recommendation_list(ra_length, states.reshape(1, -1), embeddings).reshape(ra_length, embeddings.size()) 
      rewards, next_states = environment.step(actions)
      replay_memory.add(states.reshape(history_length * embeddings.size()), actions.reshape(ra_length * embeddings.size()), [rewards], next_states.reshape(history_length * embeddings.size()))
      states = next_states
      session_reward += rewards

      if replay_memory.size() >= batch_size:
        replay = True
        replay_Q_value, critic_loss = experience_replay(replay_memory, batch_size, actor, critic, embeddings, ra_length, history_length * embeddings.size(), ra_length * embeddings.size(), discount_factor)
        session_Q_value += replay_Q_value
        session_critic_loss += critic_loss

      summary_str = sess.run(summary_ops, feed_dict={summary_vars[0]: session_reward, summary_vars[1]: session_Q_value, summary_vars[2]: session_critic_loss})
      
      writer.add_summary(summary_str, i_session)

    str_loss = str('Loss=%0.4f' % session_critic_loss)
    print(('Episode %d/%d Reward=%d Time=%ds ' + (str_loss if replay else 'No replay')) % (i_session + 1, nb_episodes, session_reward, time.time() - start_time))
    start_time = time.time()

  writer.close()
  tf.compat.v1.train.Saver().save(sess, 'models.h5', write_meta_graph = False)

# Hyperparameters
history_length = 12
ra_length = 4
discount_factor = 0.99
actor_lr = 0.0001
critic_lr = 0.001
tau = 0.001
batch_size = 64
nb_episodes = 2
nb_rounds = 100
filename_summary = 'summary.txt'
alpha = 0.5
gamma = 0.9
buffer_size = 1000000
fixed_length = True

dg = DataGenerator('ml-100k/u.data', 'ml-100k/u.item')
dg.gen_train_test(0.8, seed=42)

dg.write_csv('train.csv', dg.train, nb_states=[history_length], nb_actions=[ra_length])
dg.write_csv('test.csv', dg.test, nb_states=[history_length], nb_actions=[ra_length])

data = read_file('train.csv')

eg = EmbeddingsGenerator(dg.user_train, pd.read_csv('ml-100k/u.data', sep='\t', names=['userId', 'itemId', 'rating', 'timestamp']))
eg.train(nb_epochs = 3000)
train_loss, train_accuracy = eg.test(dg.user_train)
print('Train set: Loss = %.4f ; Accuracy = %.1f%%' % (train_loss, train_accuracy * 100))
test_loss, test_accuracy = eg.test(dg.user_test)
print('Test set: Loss = %.4f ; Accuracy = %.1f%%' % (test_loss, test_accuracy * 100))
eg.save_embeddings('embeddings.csv')

embeddings = Embeddings(read_embeddings('embeddings.csv'))

state_space_size = embeddings.size() * history_length
action_space_size = embeddings.size() * ra_length

environment = Environment(data, embeddings, alpha, gamma, fixed_length)

tf.compat.v1.reset_default_graph() # For multiple consecutive executions

sess = tf.compat.v1.Session()
actor = Actor(sess, state_space_size, action_space_size, batch_size, ra_length, history_length, embeddings.size(), tau, actor_lr)
critic = Critic(sess, state_space_size, action_space_size, history_length, embeddings.size(), tau, critic_lr)

train(sess, environment, actor, critic, embeddings, history_length, ra_length, buffer_size, batch_size, discount_factor, nb_episodes, filename_summary)

dict_embeddings = {}
for i, item in enumerate(embeddings.get_embedding_vector()):
  str_item = str(item)
  assert(str_item not in dict_embeddings)
  dict_embeddings[str_item] = i

def state_to_items(state, actor, ra_length, embeddings, dict_embeddings, target=False):
  return [dict_embeddings[str(action)] for action in actor.get_recommendation_list(ra_length, np.array(state).reshape(1, -1), embeddings, target).reshape(ra_length, embeddings.size())]

def test_actor(actor, test_df, embeddings, dict_embeddings, ra_length, history_length, target=False, nb_rounds=1):
  movie_titles = pd.read_csv('ml-100k/u.item', sep = '|', names = ['itemId', 'itemName'], usecols = range(2), encoding = 'latin-1')
  ratings = []
  unknown = 0
  random_seen = []
  recommendations = {}
  for _ in range(nb_rounds):
    for i in range(len(test_df)):
      recommendations[i] = []
      history_sample = list(test_df[i].sample(history_length)['itemId'])
      recommendation = state_to_items(embeddings.embed(history_sample), actor, ra_length, embeddings, dict_embeddings, target)
      for item in recommendation:
        recommendation_names = list(movie_titles.loc[movie_titles['itemId'] == item]['itemName'].values)
        if len(recommendation_names) > 0:
          recommendations[i].append(recommendation_names[0])
        l = list(test_df[i].loc[test_df[i]['itemId'] == item]['rating'])
        assert(len(l) < 2)
        if len(l) == 0:
          unknown += 1
        else:
          ratings.append(l[0])
      for item in history_sample:
        random_seen.append(list(test_df[i].loc[test_df[i]['itemId'] == item]['rating'])[0])

  return ratings, unknown, random_seen, recommendations

ratings, unknown, random_seen, recommendations = test_actor(actor, dg.test, embeddings, dict_embeddings, ra_length, history_length, target = True, nb_rounds = 100)
print('%0.1f%% unknown' % (100 * unknown / (len(ratings) + unknown)))

plt.figure(4)
plt.subplot(1, 2, 1)
plt.hist(ratings)
plt.title('Predictions ; Mean = %.4f' % (np.mean(ratings)))
plt.subplot(1, 2, 2)
plt.hist(random_seen)
plt.title('Random ; Mean = %.4f' % (np.mean(random_seen)))
plt.savefig("ratings - temp.png")

for i in random.sample(recommendations.keys(), 10):
  print ("Recommendations for user " + str(i) + ": " + str(recommendations[i]))