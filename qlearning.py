import gym
import numpy as np

env = gym.make('Acrobot-v1')


class Ai:
    def __init__(self,env):

        self.DISCRETE_OS_SIZE = [4,4,4,4,4, 4]
        self.discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / self.DISCRETE_OS_SIZE
        self.LearningRate = 0.1
        self.Discount = 0.95
        self.Episodes = 25000
        self.epsilon = 1
        self.startepsilondecay = 1
        self.endepsilondecay = self.Episodes // 2
        self.epsilondecayvalue = self.epsilon / (self.endepsilondecay - self.startepsilondecay)
        self.q_table = np.random.uniform(low=-2, high=0, size=(self.DISCRETE_OS_SIZE + [env.action_space.n]))
        self.score = 0

    def get_discrete_state(self,state):

        discrete_state = (state - env.observation_space.low) / self.discrete_os_win_size
        discrete_state= tuple(discrete_state.astype(np.int))
        return discrete_state

    def move(self):

        if np.random.random() > self.epsilon:
            self.action = np.argmax(self.q_table[self.discrete_state])
        else:
            self.action = np.random.randint(0, env.action_space.n)
        self.new_state, self.reward, done, _ = env.step(self.action)
        self.new_discrete_state = self.get_discrete_state(self.new_state)
        return done

    def newq(self):

        self.max_future_q = np.max(self.q_table[self.new_discrete_state])
        self.current_q = self.q_table[self.discrete_state + (self.action,)]
        self.new_q = (1 - self.LearningRate) * self.current_q + self.LearningRate * (self.reward + self.Discount * self.max_future_q)
        self.q_table[self.discrete_state + (self.action,)] = self.new_q

        # if self.new_state[0] >= env.goal_position:
        #     # q_table[discrete_state + (action,)] = reward
        #     self.q_table[self.discrete_state + (self.action,)] = 0
        self.discrete_state = self.new_discrete_state

    def epsilondecay(self):
        if self.endepsilondecay >= episode >= self.startepsilondecay:
            self.epsilon -= self.epsilondecayvalue


qlearner = Ai(env)
qlearner.discrete_state = qlearner.get_discrete_state(env.reset())
done = False
showEvery = 1

for episode in range(qlearner.Episodes):
    print(episode)
    qlearner.discrete_state = qlearner.get_discrete_state(env.reset())
    done = False

    if episode % showEvery == 0:
        render = True
    else:
        render = False
    while not done:
        done = qlearner.move()
        if render == True:
            env.render()
        if not done:
            qlearner.newq()
    qlearner.epsilondecay()


env.close()
print('final:', 50000-qlearner.score)
