import gym



env = gym.make("CartPole-v1")
state = env.reset()
while True:
    env.render()
    action = model.predict(state)
    next_state, reward, done, info = env.step(action)
    if done:
        break
