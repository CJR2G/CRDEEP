
import os, time, numpy as np
from dqn_agent import DQNAgent
from env import Env

def train(episodes=1000, max_steps=200):
    env = Env()
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)
    for ep in range(1, episodes+1):
        s = env.reset()
        total_r, losses = 0.0, []
        for t in range(max_steps):
            a = agent.act(s)
            s2, r, done, info = env.step(a)
            agent.remember(s, a, r, s2, done)
            s = s2
            total_r += r
            loss = agent.replay()
            if loss: losses.append(loss)
            if done: break
        print(f"Ep {ep:04d} | R={total_r:.2f} | steps={t+1} | eps={agent.epsilon:.3f} | loss={np.mean(losses) if losses else 0:.4f}")
        if ep % 10 == 0:
            os.makedirs("models", exist_ok=True)
            agent.save(f"models/model_{ep:04d}.pth")

if __name__ == "__main__":
    train()
