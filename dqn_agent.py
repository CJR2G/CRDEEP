
import random, numpy as np, torch, torch.nn as nn, torch.optim as optim, os, json

class QNet(nn.Module):
    def __init__(self, state_size, action_size, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_size)
        )
    def forward(self, x): return self.net(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma  = float(os.getenv("GAMMA", 0.95))
        self.epsilon= float(os.getenv("EPSILON_START", 1.0))
        self.eps_min= float(os.getenv("EPSILON_MIN", 0.05))
        self.eps_decay=float(os.getenv("EPSILON_DECAY", 0.995))
        self.lr     = float(os.getenv("LR", 5e-4))
        self.batch_size = int(os.getenv("REPLAY_BATCH_SIZE", 64))
        self.tgt_update_steps = int(os.getenv("TARGET_UPDATE_STEPS", 1000))

        self.model = QNet(state_size, action_size)
        self.target_model = QNet(state_size, action_size)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.memory = []
        self.max_mem = 20000
        self.steps = 0

    def remember(self, s, a, r, s2, done):
        if len(self.memory) >= self.max_mem: self.memory.pop(0)
        self.memory.append((s, a, r, s2, done))

    def act(self, state):
        self.steps += 1
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q = self.model(s)[0]
            return int(torch.argmax(q).item())

    def replay(self):
        if len(self.memory) < self.batch_size: return 0.0
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s2, d = zip(*batch)
        s  = torch.tensor(np.array(s),  dtype=torch.float32)
        a  = torch.tensor(a, dtype=torch.long).unsqueeze(1)
        r  = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
        s2 = torch.tensor(np.array(s2), dtype=torch.float32)
        d  = torch.tensor(d, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            q_next = self.target_model(s2).max(1, keepdim=True)[0]
            y = r + self.gamma * q_next * (1 - d)

        q = self.model(s).gather(1, a)
        loss = self.criterion(q, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.eps_min:
            self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

        if self.steps % self.tgt_update_steps == 0:
            self.update_target_model()

        return float(loss.item())

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, path):
        meta = {"epsilon": self.epsilon, "state_size": self.state_size, "action_size": self.action_size}
        torch.save(self.model.state_dict(), path)
        with open(path.replace(".pth", ".meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location="cpu"))
        meta_path = path.replace(".pth", ".meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.epsilon = meta.get("epsilon", self.epsilon)
        self.update_target_model()
