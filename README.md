# Smart Elevator System

A simulation of an elevator system powered by **Q-Learning**, capable of optimizing floor visitation order based on requests. Includes an **animated visualization** of the elevator journey and **text-to-speech (TTS)** floor announcements.

---

## Features

- Reinforcement Learning using Q-Learning
- Intelligent floor selection based on learned experience
- Reward and epsilon decay visualization
- Animated simulation of elevator movement
- Voice floor announcements (TTS)
- Modular structure: Environment, Agent, Training, Animation

---

## How It Works

The elevator is treated as an agent operating in an environment of multiple floors. It receives requests to visit specific floors and learns the most efficient order to serve them using Q-learning. Over time, it learns to minimize travel distance and improve response efficiency.

---

## Tech Stack

- `Python 3`
- `NumPy` for numerical operations
- `Matplotlib` for animation
- `pyttsx3` for offline text-to-speech
- `Threading` for non-blocking audio
- `Q-Learning` for reinforcement learning logic

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/elevator-ai-optimizer.git
   cd elevator-ai-optimizer
   ```

2. **Install dependencies**
   ```bash
   pip install numpy matplotlib pyttsx3
   ```

---

## Running the Simulation

Run the main script:

```bash
python elevator_simulation.py
```

You will be prompted to:
- Enter the starting floor (0–9)
- Enter the requested floors (space-separated)

The system will train a Q-learning agent, display the optimized path, and animate the elevator's movement. Voice announcements will indicate when each floor is reached.

---

## Output Example

- Optimized order of floors: [2, 5, 8]
- Training complete in 5000 episodes
- Reward and epsilon decay plotted in real-time
- Animated elevator movement from start to target floors

---

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute it.

---

## Contributing

Pull requests are welcome. If you have suggestions for improvements or bug fixes, feel free to open an issue or fork the repository and submit a PR.

---

## Acknowledgments

Built as a demonstration of applying Reinforcement Learning concepts to real-world problems using Python.