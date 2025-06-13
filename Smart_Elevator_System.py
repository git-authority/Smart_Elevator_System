import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import math
import threading
import pyttsx3


# ---------------------
# Elevator Environment
# ---------------------

class ElevatorEnv:
    def __init__(self, num_floors=10):
        self.num_floors = num_floors
        self.reset()

    def reset(self, start_floor=None, requests=None):
        self.current_floor = random.randint(0, self.num_floors - 1) if start_floor is None else start_floor
        self.requests = random.sample(range(self.num_floors), 3) if requests is None else requests.copy()
        self.visited = []
        return self._get_state()

    def _get_state(self):
        state = np.zeros(self.num_floors)
        state[self.current_floor] = 1
        return state

    def step(self, action):
        reward = -abs(self.current_floor - action)
        self.current_floor = action
        if action in self.requests:
            self.requests.remove(action)
            self.visited.append(action)
        done = len(self.requests) == 0
        return self._get_state(), reward, done




# ----------------
# Q-Learning Agent
# ----------------

class QLearningAgent:
    def __init__(self, num_floors, lr=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.q_table = np.zeros((num_floors, num_floors))
        self.num_floors = num_floors
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

# Decide which floor to go to next.

    def select_action(self, state, requests):
        current_floor = np.argmax(state)
        if np.random.rand() < self.epsilon:
            return random.choice(requests)
        q_values = self.q_table[current_floor]
        masked_q_values = [q_values[r] for r in requests]
        return requests[np.argmax(masked_q_values)]


# Update the Q-table based on the reward received from taking an action.

    def update(self, state, action, reward, next_state):
        current_floor = np.argmax(state)
        next_floor = np.argmax(next_state)
        best_next_action = np.max(self.q_table[next_floor])           # Value of q in q-table for the next target floor
        td_target = reward + self.gamma * best_next_action            # Observed Value
        td_error = td_target - self.q_table[current_floor][action]    # TD Error
        self.q_table[current_floor][action] += self.lr * td_error     # Update Rule using alpha



# --------------
# Epsilon Update
# --------------

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)




# -------------
# Training Loop
# -------------

env = ElevatorEnv(num_floors=10)
agent = QLearningAgent(num_floors=10)

num_episodes = 5000
reward_list = []
epsilon_list = []

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state, env.requests)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
        total_reward += reward

    agent.decay_epsilon()
    reward_list.append(total_reward)
    epsilon_list.append(agent.epsilon)

    if episode % 500 == 0:
        print(f"Episode {episode}: Total Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.3f}")

print("✅ Training complete!")

# Plotting rewards and epsilon
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(reward_list)
plt.title("Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Total Reward")

plt.subplot(1, 2, 2)
plt.plot(epsilon_list)
plt.title("Epsilon Decay")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.tight_layout()
plt.show()




# ----------------------------------------
# Get Optimized Path Using Trained Q-Table
# ----------------------------------------

def get_optimized_path(starting_floor, requested_floors, q_table):
    current_floor = starting_floor
    path = [current_floor]
    pending_requests = requested_floors.copy()
    while pending_requests:
        q_values = q_table[current_floor]
        masked_q_values = [q_values[f] for f in pending_requests]
        next_floor = pending_requests[np.argmax(masked_q_values)]
        path.append(next_floor)
        pending_requests.remove(next_floor)
        current_floor = next_floor


    # Remove consecutive duplicates

    optimized = [path[0]]
    for floor in path[1:]:
        if floor != optimized[-1]:
            optimized.append(floor)
    return optimized

starting_floor = int(input("\nEnter starting floor (0 to 9): "))
requested_floors = list(map(int, input("Enter requested floors (space-separated): ").split()))

optimized_path = get_optimized_path(starting_floor, requested_floors, agent.q_table)
print("\n🚀 Optimized Path:", optimized_path)

# -----------------------------
# Animation Setup Using the Optimized Path
# -----------------------------
NUM_FLOORS = 10
frames_per_floor = 30
door_anim_frames = 20

# Increase floor and box height.
floor_height = 1.5  # Each floor is 1.5 units tall
box_height = 1.5    # Elevator box is now taller
ordered_route = optimized_path

print("\n🏁 Elevator Trip Summary:")
print(f"🔹 Starting Floor: {starting_floor}")
print(f"🔹 Requested Floors (input): {requested_floors}")
print(f"🔹 Optimized Route: {ordered_route}")

# -----------------------------
# Utility Functions for Animation & TTS (No Ding)
# -----------------------------
def ease_in_out(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)

def compute_movement_frames(sf, ef, nframes=30):
    sf_pos = sf * floor_height
    ef_pos = ef * floor_height
    total_distance = ef_pos - sf_pos
    if nframes < 3:
        return [sf_pos + (ef_pos - sf_pos) * (i/(nframes-1)) for i in range(nframes)]
    T = nframes
    T_a = max(1, T // 3)
    T_d = T_a
    T_c = T - T_a - T_d
    v_max = total_distance / (T_c + T_a)
    positions = []
    for t in range(T):
        if t < T_a:
            a = v_max / T_a
            pos = sf_pos + 0.5 * a * (t**2)
        elif t < T_a + T_c:
            pos = sf_pos + 0.5 * v_max * T_a + v_max * (t - T_a)
        else:
            t_dec = t - (T_a + T_c)
            a = v_max / T_d
            pos = sf_pos + 0.5 * v_max * T_a + v_max * T_c + v_max * t_dec - 0.5 * a * (t_dec**2)
        positions.append(pos)
    return positions

def compute_door_frames(door_anim_frames=20, opening=True):
    door_positions = []
    max_offset = 0.2
    for i in range(door_anim_frames):
        if opening:
            t = ease_in_out(i / (door_anim_frames - 1))
            door_positions.append(t * max_offset)
        else:
            t = ease_in_out(1 - i / (door_anim_frames - 1))
            door_positions.append(t * max_offset)
    return door_positions

def speak_floor_announcement(floor):
    try:
        engine = pyttsx3.init()
        engine.say(f"Floor {floor}")
        engine.runAndWait()
    except Exception as e:
        print(f"TTS error: {e}")

def play_floor_announcement(floor):
    # Only TTS (ding sound removed)
    threading.Thread(target=speak_floor_announcement, args=(floor,), daemon=True).start()

# -----------------------------
# Build Combined Animation Frames (Movement, Door, Pause)
# -----------------------------
all_frames = []
frame_actions = []

def add_door_frames(floor, phase):
    door_frames = compute_door_frames(door_anim_frames, opening=(phase=='door_open'))
    for idx, offset in enumerate(door_frames):
        frame = {'y': floor * floor_height, 'door_offset': offset, 'phase': phase, 'current_floor': floor}
        if phase == 'door_open' and idx == 0:
            frame['announce'] = True
        all_frames.append(frame)
        frame_actions.append(phase)

# Animate door at starting floor.
add_door_frames(starting_floor, 'door_open')
for _ in range(10):
    all_frames.append({'y': starting_floor * floor_height,
                       'door_offset': compute_door_frames(door_anim_frames, opening=True)[-1],
                       'phase': 'pause', 'current_floor': starting_floor})
    frame_actions.append('pause')
add_door_frames(starting_floor, 'door_close')

# Process each transition in the optimized route.
for i in range(len(ordered_route) - 1):
    sf = ordered_route[i]
    ef = ordered_route[i + 1]
    move_frames = compute_movement_frames(sf, ef, frames_per_floor)
    for pos in move_frames:
        all_frames.append({'y': pos, 'phase': 'move', 'current_floor': None})
        frame_actions.append('move')
    # For all stops except the final destination, add door open/close.
    if i < len(ordered_route) - 2:
        add_door_frames(ef, 'door_open')
        for _ in range(10):
            all_frames.append({'y': ef * floor_height,
                               'door_offset': compute_door_frames(door_anim_frames, opening=True)[-1],
                               'phase': 'pause', 'current_floor': ef})
            frame_actions.append('pause')
        add_door_frames(ef, 'door_close')
    else:
        # For final destination, only open doors and pause.
        add_door_frames(ef, 'door_open')
        for _ in range(10):
            all_frames.append({'y': ef * floor_height,
                               'door_offset': compute_door_frames(door_anim_frames, opening=True)[-1],
                               'phase': 'pause', 'current_floor': ef})
            frame_actions.append('pause')

# -----------------------------
# Setup the Animation Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(6, 12))  # Taller figure
ax.set_xlim(0, 5)                       # Widen x-limits
ax.set_ylim(-1, NUM_FLOORS * floor_height + 1)
# Force equal aspect so the elevator is not distorted
ax.set_aspect('equal', adjustable='box')

ax.set_xticks([])
ax.set_yticks([i * floor_height for i in range(NUM_FLOORS)])
ax.set_yticklabels([f'Floor {i}' for i in range(NUM_FLOORS)])
ax.set_facecolor("#f0f0f0")

# Draw horizontal lines for floors
for i in range(NUM_FLOORS):
    ax.axhline(i * floor_height, color='lightgray', linestyle='--', linewidth=0.6)

# Elevator shaft
shaft_x = 2
shaft_w = 1.2  # Increased shaft width for a wider outer box
shaft_rect = patches.Rectangle(
    (shaft_x, 0),
    shaft_w,
    NUM_FLOORS * floor_height,
    fill=False,
    edgecolor='black',
    linewidth=2
)
ax.add_patch(shaft_rect)

# Cable (removed "CABLE" text)
ax.plot(
    [shaft_x + shaft_w / 2, shaft_x + shaft_w / 2],
    [NUM_FLOORS * floor_height, NUM_FLOORS * floor_height + 1],
    color='gray',
    linewidth=2
)

# Elevator cabin
elevator_box = patches.Rectangle(
    (shaft_x + (shaft_w - 0.8) / 2, starting_floor * floor_height),
    0.8,
    box_height,
    facecolor='green',
    edgecolor='green',
    linewidth=2
)
ax.add_patch(elevator_box)

# Door panels
door_left = patches.Rectangle(
    (shaft_x + (shaft_w - 0.8) / 2, starting_floor * floor_height),
    0.8 / 2,
    box_height,
    facecolor='gray',
    edgecolor='black'
)
door_right = patches.Rectangle(
    (shaft_x + (shaft_w - 0.8) / 2 + 0.8 / 2, starting_floor * floor_height),
    0.8 / 2,
    box_height,
    facecolor='gray',
    edgecolor='black'
)
ax.add_patch(door_left)
ax.add_patch(door_right)

# Digital indicator panel
indicator = ax.text(
    shaft_x + shaft_w + 0.1,
    starting_floor * floor_height + box_height / 2,
    f"Floor: {starting_floor}",
    va='center',
    ha='left',
    fontsize=14,
    color='blue',
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
)

# Floor markers for the requested floors
marker_texts = {}
for floor in ordered_route[1:]:
    txt = ax.text(
        0.5,
        floor * floor_height,
        "●",
        va='center',
        ha='center',
        fontsize=16,
        color='red'
    )
    marker_texts[floor] = txt

ax.set_title(f"Elevator Trip: Route {ordered_route}", fontsize=16)

# -----------------------------
# Animation Update Function
# -----------------------------
def update(frame_index):
    frame = all_frames[frame_index]
    y = frame['y']
    phase = frame['phase']
    current_floor = frame.get('current_floor', None)

    # Move the cabin
    elevator_box.set_y(y)
    door_left.set_y(y)
    door_right.set_y(y)

    # Color changes for door phases
    if phase in ['door_open', 'pause', 'door_close']:
        elevator_box.set_facecolor("white")
        elevator_box.set_edgecolor("green")
    else:
        elevator_box.set_facecolor("green")
        elevator_box.set_edgecolor("green")

    # Update indicator text
    if current_floor is not None:
        indicator.set_text(f"Floor: {current_floor}")
        indicator.set_y(y + box_height / 2)
    else:
        # Convert y to floor by dividing by floor_height
        indicator.set_text(f"Floor: {round(y / floor_height)}")
        indicator.set_y(y + box_height / 2)

    # Adjust door panels if opening/closing
    if phase in ['door_open', 'pause', 'door_close']:
        door_offset = frame.get('door_offset', 0)
        door_left.set_x(shaft_x + (shaft_w - 0.8) / 2 - door_offset)
        door_right.set_x(shaft_x + (shaft_w - 0.8) / 2 + 0.8 / 2 + door_offset)
    else:
        door_left.set_x(shaft_x + (shaft_w - 0.8) / 2)
        door_right.set_x(shaft_x + (shaft_w - 0.8) / 2 + 0.8 / 2)

    # Color requested floors green when elevator arrives
    for floor, txt_obj in marker_texts.items():
        if current_floor is not None and abs(floor - current_floor) < 0.1:
            txt_obj.set_color('green')
        else:
            txt_obj.set_color('red')

    # Floor announcement
    if phase == 'door_open' and frame.get('announce', False):
        play_floor_announcement(current_floor)
        frame.pop('announce', None)

    return [elevator_box, door_left, door_right, indicator] + list(marker_texts.values())

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(all_frames),
    interval=50,
    blit=True,
    repeat=False
)

plt.tight_layout()
plt.show()
