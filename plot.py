import time
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 📂 Kayıt klasörünü oluştur (eğer yoksa)
save_dir = "plots"
os.makedirs(save_dir, exist_ok=True)

def read_log_file(filename):
    try:
        data = np.loadtxt(filename, delimiter=',')
        return data if data.size > 0 else np.array([])
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return np.array([])

def moving_average(data, window_size=1000):
    if len(data) < window_size:
        return np.zeros(len(data))
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_and_save():
    while True:
        # 📊 Yeni figür ve eksenleri oluştur
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Log dosyalarını oku
        reward_per_step = read_log_file("reward_per_step.txt")
        total_reward_per_episode = read_log_file("total_reward_per_episode.txt")

        if reward_per_step.size > 0:
            rewards = reward_per_step[:, 1]
            global_steps = np.arange(len(rewards))  # 🔹 Satır numarası = Step numarası

            #axs[0].plot(global_steps, rewards, label="Reward per Step", alpha=0.5)
            axs[0].plot(global_steps[:len(moving_average(rewards))], moving_average(rewards), label="Moving Avg (100)", color='red')
            axs[0].set_title("Reward per Step")
            axs[0].set_xlabel("Cumulative Step")
            axs[0].set_ylabel("Reward")
            axs[0].legend()

        if total_reward_per_episode.size > 0 and total_reward_per_episode.ndim == 2:
            episodes, total_rewards = total_reward_per_episode[:, 0], total_reward_per_episode[:, 1]

            #axs[1].plot(episodes, total_rewards, label="Total Reward per Episode", alpha=0.5)
            axs[1].plot(episodes[:len(moving_average(total_rewards))], moving_average(total_rewards), label="Moving Avg (100)", color='red')
            axs[1].set_title("Total Reward per Episode")
            axs[1].set_xlabel("Episode")
            axs[1].set_ylabel("Total Reward")
            axs[1].legend()

        # 📂 PNG olarak kaydet (tarih + saat ile)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = os.path.join(save_dir, f"plot_{timestamp}.png")
        plt.savefig(save_path)
        print(f"Plot saved: {save_path}")

        plt.close(fig)  # 🔹 Belleği temizle
        time.sleep(600)  # ⏳ 10 dakika bekle (600 saniye)

if __name__ == "__main__":
    plot_and_save()
