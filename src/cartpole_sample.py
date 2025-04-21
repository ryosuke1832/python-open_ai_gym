import gymnasium as gym
import time

def run_cartpole():
    """CartPole-v1環境でのサンプル実行"""
    env = gym.make("CartPole-v1", render_mode="human")  # GUI環境の開始
    
    for episode in range(5):
        observation, info = env.reset()  # 環境の初期化
        total_reward = 0
        
        for step in range(500):  # 最大500ステップまで実行
            env.render()  # レンダリング(画面の描画)
            action = env.action_space.sample()  # ランダムな行動の決定
            
            # 行動による次の状態の決定
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"===== エピソード {episode+1} ステップ {step+1} =====")
            print(f"行動: {action} (0:左, 1:右)")
            print(f"状態: {observation}")
            print(f"報酬: {reward}")
            print(f"終了: {terminated}")
            print(f"情報: {info}")
            
            # 0.01秒待機（動作を見やすくするため）
            time.sleep(0.01)
            
            # エピソード終了条件
            if terminated or truncated:
                print(f"エピソード {episode+1} 終了: 合計報酬 {total_reward}")
                break
    
    env.close()  # GUI環境の終了

if __name__ == "__main__":
    run_cartpole()