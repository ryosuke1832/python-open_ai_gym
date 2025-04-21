import gym
import ppaquette_gym_super_mario
import numpy as np
import time

def run_mario():
    """スーパーマリオブラザーズのサンプル実行"""
    # 環境の作成
    env = gym.make('ppaquette/SuperMarioBros-1-1-v0')
    
    for episode in range(3):
        observation = env.reset()  # 環境の初期化
        total_reward = 0
        
        for step in range(200):  # 最大200ステップまで実行
            # ランダムな行動の決定 (up, left, down, right, A, B)
            action = np.random.randint(0, 2, 6)
            
            # 行動による次の状態の決定
            observation, reward, done, info = env.step(action)
            total_reward += reward
            
            print(f"===== エピソード {episode+1} ステップ {step+1} =====")
            print(f"行動: {action} [上, 左, 下, 右, A, B]")
            print(f"報酬: {reward}")
            print(f"終了: {done}")
            print(f"情報: {info}")
            
            # 0.05秒待機（動作を見やすくするため）
            time.sleep(0.05)
            
            # エピソード終了条件
            if done:
                print(f"エピソード {episode+1} 終了: 合計報酬 {total_reward}")
                print(f"進んだ距離: {info.get('distance', 0)}")
                print(f"残りライフ: {info.get('life', 0)}")
                print(f"スコア: {info.get('score', 0)}")
                break
    
    env.close()  # GUI環境の終了

if __name__ == "__main__":
    run_mario()