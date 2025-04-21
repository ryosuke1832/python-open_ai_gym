import gymnasium as gym
from gymnasium import envs
import sys

def list_available_envs():
    """利用可能な環境の一覧を表示"""
    all_envs = envs.registry.keys()
    print(f"利用可能な環境数: {len(all_envs)}")
    
    # カテゴリー別に整理
    categories = {}
    for env_id in all_envs:
        category = env_id.split('/')[0] if '/' in env_id else env_id.split('-')[0]
        if category not in categories:
            categories[category] = []
        categories[category].append(env_id)
    
    # カテゴリー別に表示
    for category, envs_list in sorted(categories.items()):
        print(f"\n== {category} ({len(envs_list)}件) ==")
        for env_id in sorted(envs_list)[:10]:  # カテゴリごとに最大10件表示
            print(f"  - {env_id}")
        if len(envs_list) > 10:
            print(f"  ... 他 {len(envs_list) - 10} 件")

def check_environment():
    """環境のチェック"""
    try:
        print("=== Gymnaisum環境のテスト ===")
        env = gym.make("CartPole-v1")
        print(f"行動空間: {env.action_space}")
        print(f"観測空間: {env.observation_space}")
        env.close()
        print("✅ Gymnasium環境は正常です")
    except Exception as e:
        print(f"❌ Gymnasium環境に問題があります: {e}")
        
    try:
        print("\n=== マリオ環境のテスト ===")
        import ppaquette_gym_super_mario
        env = gym.make('ppaquette/SuperMarioBros-1-1-v0')
        env.reset()
        env.close()
        print("✅ マリオ環境は正常です")
    except Exception as e:
        print(f"❌ マリオ環境に問題があります: {e}")

def main():
    """メイン関数"""
    print("=== OpenAI Gym/Gymnasium 環境テスト ===")
    print(f"Python バージョン: {sys.version}")
    
    # 利用可能な環境一覧を表示
    list_available_envs()
    
    # 環境チェック
    print("\n")
    check_environment()

if __name__ == "__main__":
    main()