import time

import gym


def test_render():
    print("MountainCar環境のレンダリングテスト")

    # 環境情報の表示
    print(f"Gym version: {gym.__version__}")

    try:
        # 環境の作成（新しいバージョン向け）
        try:
            env = gym.make("MountainCar-v0", render_mode="human")
            print("新しいバージョンのGymで環境を作成しました")
        except (TypeError, ValueError):
            # 古いバージョン向け
            env = gym.make("MountainCar-v0")
            print("古いバージョンのGymで環境を作成しました")

        # 環境情報の表示
        print(f"行動空間: {env.action_space}")
        print(f"状態空間: {env.observation_space}")
        print(f"状態範囲: {env.observation_space.low} 〜 {env.observation_space.high}")

        # 初期化と描画テスト
        print("\n環境をリセットして描画テスト開始")
        observation = env.reset()
        if isinstance(observation, tuple):
            observation = observation[0]

        print(f"初期状態: {observation}")

        # レンダリングテスト
        try:
            print("レンダリング開始...")
            env.render()
            print("レンダリング成功!")
            time.sleep(2)
        except Exception as e:
            print(f"レンダリングエラー: {e}")
            print("\nデバッグ情報:")
            print(f"エラータイプ: {type(e).__name__}")
            print(f"エラー詳細: {str(e)}")

            # エラースタックトレースの表示
            import traceback

            traceback.print_exc()

            print("\n考えられる解決策:")
            print("1. GUIサポートのある環境での実行（X11転送など）")
            print("2. 'xvfb-run python script.py' を使用する")
            print("3. 'headless' レンダリング方法を使用する（新しいGymバージョンのみ）")

        # 数ステップ実行してみる
        print("\n数ステップ実行テスト:")
        for i in range(10):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if isinstance(info, tuple):
                info = info[0]

            print(f"ステップ {i + 1}:")
            print(f"  行動: {action}")
            print(f"  状態: {observation}")
            print(f"  報酬: {reward}")
            print(f"  終了: {done}")

            try:
                env.render()
                time.sleep(0.5)
            except:
                pass

            if done:
                print("  ゴール到達!")
                break

    except Exception as e:
        print(f"全体的なエラー: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # 環境を閉じる
        try:
            env.close()
            print("\n環境を正常に閉じました")
        except:
            print("\n環境を閉じる際にエラーが発生しました")


if __name__ == "__main__":
    test_render()
