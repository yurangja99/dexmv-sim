import click
import pickle

DESC = '''
Helper script to visualize learning.\n
USAGE:\n
    Visualizes learning on the env\n
    $ python visualize_policy --env_name relocate --object_name potted_meat_can \n
'''
from hand_imitation.env.environments.ycb_relocate_env import YCBRelocate
from hand_imitation.env.environments.mug_pour_water_env import WaterPouringEnv
from hand_imitation.env.environments.mug_place_object_env import MugPlaceObjectEnv


# MAIN =========================================================
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', required=True)
@click.option('--object_name', type=str, help='environment to load', required=False, default=None)
@click.option('--policy_path', type=str, help='path to pretrained policy', required=False, default=None)
@click.option('--render', is_flag=True, help='render if flagged', required=False, default=False)
@click.option('--num_eval_episodes', type=int, help='num of eval episodes', required=False, default=1000)
def main(env_name, object_name, policy_path, render, num_eval_episodes):
    print(env_name, object_name, policy_path, render, num_eval_episodes)
    friction = (1, 0.5, 0.01)
    if env_name == "relocate":
        if object_name is None:
            raise ValueError("For relocate task, object name is needed.")
        e = YCBRelocate(has_renderer=render, object_name=object_name, friction=friction, object_scale=0.8,
                        solref="-6000 -300", randomness_scale=0.25)
        T = 100
    elif env_name == "pour":
        e = WaterPouringEnv(has_renderer=render, scale=1.0, tank_size=(0.15, 0.15, 0.08))
        T = 200
    elif env_name == "place_inside":
        e = MugPlaceObjectEnv(has_renderer=render, object_scale=0.8, mug_scale=1.5)
        T = 200
    else:
        raise NotImplementedError

    if policy_path is None:
        if env_name == "relocate":
            policy = f"../pretrained_model/{env_name}-{object_name}.pickle"
        else:
            policy = f"../pretrained_model/{env_name}.pickle"
    else:
        policy = policy_path
    
    pi = pickle.load(open(policy, 'rb'))
    total_reward = 0
    total_success_rate = 0
    for _ in range(num_eval_episodes):
        state, done = e.reset(), False
        step = 0
        reward_sum = 0
        success_rate = 0
        while True:
            step += 1
            if step >= T:
                success_rate = e.success()
                break
            action = pi.get_action(state)[1]['evaluation']
            state, reward, done, _ = e.step(action)
            if render:
                for _ in range(2):
                    e.render()
            reward_sum += reward
        total_reward += reward_sum
        total_success_rate += success_rate
        print(f'total reward {reward_sum}, success rate {success_rate}')
    print(f"Eval {num_eval_episodes} episodes.")
    print(f"Average reward: {total_reward / num_eval_episodes}")
    print(f"Average success rate: {total_success_rate / num_eval_episodes}")


if __name__ == '__main__':
    main()
