import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_rewards(data_path):
    list_rewards = []
    with open(data_path + '/eval.log') as file:
        for line in file.readlines():
            rewards_ = float(line.strip().split(',')[1].split(':')[1].strip())
            rewards_ = round(rewards_, 5)
            list_rewards.append(rewards_)
            # if len(list_rewards) == 50:
            #     break
    return list_rewards

def plot_res(task_name, data1, data2):
    ax = plt.figure(figsize=(8, 6))
    sns.set(style='darkgrid')
    x1 = [i for i in range(150)]
    x2 = [i for i in range(150)]

    plt.title(task_name)
    plt.xlabel('episodes')
    plt.ylabel('rewards')
    plt.plot(x1, data1, label='CUIRL', linewidth=2)
    plt.plot(x2, data2, label='CURL', linewidth=2)
    plt.legend()
    # plt.savefig('pic/' + task_name + '.png', dpi=200)
    plt.show()

if __name__ == '__main__':
    dict_data = {'cheetah': ['results/cheetah-run-05-11-19-22-im84-b64-s1-pixel', 'results/cheetah-run-05-12-18-22-im84-b64-s1-pixel'],
                 'cartpole': ['results/cartpole-swingup-05-11-21-02-im84-b64-s1-pixel', 'results/cartpole-swingup-05-12-19-51-im84-b64-s1-pixel'],
                 'finger': ['results/finger-spin-05-12-15-41-im84-b64-s1-pixel', 'results/finger-spin-05-12-16-59-im84-b64-s1-pixel'],
                 'walker':['results/walker-walk/ctmr_sac-walker-walk-05-24-04-32-im84-b64-pixel-s1', 'results/walker-walker-curl']
                 }


    # # data_path_pixel = 'results/cartpole-swingup-05-11-21-02-im84-b64-s1-pixel'
    # # data_path_identity = 'results/cartpole-swingup-05-11-22-17-im84-b64-s1-identity'
    # data_path_pixel = 'results/cheetah-run-05-11-19-22-im84-b64-s1-pixel'
    # data_path_identity = 'results/cheetah-run-05-11-20-56-im84-b64-s1-identity'
    # task_name = 'cheetah'
    # task_name = 'cartpole'
    # task_name = 'finger'
    task_name = 'walker'

    list_rewards_1 = get_rewards(dict_data[task_name][0])
    list_rewards_1 = list_rewards_1[::2]
    list_rewards_1 = list_rewards_1[:150]
    print(len(list_rewards_1))

    list_rewards_2 = get_rewards(dict_data[task_name][1])
    list_rewards_2 = list_rewards_2[:150]


    plot_res(task_name, list_rewards_1, list_rewards_2)