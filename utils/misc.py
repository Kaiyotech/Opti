from prettytable import PrettyTable
import numpy as np

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    critic_params = 0
    actor_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        if "critic" in name:
            critic_params += params
        if "actor" in name:
            actor_params += params
        total_params += params
    print(table)
    print(f"Actor Params: {actor_params}")
    print(f"Critic Params: {critic_params}")
    print(f"Total Trainable Params: {total_params}")
    return total_params


def remove_bad_states():
    files = ["replays/easy_double_tap_1v0.npy", "replays/easy_double_tap_1v1.npy"]
    datas = []
    for file in files:
        datas.append(np.load(file))
    bad_states = [1154, 966]
    bad_states.sort(reverse=True)
    for i, data in enumerate(datas):
        data = np.delete(data, bad_states, 0)
        np.save(files[i], data)

