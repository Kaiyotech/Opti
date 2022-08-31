from prettytable import PrettyTable


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
