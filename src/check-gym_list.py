from gym import envs

envids = [print(spec.id) for spec in envs.registry.all()]
