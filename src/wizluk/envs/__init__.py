# -*- coding: utf-8 -*-
"""
    OpenAI Gym Environments
"""

from ._history         import History
from gym.envs.registration  import register
#from gym.scoreboard.registration import add_task, add_group
#from .package_info import USERNAME

# Environment registration

register(
    id='CliffWorld-v1',
    entry_point='wizluk.envs.cliffworld:CliffWorldEnv',
    max_episode_steps = 100,
    kwargs = {
        'width': 12,
        'height': 4,
        'sticky_prob' : 0.0
    }
)

register(
    id='StickyCliffWorld-v1',
    entry_point='wizluk.envs.cliffworld:CliffWorldEnv',
    max_episode_steps = 100,
    kwargs = {
        'width': 12,
        'height': 4,
        'sticky_prob' : 0.25
    }
)

register(
    id='WumpusWorld-v1',
    entry_point='wizluk.envs.wumpus_world:WumpusWorldEnv',
    max_episode_steps = 1000,
    kwargs = {'observable': 'perceptions'}
)

register(
    id='WumpusWorld-v2',
    entry_point='wizluk.envs.wumpus_world:WumpusWorldEnv',
    max_episode_steps = 1000,
    kwargs = {'observable': 'full'}
)

register(
    id='WumpusWorld-v3',
    entry_point='wizluk.envs.wumpus_world:WumpusWorldEnv',
    max_episode_steps = 1000,
    kwargs = {'observable': 'full',
        'setting': 'gold_on_shortest_path',
        'deviation_of_gold_from_shortest_path': 1
        }
)
register(
    id='WumpusWorld-v4',
    entry_point='wizluk.envs.wumpus_world:WumpusWorldEnv',
    max_episode_steps = 1000,
    kwargs = {'observable': 'full',
        'setting': 'gold_on_shortest_path',
        'deviation_of_gold_from_shortest_path': 5
        }
)
register(
    id='WumpusWorld-v5',
    entry_point='wizluk.envs.wumpus_world:WumpusWorldEnv',
    max_episode_steps = 1000,
    kwargs = {'observable': 'full',
        'setting': 'gold_on_shortest_path',
        'deviation_of_gold_from_shortest_path': 6
        }
)
register(
    id='WumpusWorld-v6',
    entry_point='wizluk.envs.wumpus_world:WumpusWorldEnv',
    max_episode_steps = 1000,
    kwargs = {'observable': 'full',
        'setting': 'gold_on_shortest_path',
        'deviation_of_gold_from_shortest_path': 7
        }
)
register(
    id='WumpusWorld-v7',
    entry_point='wizluk.envs.wumpus_world:WumpusWorldEnv',
    max_episode_steps = 1000,
    kwargs = {'observable': 'full',
        'setting': 'gold_on_shortest_path',
        'deviation_of_gold_from_shortest_path': 8
        }
)

register(
    id='WumpusWorld-randActions-v3',
    entry_point='wizluk.envs.wumpus_world:WumpusWorldEnv',
    max_episode_steps = 1000,
    kwargs = {'observable': 'full',
        'setting': 'gold_on_shortest_path',
        'deviation_of_gold_from_shortest_path': 1,
        'random_action_prob': 0.25
        }
)

register(
    id='WumpusWorld-randActions-v7',
    entry_point='wizluk.envs.wumpus_world:WumpusWorldEnv',
    max_episode_steps = 1000,
    kwargs = {'observable': 'full',
        'setting': 'gold_on_shortest_path',
        'deviation_of_gold_from_shortest_path': 8,
        'random_action_prob': 0.25
        }
)

for d in [ 4, 8, 10, 20, 50, 16, 32, 64, 100, 1000 ] :
    register(
        id='GridWorld-{}x{}-v1'.format(d,d),
        entry_point='wizluk.envs.gridworld:GridWorldEnv',
        max_episode_steps=1000,
        kwargs={
            'dimension': d
        }
    )

for d in [ 4, 8, 10, 20, 50, 16, 32, 64, 100, 1000 ] :
    register(
        id='GridWorld-{}x{}-v2'.format(d,d),
        entry_point='wizluk.envs.gridworld:GridWorldEnv',
        max_episode_steps=1000,
        kwargs={
            'dimension': d,
            'terminal_def': "center"
        }
    )
    for s in range(10):
        register(
            id='GridWorld-{}x{}-initS{}-v2'.format(d,d,s),
            entry_point='wizluk.envs.gridworld:GridWorldEnv',
            max_episode_steps=1000,
            kwargs={
                'dimension': d,
                'initState': s,
                'terminal_def': "center",
            }
        )

for d in [ 4, 8, 10, 20, 50, 16, 32, 64, 100, 1000 ] :
    register(
        id='GridWorld-{}x{}-v3'.format(d,d),
        entry_point='wizluk.envs.gridworld:GridWorldEnv',
        max_episode_steps=1000,
        kwargs={
            'dimension': d,
            'terminal_def': "moving_diag"
        }
    )
    for s in range(10):
        register(
            id='GridWorld-{}x{}-initS{}-v3'.format(d,d,s),
            entry_point='wizluk.envs.gridworld:GridWorldEnv',
            max_episode_steps=1000,
            kwargs={
                'dimension': d,
                'initState': s,
                'terminal_def': "moving_diag"
            }
        )

for d in [ 4, 8, 10, 20, 50, 16, 32, 64, 100, 1000 ] :
    register(
        id='GridWorld-{}x{}-v4'.format(d,d),
        entry_point='wizluk.envs.gridworld:GridWorldEnv',
        max_episode_steps=1000,
        kwargs={
            'dimension': d,
            'terminal_def': "boundary"
        }
    )

for d in [ 4, 8, 10, 20, 50, 16, 32, 64, 100, 1000 ] :
    register(
        id='GridWorld-{}x{}-v5'.format(d,d),
        entry_point='wizluk.envs.gridworld:GridWorldEnv',
        max_episode_steps=1000,
        kwargs={
            'dimension': d,
            'terminal_def': "center",
            'obstacles': "True"
        }
    )

    for s in range(10):
        register(
            id='GridWorld-{}x{}-initS{}-v5'.format(d,d,s),
            entry_point='wizluk.envs.gridworld:GridWorldEnv',
            max_episode_steps=1000,
            kwargs={
                'dimension': d,
                'initState': s,
                'terminal_def': "center",
                'obstacles': "True"
            }
        )

for d in [ 4, 8, 10, 20, 50, 16, 32, 64, 100, 1000 ] :
    register(
        id='CTP-{}x{}-v1'.format(d,d),
        entry_point='wizluk.envs.CanadianTravellersProblem:CTPEnv',
        max_episode_steps=1000,
        kwargs={
            'dimension': d,
            'terminal_def': "center",
            'obstacles': "True"
        }
    )

    for s in range(10):
        register(
            id='CTP-{}x{}-initS{}-v1'.format(d,d,s),
            entry_point='wizluk.envs.CanadianTravellersProblem:CTPEnv',
            max_episode_steps=1000,
            kwargs={
                'dimension': d,
                'initState': s,
                'terminal_def': "center",
                'obstacles': "True"
            }
        )

register(
    id='WalkBot-v0',
    entry_point = 'wizluk.envs.walker:WalkBotSCEnv',
    max_episode_steps=100,
    kwargs={'cost_function' : 'shortest_path'},
)

register(
    id='WalkBot-v1',
    entry_point = 'wizluk.envs.walker:WalkBotSCEnv',
    max_episode_steps=100,
    kwargs={'done_on_invalid':False,
            'cost_function' : 'shortest_path'
    },
)

register(
    id='WalkBot-v2',
    entry_point = 'wizluk.envs.walker:WalkBotSCEnv',
    max_episode_steps=100,
    kwargs={'done_on_invalid':False,
            'cost_function' : 'QR'
    },
)

register(
    id='WalkBot-v3',
    entry_point = 'wizluk.envs.walker:WalkBotSCEnv',
    max_episode_steps=100,
    kwargs={'done_on_invalid':False,
            'cost_function' : 'QR',
            'perturb_velocities': True,
            'sigma_vx': 0.05,
            'sigma_vy': 0.05
    },
)


register(
    id='WalkBot-RandomInit-v0',
    entry_point = 'wizluk.envs.walker:WalkBotSCEnv',
    max_episode_steps=100,
    kwargs={'cost_function' : 'QR',
        'random_initial_state' : True
    },
)

register(
    id='WalkBot-RandomInit-v1',
    entry_point = 'wizluk.envs.walker:WalkBotSCEnv',
    max_episode_steps=100,
    kwargs={'done_on_invalid':False,
            'random_initial_state': True,
            'cost_function' : 'QR',
            'perturb_velocities': True,
            'sigma_vx': 0.05,
            'sigma_vy': 0.05
    },
)



register(
    id='ContinuousWalkBot-v0',
    entry_point = 'wizluk.envs.walker:WalkBotEnv',
    max_episode_steps=100,
    kwargs={'cost_function' : 'QR'
    },
)

register(
    id='ContinuousWalkBot-v1',
    entry_point = 'wizluk.envs.walker:WalkBotEnv',
    max_episode_steps=100,
    kwargs={'cost_function' : 'QR',
        'perturb_velocities': True,
        'sigma_vx': 0.05,
        'sigma_vy': 0.05
    },
)

register(
    id='ContinuousWalkBot-RandomInit-v0',
    entry_point = 'wizluk.envs.walker:WalkBotEnv',
    max_episode_steps=100,
    kwargs={'cost_function' : 'QR',
        'random_initial_state' : True
    },
)

register(
    id='ContinuousWalkBot-RandomInit-v1',
    entry_point = 'wizluk.envs.walker:WalkBotEnv',
    max_episode_steps=100,
    kwargs={'cost_function' : 'QR',
        'random_initial_state': True,
        'perturb_velocities': True,
        'sigma_vx': 0.05,
        'sigma_vy': 0.05
    },
)

register(
    id='MountainCarContinuous-v1',
    entry_point = 'wizluk.envs.mountain_car_QR:Continuous_MountainCarEnv',
    max_episode_steps=10000,
    kwargs = {},
)

register(
    id='RechtLQR-v0',
    entry_point = 'wizluk.envs.recht_lqr:LQR_Env',
    max_episode_steps=1000,
    kwargs = {}
)

register(
    id='Bertsekas671-000-v1',
    entry_point='envs.scalar:Bertsekas671_Env',
    kwargs = {
        'w_mu': 0.0,
        'w_sigma': 1.0,
        'x0': 75.0
    })

register(
    id='Antishape-v1',
    entry_point = 'wizluk.envs.antishape:Antishape_Env',
    kwargs = {
        'num_states' : 100
    }
)

for d in [ 10, 50, 100, 200, 500 ] :
    register(
        id='Antishape-{}-v1'.format(d),
        entry_point = 'wizluk.envs.antishape:Antishape_Env',
        max_episode_steps= 100000,
        kwargs = {
            'num_states' : d
        }
    )

    for s in range(10):
        register(
            id='Antishape-{}-initS{}-v1'.format(d,s),
            entry_point = 'wizluk.envs.antishape:Antishape_Env',
            max_episode_steps= 100000,
            kwargs = {
                'num_states' : d,
                'initState': s
            }
        )

for d in [ 10, 50, 100, 200, 500 ] :
    register(
        id='Antishape-{}-v2'.format(d),
        entry_point = 'wizluk.envs.antishape_v2:Antishape_EnvV2',
        max_episode_steps= 100000,
        kwargs = {
            'num_states' : d
        }
    )

    for s in range(10):
        register(
            id='Antishape-{}-initS{}-v2'.format(d,s),
            entry_point = 'wizluk.envs.antishape_v2:Antishape_EnvV2',
            max_episode_steps= 100000,
            kwargs = {
                'num_states' : d,
                'initState': s
            }
        )

register(
    id='Combolock-v1',
    entry_point = 'wizluk.envs.combolock:Combolock_Env',
    kwargs = {
        'num_states' : 100
    }
)

for d in [ 10, 50, 100, 200, 500 ] :
    register(
        id='Combolock-{}-v1'.format(d),
        entry_point = 'wizluk.envs.combolock:Combolock_Env',
        max_episode_steps= 100000,
        kwargs = {
            'num_states' : d
        }
    )

    for s in range(10):
        register(
            id='Combolock-{}-initS{}-v1'.format(d,s),
            entry_point = 'wizluk.envs.combolock:Combolock_Env',
            max_episode_steps= 100000,
            kwargs = {
                'num_states' : d,
                'initState': s
            }
        )

for d in [ 10, 50, 100, 200, 500 ] :
    register(
        id='Combolock-{}-v2'.format(d),
        entry_point = 'wizluk.envs.combolock_v2:Combolock_EnvV2',
        max_episode_steps= 100000,
        kwargs = {
            'num_states' : d
        }
    )

    for s in range(10):
        register(
            id='Combolock-{}-initS{}-v2'.format(d,s),
            entry_point = 'wizluk.envs.combolock_v2:Combolock_EnvV2',
            max_episode_steps= 100000,
            kwargs = {
                'num_states' : d,
                'initState': s
            }
        )


register(
    id='LunarLanderContinuous-v3',
    entry_point='wizluk.envs.lunar_lander:LunarLanderContinuous',
    max_episode_steps=1000,
    reward_threshold=200,
)


# Scoreboard registration
# ==========================
#add_group(
#    id= 'gridworld',
#    name= 'GridWorld',
#    description= 'Sutton & Barto classic Gridworld environment.'
#)

#for d in [ 4, 8, 16, 32, 64 ] :
#    add_task(
#        id='{}/GridWorld-{}x{}'.format(USERNAME,d,d),
#        group='gridworld',
#        summary='Sutton and Barto Grid world, {}x{}.'.format(d),
#        description="""
#        """)

# Atari - code modified from https://github.com/openai/gym/blob/master/gym/envs/__init__.py
# ----------------------------------------

# # print ', '.join(["'{}'".format(name.split('.')[0]) for name in atari_py.list_games()])
for game in ['air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
    'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival',
    'centipede', 'chopper_command', 'crazy_climber', 'defender', 'demon_attack', 'double_dunk',
    'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar',
    'hero', 'ice_hockey', 'jamesbond', 'journey_escape', 'kangaroo', 'krull', 'kung_fu_master',
    'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan',
    'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing',
    'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down',
    'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']:
    for obs_type in ['image', 'ram']:
        # space_invaders should yield SpaceInvaders-v0 and SpaceInvaders-ram-v0
        name = ''.join([g.capitalize() for g in game.split('_')])
        if obs_type == 'ram':
            name = '{}-ram'.format(name)

        nondeterministic = False
        if game == 'elevator_action' and obs_type == 'ram':
            # ElevatorAction-ram-v0 seems to yield slightly
            # non-deterministic observations about 10% of the time. We
            # should track this down eventually, but for now we just
            # mark it as nondeterministic.
            nondeterministic = True

        # Standard Deterministic (as in the original DeepMind paper)
        if game == 'space_invaders':
            frameskip = 3
        else:
            frameskip = 4

        # settings from the Machado 2017 paper: "Revisiting the Arcade Learning Environment:
        # Evaluation Protocols and Open Problems for General Agents".
        register(
            id='{}-machado-sticky-v0'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': 5, 'repeat_action_probability': 0.25, 'full_action_space': True},
            max_episode_steps=18000,
            nondeterministic=nondeterministic,
        )

        register(
            id='{}-machado-v0'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': 5, 'repeat_action_probability': 0, 'full_action_space': True},
            max_episode_steps=18000,
            nondeterministic=nondeterministic,
        )

        # settings from the Bandres 2018 paper: "Planning with Pixels in (Almost) Real Time".
        register(
            id='{}-bandres-v0'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': 15, 'repeat_action_probability': 0, 'full_action_space': True},
            max_episode_steps=18000,
            nondeterministic=nondeterministic,
        )

        register(
            id='{}-noFrameSkip-v0'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': 1, 'repeat_action_probability': 0},
            max_episode_steps=18000,
            nondeterministic=nondeterministic,
        )
