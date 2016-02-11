from dd import cudd


b = cudd.BDD()
u_gr1x = cudd.load('winning_set', b)
u_slugs = b.load('winning_set_bdd.txt')

env_action_slugs = b.load('env_action_slugs.txt')
sys_action_slugs = b.load('sys_action_slugs.txt')
assumption_0_slugs = b.load('assumption_0_slugs.txt')
goal_0_slugs = b.load('goal_0_slugs.txt')

env_action_gr1x = b.load('env_action_gr1x.txt')
sys_action_gr1x = b.load('sys_action_gr1x.txt')
assumption_0_gr1x = b.load('assumption_0_gr1x.txt')
goal_0_gr1x = b.load('goal_0_gr1x.txt')

assert env_action_slugs == env_action_gr1x
assert sys_action_slugs == sys_action_gr1x
assert assumption_0_slugs == assumption_0_gr1x
assert goal_0_slugs == goal_0_gr1x

if u_gr1x == u_slugs:
    print('Winning set is the same.')
else:
    print('Different winning sets!')
del u_gr1x, u_slugs
