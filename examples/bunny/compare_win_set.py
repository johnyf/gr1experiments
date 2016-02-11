from dd import cudd


def assert_equal():
    """Assert equal automata and winning sets."""
    b = cudd.BDD()
    u_gr1x = cudd.load('winning_set', b)
    u_slugs = b.load('winning_set_bdd.txt')
    # load slugs output
    env_action_slugs = b.load('env_action_slugs.txt')
    sys_action_slugs = b.load('sys_action_slugs.txt')
    assumption_0_slugs = b.load('assumption_0_slugs.txt')
    goal_0_slugs = b.load('goal_0_slugs.txt')
    # load gr1x output
    env_action_gr1x = b.load('env_action_gr1x.txt')
    sys_action_gr1x = b.load('sys_action_gr1x.txt')
    assumption_0_gr1x = b.load('assumption_0_gr1x.txt')
    goal_0_gr1x = b.load('goal_0_gr1x.txt')
    # assert same
    assert env_action_slugs == env_action_gr1x
    assert sys_action_slugs == sys_action_gr1x
    assert assumption_0_slugs == assumption_0_gr1x
    assert goal_0_slugs == goal_0_gr1x
    assert u_gr1x == u_slugs
    # cleanup
    del env_action_slugs, env_action_gr1x
    del sys_action_slugs, sys_action_gr1x
    del assumption_0_slugs, assumption_0_gr1x
    del goal_0_slugs, goal_0_gr1x
    del u_gr1x, u_slugs
    print('rejoice: all equal')


if __name__ == '__main__':
    assert_equal()
