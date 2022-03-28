from baseline import *

def make_env(env, stack_frames=True, episodic_life=True, clip_rewards=False, scale=False):
    # only one life
    if episodic_life:
        env = EpisodicLifeEnv(env) 
    env = NoopResetEnv(env, noop_max=30)
    # skip four frame  
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    # downsample, gray -> 84 * 84
    env = WarpFrame(env)
    if stack_frames:
        # combine four frame into one state
        env = FrameStack(env, 4)
    if clip_rewards:
        env = ClipRewardEnv(env)
    return env