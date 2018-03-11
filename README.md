# ReinforcementLearning

* each person write and maintain its own python file

* we can add  algorithm class and environment class

* for environment class _env_, we have

    * env.step(), env.reset(), env.render(), env.destroy(), env.action_space.sample()
    * if we use built-in environment in gym such as mujoco, the interface is the same

* for algorithm class _rl_agent_, we have

    *  rl_agent.choose_action(), rl_agent.store_transition()
