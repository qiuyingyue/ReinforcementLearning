# ReinforcementLearning

* each person write and maintain its own python file

* we can add  algorithm class and environment class and try different ones

* for environment class _env_, we have

    * if we use built-in environment in gym such as mujoco, the interface is like the follows:
    * env.step(), env.reset(), env.render(), env.destroy(), env.action_space.sample()
    * if we write our own environment, please maintain the same interface 

* for algorithm class _rl_agent_, we have

    * rl_agent.choose_action(), rl_agent.store_transition(), and so on 

* There are some codes from online resources for reference in the directory of _example_
