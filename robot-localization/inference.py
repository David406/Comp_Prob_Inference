#!/usr/bin/env python
# inference.py
# Base code by George H. Chen (georgehc@mit.edu) -- updated 10/18/2016
import collections
import sys

import graphics
import numpy as np
import robot


# Throughout the code, we use these variables.
# Do NOT change these (but you'll need to use them!):
# - all_possible_hidden_states: a list of possible hidden states
# - all_possible_observed_states: a list of possible observed states
# - prior_distribution: a distribution over states
# - transition_model: a function that takes a hidden state and returns a
#     Distribution for the next state
# - observation_model: a function that takes a hidden state and returns a
#     Distribution for the observation from that hidden state
all_possible_hidden_states = robot.get_all_hidden_states()
all_possible_observed_states = robot.get_all_observed_states()
prior_distribution = robot.initial_distribution()
transition_model = robot.transition_model
observation_model = robot.observation_model


# You may find this function helpful for computing logs without yielding a
# NumPy warning when taking the log of 0.
def careful_log(x):
    # computes the log of a non-negative real number
    if x == 0:
        return -np.inf
    else:
        return np.log(x)


# -----------------------------------------------------------------------------
# Functions for you to implement
#

def forward_backward(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    robot.py and see how it is used in both robot.py and the function
    generate_data() above, and the i-th Distribution should correspond to time
    step i
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = prior_distribution
    # TODO: Compute the forward messages
    for t in range(1, num_time_steps):
        forward_messages[t] = robot.Distribution()
        for state, p in forward_messages[t-1].items():
            observation = observations[t-1]
            if observation is not None:
                p_observation = observation_model(state)[observation]
            else:
                p_observation = 1.0    # This means do not include observation
            for next_state, p_next_state in transition_model(state).items():
                forward_messages[t][next_state] += p * p_observation * p_next_state
        forward_messages[t].renormalize()
    
    backward_messages = [None] * num_time_steps
    # TODO: Compute the backward messages
    
    # Compute backward transition model
    backward_transition_model = collections.defaultdict(robot.Distribution)
    for state in all_possible_hidden_states:
        for next_state, p in transition_model(state).items():
            backward_transition_model[next_state][state] += p
    
    backward_messages[-1] = robot.Distribution()
    for state in all_possible_hidden_states:
        backward_messages[-1][state] = 1;
        
    for t in reversed(range(num_time_steps-1)):
        backward_messages[t] = robot.Distribution()
        for state, p in backward_messages[t+1].items():
            possible_previous_states = backward_transition_model[state]
            observation = observations[t+1]
            if observation is not None:
                p_observation = observation_model(state)[observation]
            else:
                p_observation = 1.0  # This means do not include observation
            for previous_state, p_previous_state in possible_previous_states.items():
                backward_messages[t][previous_state] += p * p_observation * p_previous_state
        backward_messages[t].renormalize()
        
    marginals = [None] * num_time_steps 
    # TODO: Compute the marginals
    for t in range(num_time_steps):
        marginals[t] = robot.Distribution()
        observation = observations[t]
        for state in all_possible_hidden_states:
            if observation is not None:
                p_observation = observation_model(state)[observation]
            else:
                p_observation = 1.0
            marginals[t][state] = forward_messages[t][state] * backward_messages[t][state] * p_observation
        marginals[t].renormalize()
    
    return marginals


def Viterbi(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #
    

    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps # remove this
    
    min_sum_messages = [prior_distribution]
    for state, p in prior_distribution.items():
        min_sum_messages[0][state] = (-careful_log(p), None)
    
    x_end = {}
    for t in range(num_time_steps):
        next_min_sum_message = {}
        for state, (p, _) in min_sum_messages[t].items():
            observation = observations[t]
            if observation is not None:
                p_observation = observation_model(state)[observation]
            else:
                p_observation = 1
            if t != num_time_steps-1:
                for next_state, p_next_state in transition_model(state).items():
                    m = - careful_log(p_next_state) - careful_log(p_observation) + p
                    if m < next_min_sum_message.setdefault(next_state, (np.inf, None))[0]:
                        next_min_sum_message[next_state] = (m, state)
            else:
                m = -careful_log(p_observation) + p
                if m < x_end.setdefault(state, np.inf):
                    x_end[state] = m
                    estimated_hidden_states[t] = state
        if t != num_time_steps-1:
            min_sum_messages.append(next_min_sum_message)
            
    # Backtrace to decide all hidden states
    for t in reversed(range(1, num_time_steps)):
        backward_coming_state = estimated_hidden_states[t]
        estimated_hidden_states[t-1] = min_sum_messages[t][backward_coming_state][1]
        
    return estimated_hidden_states


def second_best(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #


    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps # remove this

    return estimated_hidden_states


# -----------------------------------------------------------------------------
# Generating data from the hidden Markov model
#

def generate_data(num_time_steps, make_some_observations_missing=False,
                  random_seed=None):
    # generate samples from this project's hidden Markov model
    hidden_states = []
    observations = []

    # if the random seed is not None, then this makes the randomness
    # deterministic, which may be helpful for debug purposes
    np.random.seed(random_seed)

    # draw initial state and emit an observation
    initial_state = prior_distribution.sample()
    initial_observation = observation_model(initial_state).sample()

    hidden_states.append(initial_state)
    observations.append(initial_observation)

    for time_step in range(1, num_time_steps):
        # move the robot
        prev_state = hidden_states[-1]
        new_state = transition_model(prev_state).sample()

        # maybe emit an observation
        if not make_some_observations_missing:
            new_observation = observation_model(new_state).sample()
        else:
            if np.random.rand() < .1:  # 0.1 prob. of observation being missing
                new_observation = None
            else:
                new_observation = observation_model(new_state).sample()

        hidden_states.append(new_state)
        observations.append(new_observation)

    return hidden_states, observations


# -----------------------------------------------------------------------------
# Main
#

def main():
    # flags
    make_some_observations_missing = False
    use_graphics = True
    need_to_generate_data = True

    # parse command line arguments
    for arg in sys.argv[1:]:
        if arg == '--missing':
            make_some_observations_missing = True
        elif arg == '--nographics':
            use_graphics = False
        elif arg.startswith('--load='):
            filename = arg[7:]
            hidden_states, observations = robot.load_data(filename)
            need_to_generate_data = False
            num_time_steps = len(hidden_states)

    # if no data is loaded, then generate new data
    if need_to_generate_data:
        num_time_steps = 100
        hidden_states, observations = \
            generate_data(num_time_steps,
                          make_some_observations_missing)

    print('Running forward-backward...')
    marginals = forward_backward(observations)
    print("\n")

    timestep = 99
    print("Most likely parts of marginal at time %d:" % (timestep))
    if marginals[timestep] is not None:
        print(sorted(marginals[timestep].items(),
                     key=lambda x: x[1],
                     reverse=True)[:10])
    else:
        print('*No marginal computed*')
    print("\n")

    print('Running Viterbi...')
    estimated_states = Viterbi(observations)
    print("\n")

    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states[time_step] is None:
            print('Missing')
        else:
            print(estimated_states[time_step])
    print("\n")

    print('Finding second-best MAP estimate...')
    estimated_states2 = second_best(observations)
    print("\n")

    print("Last 10 hidden states in the second-best MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states2[time_step] is None:
            print('Missing')
        else:
            print(estimated_states2[time_step])
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP estimate and true hidden " +
          "states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states2[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between second-best MAP estimate and " +
          "true hidden states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != estimated_states2[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP and second-best MAP " +
          "estimates:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    # display
    if use_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()


if __name__ == '__main__':
    main()
