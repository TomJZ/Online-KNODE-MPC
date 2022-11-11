def system_change(quad, curr_t, change_dict):
    """
    param quad: quadrotor object
    param curr_t: the current time during simulation
    param change_dict: dictionary specifying changes and the corresponding time
    """
    mass_change_t = change_dict['mass'][0]
    mass_change = change_dict['mass'][1]
    mass_change_t2 = change_dict['mass2'][0]
    mass_change2 = change_dict['mass2'][1]
    if curr_t == mass_change_t:
        print("[System mass: {0}] Mass change at {1} s of {2} kg".format(quad.mass, mass_change_t, mass_change))
        quad.mass = quad.mass + mass_change
    if curr_t == mass_change_t2:
        print("[System mass: {0}] Mass change at {1} s of {2} kg".format(quad.mass, mass_change_t2, mass_change2))
        quad.mass = quad.mass + mass_change2
