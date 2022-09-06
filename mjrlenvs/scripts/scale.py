
def scale_actions(action_val, action_specs, scale_range = [-1,+1]): 
    action_scales = [0]*len(action_val)
    for i,aname in enumerate(action_specs.keys()):
        arange = action_specs[aname]
        action_scales[i] = arange[0] + (action_val[i]-scale_range[0])*(arange[1]-arange[0])/(scale_range[1]-scale_range[0]) 
    return action_scales


def scale_state(s, state_range, scale_range=[0,1]): 
    if len(s) == 1: 
        s[0] = (scale_range[1]-scale_range[0])*(s[0]-state_range[0])/(state_range[1]-state_range[0]) + scale_range[0]
    else:
        for i, r in enumerate(state_range): 
            s[i] = (scale_range[1]-scale_range[0])*(s[i]-r[0])/(r[1]-r[0]) + scale_range[0]
    return s

