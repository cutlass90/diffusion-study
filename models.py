def simple_linear_sceduler(T, diffusion_steps):
    # T is tensor of shape batch x 1 of type int
    # returns value that should be multiplied to img
    x = 1 - T/(diffusion_steps+1)
    return x