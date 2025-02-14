"""
    flux_rusanov(u_ll, u_rr, orientation::Integer, equations::Euler1D)

Compute the Rusanov flux for the Euler1D equations.
"""
function flux_rusanov(u_ll, u_rr, orientation::Integer, equations)
    lamba_l = max_abs_speeds(u_ll, equations)[1]
    lamba_r = max_abs_speeds(u_rr, equations)[1]
    lamba = max(lamba_l, lamba_r)
    f_ll, f_rr = flux(u_ll, orientation, equations), flux(u_rr, orientation, equations)
    return 0.5f0 * (f_ll + f_rr - lamba * (u_rr - u_ll))
end