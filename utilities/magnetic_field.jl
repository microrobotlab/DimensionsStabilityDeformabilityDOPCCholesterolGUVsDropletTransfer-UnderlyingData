# This script calculates the superpose field of two permanent magnets
# ===

# Importing packages
# ---
# ! Please run _init_.jl to initialize the computing environment !using Statistics
using Plots

# Functions
# ---
function singlemagnet(z, L, W, H, Br)
    f(z) = (2*z)*sqrt(4*z^2 + L^2 + W^2)
    B = (Br/pi)*(atan(L*W/f(z))-atan(L*W/f(H+z)))
    return B
end;

function magfield(z, L, W, H, Br, d)
    B_left = singlemagnet(z+d/2, L, W, H, Br)
    B_right = singlemagnet(-z+d/2, L, W, H, Br)
    B = B_left+B_right
    return B
end;

function plotField(z,m)
    p = plot(1e3z,1e3m,label="B₀ = $(round(1e3*magfield(0,params...),digits=1)) mT")
    ylims!(0,round(maximum(1e3m)))
    xlims!(1e3z[1],1e3z[end])
    ylabel!("B (mT)")
    xlabel!("z (mm)")
    return p
end;

# Example
# ---
L = 15e-3
W = 15e-3
H = 8e-3
Br = 1.4
d = 110e-3
z = 1e-3LinRange(-5,5,101)
params = (L, W, H, Br, d)

m = magfield.(z, params...)
plotField(z,m)