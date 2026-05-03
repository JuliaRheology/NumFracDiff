# # Architecture

# Normal text

using NumFracDiff
using Plots

# ## SubTitle

#text

##Commented test code
foo = timeline(t_start = 0, t_end = 10, step = 0.4)
foo = strainfunction(foo, t -> sin(t))
plt = plot(size = (500, 500))
plot!(plt, foo.t, foo.ϵ,
      linestyle=:dash,
      marker=:circle,
      markersize=8,
      color=:blue,
      label="", 
      framestyle = :box) 
xlabel!("Time")
ylabel!("Strain")
#!nb plt #hide

#md # !!! note "Note"
#md #     Note
