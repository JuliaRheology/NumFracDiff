# # Architecture

# Normal text

using NumFracDiff
using Plots

# ## SubTitle

#text

##Commented test code
x = 0:0.4:1.0
y = x.*2
plt = plot(size = (500, 500))
plot!(plt, x, y,
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
