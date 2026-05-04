# # Usage


using NumFracDiff
using Plots


#Commented test code
dt=0.001
x = collect(0:dt:10.0)
y = x.^2
plt = plot(size = (500, 500))
plot!(plt, x, y,
      linestyle=:dash,
      color=:blue,
      label="f(x)", 
      framestyle = :box) 

method = GLFFT()
Problem = NumDiffProblem(dt=dt,order=1.0,n=length(y),method=method)
WorkSpace = init_workspace(Problem)
compute!(Problem.method,WorkSpace,y,Problem)

plot!(plt, x, WorkSpace.deriv,
      linestyle=:dash,
      color=:red,
      label="f'(x)", 
      framestyle = :box) 

update_order!(Problem,WorkSpace,0.5)
compute!(Problem.method,WorkSpace,y,Problem)

plot!(plt, x, WorkSpace.deriv,
      linestyle=:dash,
      color=:green,
      label="f^0.5(x)", 
      framestyle = :box) 
xlabel!("x")
ylabel!("y")
#!nb plt #hide

#md # !!! note "Note"
#md #     Note