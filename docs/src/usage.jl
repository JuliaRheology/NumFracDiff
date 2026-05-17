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

method = GL()
Problem = NumDiffProblem(dt=dt,order=1.0,n=length(y),method=method)
WorkSpace = init_workspace(Problem)
compute!(Problem.method,WorkSpace,y,Problem)

plot!(plt, x, WorkSpace.deriv,
      linestyle=:dash,
      color=:red,
      label="f'(x)", 
      framestyle = :box) 
xlabel!("x")
ylabel!("y")
#!nb plt #hide

update_order!(Problem,WorkSpace,0.5)
compute!(Problem.method,WorkSpace,y,Problem)
plt1 = plot(size = (500, 500))
plot!(plt1, x, y,
      linestyle=:dash,
      color=:blue,
      label="f(x)", 
      framestyle = :box) 
plot!(plt1, x, WorkSpace.deriv,
      linestyle=:dash,
      color=:green,
      label="f^0.5(x)", 
      framestyle = :box) 
compute!(Problem.method,WorkSpace,y,Problem)
plot!(plt1, x, WorkSpace.deriv,
      linestyle=:dash,
      color=:green,
      label="f^0.5(x)", 
      framestyle = :box) 
xlabel!("x")
ylabel!("y")
#!nb plt1 #hide

y= x .* 2
update_order!(Problem,WorkSpace,-0.5)
compute!(Problem.method,WorkSpace,y,Problem)
plt2 = plot(size = (500, 500))
plot!(plt2, x, y,
      linestyle=:dash,
      color=:blue,
      label="f(x)", 
      framestyle = :box) 
plot!(plt2, x, WorkSpace.deriv,
      linestyle=:dash,
      color=:green,
      label="f^0.5(x)", 
      framestyle = :box) 
compute!(Problem.method,WorkSpace,y,Problem)
plot!(plt2, x, WorkSpace.deriv,
      linestyle=:dash,
      color=:green,
      label="f^0.5(x)", 
      framestyle = :box) 
xlabel!("x")
ylabel!("y")
#!nb plt2 #hide