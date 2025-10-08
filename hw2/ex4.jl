

include("rk3.jl")
using .Rk3
# Função que define a equação diferencial
f(t, u) = u - t^2 + 1
f2_exact(t) = (t+1)^2 - 0.5 * exp(t)

# Condição inicial e intervalo de tempo
u0 = 0.5
tspan = (0.0, 2.0)
h = 0.1
x_vals = collect(range(0.0, 2.0; step=0.1))
y_vals = f2_exact.(x_vals)


# Coeficientes personalizados para Runge-Kutta de 3ª ordem
A = [0.0  0.0  0.0;
     1/2  0.0  0.0;
     0.0  1.0  0.0]

c = [1/6, 2/3, 1/6]

b = [0.0, 1/2, 1.0]

# Resolver o problema com o método Runge-Kutta de 3ª ordem com coeficientes personalizados
u, u_vect= Rk3.runge_kutta_3(f, u0, tspan, h, A, b, c)

u_vect_err = abs.(y_vals .- u_vect)

println("Solução final u(t=1) = $u")


#--- Plots com GLMakie
using GLMakie

GLMakie.activate!()
fig1 = Figure(size=(1100, 900))

ax11 = Axis(fig1[1, 1], title="RK3, intervalo [0, 2], h=0.1", xlabel="t", ylabel="u(t)")
lines!(ax11, x_vals, y_vals, label="Exata", linewidth=4)
scatterlines!(ax11, x_vals, u_vect, label="RK3", markersize=8, color=:black)
axislegend(ax11, position=:rb)

ax12 = Axis(fig1[2, 1], title="Erros em relação a solução analítica", xlabel="t", ylabel="Erros")
scatterlines!(ax12, x_vals, u_vect_err, label="RK3", markersize=8, color=:black)

axislegend(ax12, position=:rb)

save("figs/ex41.png", fig1)