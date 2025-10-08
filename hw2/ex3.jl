
include("rk2.jl")
using .Rk2

# Função que define a equação diferencial
f1(t, u) = - u + t
f2(t, u) = u - t^2 + 1

f1_exact(t) = t - 1 + 2 * exp(-t)
f2_exact(t) = (t+1)^2 - 0.5 * exp(t)
# Condição inicial e intervalo de tempo
u0_1 = 1.0
tspan1 = (0.0, 2.0)
h = 0.1
x_vals1 = collect(range(0.0, 2.0; step=h))
y_vals1 = f1_exact.(x_vals1)

u0_2 = 0.5
tspan2 = (0.0, 4.0)
x_vals2 = collect(range(0.0, 4.0; step=h))
y_vals2 = f2_exact.(x_vals2)


# Coeficientes personalizados para Runge-Kutta de segunda ordem
A1 = [0.0  0.0;
     0.5  0.0]

c1 = [0.0, 1.0]

b1 = [0.0, 0.5]

# Resolver o problema com o método Runge-Kutta de segunda ordem personalizado

u_1, u_vect_1 = Rk2.runge_kutta_2(f1, u0_1, tspan1, h, A1, b1, c1)
u_2, u_vect_2 = Rk2.runge_kutta_2(f2, u0_2, tspan2, h, A1, b1, c1)

println("Soluções finais $u_1 e $u_2")

A2 = [0.0  0.0;
     2/3  0.0]

c2 = [0.25, 0.75]

b2 = [0.0, 2/3]

u_3, u_vect_3 = Rk2.runge_kutta_2(f1, u0_1, tspan1, h, A2, b2, c2)
u_4, u_vect_4 = Rk2.runge_kutta_2(f2, u0_2, tspan2, h, A2, b2, c2)

println("Soluções finais $u_3 e $u_4")

A3 = [0.0  0.0;
     0.75  0.0]

c3 = [1/3, 2/3]

b3 = [0.0, 0.75]


u_5, u_vect_5 = Rk2.runge_kutta_2(f1, u0_1, tspan1, h, A3, b3, c3)
u_6, u_vect_6 = Rk2.runge_kutta_2(f2, u0_2, tspan2, h, A3, b3, c3)

println("Soluções finais $u_5 e $u_6")

#--- Cálculo dos erros
u_vect_1_err = abs.(y_vals1 .- u_vect_1)
u_vect_3_err = abs.(y_vals1 .- u_vect_3)
u_vect_5_err = abs.(y_vals1 .- u_vect_5)

u_vect_2_err = abs.(y_vals2 .- u_vect_2)
u_vect_4_err = abs.(y_vals2 .- u_vect_4)
u_vect_6_err = abs.(y_vals2 .- u_vect_6)



#--- Plots com GLMakie
using GLMakie

GLMakie.activate!()
fig1 = Figure(size=(1100, 900))

ax11 = Axis(fig1[1, 1], title="Variações do RK2 no ex1, intervalo [0, 4], h=0.1", xlabel="t", ylabel="u(t)")
lines!(ax11, x_vals1, y_vals1, label="Exata", linewidth=4)
scatterlines!(ax11, x_vals1, u_vect_1, label="Ponto médio", markersize=18, color=:black)
scatterlines!(ax11, x_vals1, u_vect_3, label="Heun", markersize=14, color=:green)
scatterlines!(ax11, x_vals1, u_vect_5, label="Ralston", color=:orange)
axislegend(ax11, position=:rb)

ax12 = Axis(fig1[2, 1], title="Erros em relação a solução analítica", xlabel="t", ylabel="Erros")
scatterlines!(ax12, x_vals1, u_vect_1_err, label="Ponto médio", markersize=18, color=:black)
scatterlines!(ax12, x_vals1, u_vect_3_err, label="Heun", markersize=14, color=:green)
scatterlines!(ax12, x_vals1, u_vect_5_err, label="Ralston", color=:orange)
axislegend(ax12, position=:rb)

ax13 = Axis(fig1[3, 1], title="Diferenças entre erros das soluções duas a duas" )
scatterlines!(ax13, x_vals1, abs.(u_vect_1_err .- u_vect_3_err), label="PM - Heun")
scatterlines!(ax13, x_vals1, abs.(u_vect_1_err .- u_vect_5_err), label="PM - Ralston")
scatterlines!(ax13, x_vals1, abs.(u_vect_3_err .- u_vect_5_err), label="Heun - Ralston")
axislegend(ax13, poistion=:lt)

save("figs/ex31.png", fig1)

fig2 = Figure(size=(1100, 900))

ax21 = Axis(fig2[1, 1], title="Variações do RK2 no ex2, intervalo [0, 4], h=0.1", xlabel="t", ylabel="u(t)")
lines!(ax21, x_vals2, y_vals2, label="Exata")
scatterlines!(ax21, x_vals2, u_vect_2, label="Ponto médio")
scatterlines!(ax21, x_vals2, u_vect_4, label="Heun")
scatterlines!(ax21, x_vals2, u_vect_6, label="Ralston")
axislegend(ax21, position=:lt)

ax22 = Axis(fig2[2, 1], title="Erros em relação a solução analítica", xlabel="t", ylabel="Erros")

scatterlines!(ax22, x_vals2, u_vect_2_err, label="Ponto médio")
scatterlines!(ax22, x_vals2, u_vect_4_err, label="Heun")
scatterlines!(ax22, x_vals2, u_vect_6_err, label="Ralston")
axislegend(ax22, position=:lt)


save("figs/ex32.png", fig2)