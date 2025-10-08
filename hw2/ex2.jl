
# --- Bibliotecas necessárias
using DifferentialEquations, TaylorIntegration
using OrdinaryDiffEqTaylorSeries
using Printf

# --- Arredondamento simétrico a 6 casas decimais ------------------------------
r6(x::Real) = round(x, RoundNearestTiesAway; digits=6)
fmt(x) = @sprintf("%.6f", x)

# --- Problema ODE -------------------------------------------------------------
# y' = -y + t,  y(0) = 1, t ∈ [0,1]
f(u, p, t) = u - t^2 + 1
u0 = 0.5
T_MIN, T_MAX = 0.0, 3.0
Δt1 = 0.1 # garante Float64
Δt2 = 0.05

tspan=(T_MIN, T_MAX)
grid1 = range(T_MIN, T_MAX; step=Δt1)
grid2 = range(T_MIN, T_MAX; step=Δt2)

prob = ODEProblem(f, u0, tspan)

# Solução exata para comparação: y(t) = t - 1 + 2e^{-t}
y_exact(x) = (x+1)^2 - 0.5 * exp(x)

# --- Pontos da solução analítica
x_vals1 = collect(grid1)
y_vals1 = y_exact.(x_vals1)

x_vals2 = collect(grid2)
y_vals2 = y_exact.(x_vals2)

# -- Soluções abaixo: primeiro com passo h=0.1, depois 0.05

# --- Solução por Runge-Kutta 4 ordem
rk4_sol1 = solve(prob, RK4(); adaptive=false, dt=Δt1, reltol=1e-8, abstol=1e-8, saveat=grid1)
rk4_sol2 = solve(prob, RK4(); adaptive=false, dt=Δt2, reltol=1e-8, abstol=1e-8, saveat=grid2)

# --- Solução por Runge-kutta 2 ordem
rk2_sol1 = solve(prob, Heun(); adaptive=false, dt=Δt1, reltol=1e-8, abstol=1e-8, saveat=grid1)
rk2_sol2 = solve(prob, Heun(); adaptive=false, dt=Δt2, reltol=1e-8, abstol=1e-8, saveat=grid2)

# --- Solução por Euler explícito
exp_euler1 = solve(prob, Euler(); adaptive=false, dt=Δt1, reltol=1e-8, abstol=1e-8, saveat=grid1)
exp_euler2 = solve(prob, Euler(); adaptive=false, dt=Δt2, reltol=1e-8, abstol=1e-8, saveat=grid2)


# --- erros vs. soluções exatas ---
y_rk41 = rk4_sol1.u
y_rk21     = rk2_sol1.u
y_euler1 = exp_euler1.u

y_rk42 = rk4_sol2.u
y_rk22     = rk2_sol2.u
y_euler2 = exp_euler2.u

abs_err_rk41 = abs.(y_rk41 .- y_vals1)
abs_err_rk21     = abs.(y_rk21     .- y_vals1)
abs_err_euler1  = abs.(y_euler1  .- y_vals1)

abs_err_rk42 = abs.(y_rk42 .- y_vals2)
abs_err_rk22     = abs.(y_rk22     .- y_vals2)
abs_err_euler2  = abs.(y_euler2  .- y_vals2)



# --- Plots com GLMakie --------------------------------------------------------
using GLMakie
GLMakie.activate!()

# --- Figura para o passo 0.1
fig1 = Figure(size = (1100, 900))
# --- Figura para o passo 0.05
fig2 = Figure(size = (1100, 900))

ax11 = Axis(fig1[1, 1], title="Soluções em [0,3], h = 0.1", xlabel = "x", ylabel = "y(x)")
lines!(ax11, x_vals1, y_vals1, label="Exata", linewidth=3, color=:red)
scatterlines!(ax11, x_vals1, rk4_sol1.u, label="RK4", markersize=4)
scatterlines!(ax11, x_vals1, rk2_sol1.u, label="RK2")
scatterlines!(ax11, x_vals1, exp_euler1.u, label="Euler exp")
axislegend(ax11, position=:rb)

ax21 = Axis(fig1[2, 1], title="Erro absoluto", xlabel="x", ylabel="|erro|")
lines!(ax21, x_vals1, abs_err_rk41, label="RK4", linewidth=2)
lines!(ax21, x_vals1, abs_err_rk21,     label="RK2")
scatter!(ax21, [1], [rk4_sol1(2.0) - y_exact(2.0)], marker = :circle, color=:orange)
axislegend(ax21, position=:rb)

ax31 = Axis(fig1[3, 1], title="Diferença entre erros", xlabel="x", ylabel="err_taylor - err_rk")
lines!(ax31, x_vals1, abs.(abs_err_rk41 .- abs_err_rk21), label="err")
axislegend(ax31, position=:rb)

ax12 = Axis(fig2[1, 1], title="Soluções em [0,3], h = 0.05", xlabel = "x", ylabel = "y(x)")
lines!(ax12, x_vals2, y_vals2, label="Exata", linewidth=3, color=:red)
scatterlines!(ax12, x_vals2, rk4_sol2.u, label="RK4", markersize=4)
scatterlines!(ax12, x_vals2, rk2_sol2.u, label="RK2")
scatterlines!(ax12, x_vals2, exp_euler2.u, label="Euler exp")
axislegend(ax12, position=:rb)

ax22 = Axis(fig2[2, 1], title="Erro absoluto", xlabel="x", ylabel="|erro|")
lines!(ax22, x_vals2, abs_err_rk42, label="RK4", linewidth=2)
lines!(ax22, x_vals2, abs_err_rk22,     label="RK2")
scatter!(ax22, [1], [rk4_sol2(2.0) - y_exact(2.0)], marker = :circle, color=:orange)
axislegend(ax22, position=:rb)

ax32 = Axis(fig2[3, 1], title="Diferença entre erros", xlabel="x", ylabel="err_taylor - err_rk")
lines!(ax32, x_vals2, abs.(abs_err_rk42 .- abs_err_rk22), label="err")
axislegend(ax32, position=:rb)

save("figs/ex21.png", fig1)
save("figs/ex22.png", fig2)
println("Figuras salvas em /figs")

println("valor da solução no ponto 2.0: ")
println("Runge-Kutta 4a ordem h=0.1 -  $(rk4_sol1(2))")
println("Runge-Kutta 4a ordem h=0.05 - $(rk4_sol2(2))")
println("Runge-Kutta 2a ordem h=0.1 -  $(rk2_sol1(2))")
println("Runge-Kutta 2a ordem h=0.05 - $(rk2_sol2(2))")
println("Solução exata -               $(y_exact(2))")
