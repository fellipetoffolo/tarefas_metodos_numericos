using DifferentialEquations
using GLMakie
using Printf

# --- Arredondamento simétrico a 6 casas decimais ------------------------------
r6(x::Real) = round(x, RoundNearestTiesAway; digits=6)
fmt(x) = @sprintf("%.6f", x)

# --- Problema ODE -------------------------------------------------------------
# y' = cos(t),  y(0) = 2, t ∈ [0, π/2]
rhs(u, p, t) = cos(t)
u0 = 2.0
tspan = (0.0, π/2)
prob = ODEProblem(rhs, u0, tspan)

# Solução exata para comparação: y(t) = 2 + sin(t)
y_exact(t) = 2 + sin(t)

# --- Implementação de Euler melhorado------------------------------------------
# --- Função para resolver com Euler Melhorado (Heun) -------------------------
function solve_with_print_heun(prob, dt; label::AbstractString)
    t0, tf = prob.tspan
    u0 = prob.u0
    nsteps = Int(round((tf - t0) / dt))

    # Inicializar valores
    x_vals = Real[r6(t0)]
    y_vals = Real[r6(u0)]

    println("────────────────────────────────────────────────────────────")
    println(@sprintf("%s (h=%s):", label, fmt(dt)))
    println("passo |   t_(n+1)   |   y_(n+1)")
    println("────────────────────────────────────────────────────────────")

    # Solução numérica por Euler melhorado
    t = t0
    y = u0

    for k in 1:nsteps
        f1 = rhs(y, nothing, t)            # f(t_n, y_n)
        y_pred = y + dt * f1               # previsão (Euler explícito)
        f2 = rhs(y_pred, nothing, t + dt)  # f(t_n+1, y_pred)
        y_next = y + (dt / 2) * (f1 + f2)  # correção (Euler melhorado)

        # Avança o tempo e o valor
        t += dt
        y = y_next

        # Armazenar valores
        x_vals = push!(x_vals, r6(t))
        y_vals = push!(y_vals, r6(y))

        println(@sprintf("%4d | %11s | %11s", k, fmt(t), fmt(y)))
    end
    println("────────────────────────────────────────────────────────────\n")

    return x_vals, y_vals
end

function solve_with_print(prob, alg, dt; label::AbstractString)
    t0, tf = prob.tspan
    integ = init(prob, alg; dt=dt, adaptive=false)
    nsteps = Int(round((tf - t0) / dt))

    # Armazenar valores arredondados para usar nos gráficos/erros
    x_vals = Real[r6(integ.t)]
    y_vals = Real[r6(integ.u)]

    println("────────────────────────────────────────────────────────────")
    println(@sprintf("%s (h=%s):", label, fmt(dt)))
    println("passo |   t_(n+1)   |   y_(n+1)")
    println("────────────────────────────────────────────────────────────")

    for k in 1:nsteps
        step!(integ)                 # avança 1 passo
        xnp1 = r6(integ.t)
        ynp1 = r6(integ.u)
        println(@sprintf("%4d | %11s | %11s", k, fmt(xnp1), fmt(ynp1)))
        push!(x_vals, xnp1)
        push!(y_vals, ynp1)
    end
    println("────────────────────────────────────────────────────────────\n")

    return x_vals, y_vals
end

# --- Calculando com 20 passos de 0 até pi/2 -----------------------
nsteps = 21
dt = π/2 / (nsteps - 1)  # dt é o passo entre os pontos
x_vals = LinRange(0, π/2, nsteps)  # 20 pontos de 0 até π/2

# Resolver usando o Euler Melhorado (Heun)
x_heun, y_heun = solve_with_print_heun(prob, dt; label="Euler Melhorado")
x_calc, y_calc = solve_with_print(prob, Euler(), dt; label="Euler explícito")

y_exact_vals = r6.(y_exact.(x_vals))  # Solução exata nos pontos de x_vals

# --- Erros (absoluto e relativo) ---------------------------------------------
abs_err_heun = r6.(abs.(y_heun .- y_exact_vals))
rel_err_heun = r6.(abs_err_heun ./ abs.(y_exact_vals))

abs_err_calc = r6.(abs.(y_calc .- y_exact_vals))
rel_err_calc = r6.(abs_err_calc ./ abs.(y_exact_vals))

# --- Resumo no terminal (t = 1) ----------------------------------------------
@info "Resumo em t=1" (
    y_heun_1 = fmt(y_heun[end]),
    y_exact_1 = fmt(y_exact_vals[end]),
    abs_err_heun_1 = fmt(abs_err_heun[end]),
    rel_err_heun_1 = fmt(rel_err_heun[end])
)

# --- Plots com GLMakie --------------------------------------------------------
GLMakie.activate!()
fig = Figure(size = (1100, 900))

# 1) Soluções
x_dense = LinRange(0, π/2, 100)  # Criação de uma linha densa para a solução exata
y_dense = r6.(y_exact.(x_dense))  # Solução exata densa

ax1 = Axis(fig[1, 1], title = "Soluções em [0, π/2]",
           xlabel = "t", ylabel = "y(t)")
lines!(ax1, x_dense, y_dense, label = "Exata", color = :red, linewidth=4)
scatterlines!(ax1, x_heun, y_heun, label = "Euler Melhorado", markersize=8)
scatterlines!(ax1, x_calc, y_calc, label = "Euler explícito", markersize=8, color= :black)
axislegend(ax1, position = :rb)

# 2) Erro absoluto
ax2 = Axis(fig[2, 1], title = "Erro absoluto",
           xlabel = "t", ylabel = "Diferença entre solução numérica e exata")
scatterlines!(ax2, x_heun, abs_err_heun, label = "Euler Melhorado", markersize=8)
scatterlines!(ax2, x_calc, abs_err_calc, label = "Euler Explícito", markersize=8)
axislegend(ax2, position = :rb)

# 3) Erro relativo
ax3 = Axis(fig[3, 1], title = "Erro relativo",
           xlabel = "t", ylabel = "Proporção entre erro e solução exata")
scatterlines!(ax3, x_heun, rel_err_heun, label = "Euler Melhorado", markersize=8)
scatterlines!(ax3, x_calc, rel_err_calc, label = "Euler Explícito", markersize=8)
axislegend(ax3, position = :rb)

display(fig)
save("ex2.png", fig)
println("\nFigura salva em: solucoes_e_erros_euler_melhorado.png\n")
