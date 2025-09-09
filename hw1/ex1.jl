using DifferentialEquations
using GLMakie
using Printf

# --- Arredondamento simétrico a 6 casas decimais ------------------------------
r6(x::Real) = round(x, RoundNearestTiesAway; digits=6)
fmt(x) = @sprintf("%.6f", x)

# --- Problema ODE -------------------------------------------------------------
# y' = -y + t,  y(0) = 1, t ∈ [0,1]
rhs(u, p, t) = -u + t
u0 = 1.0
tspan = (0.0, 1.0)
prob = ODEProblem(rhs, u0, tspan)

# Solução exata para comparação: y(t) = t - 1 + 2e^{-t}
y_exact(x) = x - 1 + 2 * exp(-x)

# --- Função utilitária: integra com prints por passo --------------------------
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

# --- Executa métodos com os passos pedidos (com prints) -----------------------
# Euler explícito, h = 0.1
x_exp, y_exp = solve_with_print(prob, Euler(),         0.1;  label="Euler explícito")
# Euler implícito, h = 0.05
x_imp, y_imp = solve_with_print(prob, ImplicitEuler(), 0.05; label="Euler implícito")

# --- Curva exata para os gráficos (arredondada) -------------------------------
x_dense = r6.(collect(0.0:0.001:1.0))
y_den   = r6.(y_exact.(x_dense))

# Valores exatos nos pontos de cada método
yex_exp = r6.(y_exact.(x_exp))
yex_imp = r6.(y_exact.(x_imp))

# --- Erros (absoluto e relativo) ---------------------------------------------
abs_err_exp = r6.(abs.(y_exp .- yex_exp))
abs_err_imp = r6.(abs.(y_imp .- yex_imp))

rel_err_exp = r6.(abs_err_exp ./ abs.(yex_exp))
rel_err_imp = r6.(abs_err_imp ./ abs.(yex_imp))

# --- Resumo no terminal (t = 1) ----------------------------------------------
@info "Resumo em t=1" (
    y_exp_1 = fmt(y_exp[end]),
    y_imp_1 = fmt(y_imp[end]),
    y_exact_1 = fmt(yex_imp[end]),
    abs_err_exp_1 = fmt(abs_err_exp[end]),
    abs_err_imp_1 = fmt(abs_err_imp[end]),
    rel_err_exp_1 = fmt(rel_err_exp[end]),
    rel_err_imp_1 = fmt(rel_err_imp[end])
)

# --- Plots com GLMakie --------------------------------------------------------
GLMakie.activate!()
fig = Figure(size = (1100, 900))

# 1) Soluções
ax1 = Axis(fig[1, 1], title = "Soluções em [0, 1]",
           xlabel = "x", ylabel = "y(x)")
lines!(ax1, x_dense, y_den, label = "Exata")
scatterlines!(ax1, x_imp, y_imp, label = "Euler implícito (h=0.05)", markersize=8)
scatterlines!(ax1, x_exp, y_exp, label = "Euler explícito (h=0.1)", markersize=8)
axislegend(ax1, position = :rb)

# 2) Erro absoluto
ax2 = Axis(fig[2, 1], title = "Erro absoluto",
           xlabel = "x", ylabel = "Diferença entre solução numérica e exata")
scatterlines!(ax2, x_imp, abs_err_imp, label = "Implícito (h=0.05)", markersize=8)
scatterlines!(ax2, x_exp, abs_err_exp, label = "Explícito (h=0.1)", markersize=8)
axislegend(ax2, position = :rb)

# 3) Erro relativo
ax3 = Axis(fig[3, 1], title = "Erro relativo",
           xlabel = "x", ylabel = "Proporção entre erro e solução exata")
scatterlines!(ax3, x_imp, rel_err_imp, label = "Implícito (h=0.05)", markersize=8)
scatterlines!(ax3, x_exp, rel_err_exp, label = "Explícito (h=0.1)", markersize=8)
axislegend(ax3, position = :rb)

display(fig)
save("figs/ex1.png", fig)
println("\nFigura salva em: figs/ex1.png\n")
