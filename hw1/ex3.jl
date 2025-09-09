using DifferentialEquations
using GLMakie
using Printf

# --- Arredondamento simétrico a 6 casas decimais ------------------------------
r6(x::Real) = round(x, RoundNearestTiesAway; digits=6)
fmt(x) = @sprintf("%.6f", x)

# --- Problema ODE: v' = g - (c/m) v ------------------------------------------
p = (g = 9.8, c = 12.5, m = 68.0)
parachute(u, p, t) = p.g - (p.c/p.m) * u

u0    = 0.0
tspan = (0.0, 20.0)
prob  = ODEProblem(parachute, u0, tspan, p)

# Solução exata do paraquedista
y_exact(t) = (p.g*p.m/p.c) * (1 - exp(-(p.c/p.m) * t))

# --- Heun (Euler Melhorado) com impressão ------------------------------------
function solve_with_print_heun(prob::ODEProblem, dt; label::AbstractString)
    t0, tf = prob.tspan
    u0 = prob.u0
    nsteps = Int(round((tf - t0) / dt))

    x_vals = Float64[r6(t0)]
    y_vals = Float64[r6(u0)]

    println("────────────────────────────────────────────────────────────")
    println(@sprintf("%s (h=%s):", label, fmt(dt)))
    println("passo |   t_(n+1)   |   y_(n+1)")
    println("────────────────────────────────────────────────────────────")

    t = t0
    y = u0
    f = prob.f
    prm = prob.p

    for k in 1:nsteps
        f1 = f(y, prm, t)              # f(t_n, y_n)
        y_pred = y + dt * f1           # previsão (Euler explícito)
        f2 = f(y_pred, prm, t + dt)    # f(t_{n+1}, y_pred)
        y_next = y + (dt/2) * (f1 + f2)

        t += dt
        y  = y_next

        push!(x_vals, r6(t))
        push!(y_vals, r6(y))

        println(@sprintf("%4d | %11s | %11s", k, fmt(t), fmt(y)))
    end
    println("────────────────────────────────────────────────────────────\n")

    return x_vals, y_vals
end

# --- Wrapper p/ solvers de DifferentialEquations.jl ---------------------------
function solve_with_print(prob, alg, dt; label::AbstractString)
    t0, tf = prob.tspan
    integ = init(prob, alg; dt=dt, adaptive=false)
    nsteps = Int(round((tf - t0) / dt))

    x_vals = Float64[r6(integ.t)]
    y_vals = Float64[r6(integ.u)]

    println("────────────────────────────────────────────────────────────")
    println(@sprintf("%s (h=%s):", label, fmt(dt)))
    println("passo |   t_(n+1)   |   y_(n+1)")
    println("────────────────────────────────────────────────────────────")

    for k in 1:nsteps
        step!(integ)
        xnp1 = r6(integ.t)
        ynp1 = r6(integ.u)
        println(@sprintf("%4d | %11s | %11s", k, fmt(xnp1), fmt(ynp1)))
        push!(x_vals, xnp1)
        push!(y_vals, ynp1)
    end
    println("────────────────────────────────────────────────────────────\n")

    return x_vals, y_vals
end

# --- Parâmetros de integração --------------------------------------------------
nsteps = 61
dt = (tspan[2] - tspan[1]) / (nsteps - 1)  # garante que chegue em tf

# Resolver
x_heun, y_heun = solve_with_print_heun(prob, dt; label="Euler Melhorado (Heun)")
x_calc, y_calc = solve_with_print(prob, Euler(), dt; label="Euler Explícito")

# Solução exata nos pontos de cada método
y_exact_heun = r6.(y_exact.(x_heun))
y_exact_calc = r6.(y_exact.(x_calc))

# --- Erros (absoluto e relativo) ---------------------------------------------
abs_err_heun = r6.(abs.(y_heun .- y_exact_heun))
abs_err_calc = r6.(abs.(y_calc .- y_exact_calc))

# trata divisão por zero no t=0
rel_err_heun = map((ae, ye) -> iszero(ye) ? NaN : r6(ae/abs(ye)), abs_err_heun, y_exact_heun)
rel_err_calc = map((ae, ye) -> iszero(ye) ? NaN : r6(ae/abs(ye)), abs_err_calc, y_exact_calc)

# --- Resumo "em t=1" (realmente em t mais próximo de 1.0) --------------------
idx_heun_t1 = argmin(abs.(x_heun .- 1.0))
@info "Resumo em t≈1" (
    t_heun   = fmt(x_heun[idx_heun_t1]),
    y_heun   = fmt(y_heun[idx_heun_t1]),
    y_exact  = fmt(y_exact_heun[idx_heun_t1]),
    abs_err  = fmt(abs_err_heun[idx_heun_t1]),
    rel_err  = isnan(rel_err_heun[idx_heun_t1]) ? "NaN" : fmt(rel_err_heun[idx_heun_t1])
)

# --- Plots com GLMakie --------------------------------------------------------
GLMakie.activate!()
fig = Figure(size = (1100, 900))

# 1) Soluções
x_dense = LinRange(tspan[1], tspan[2], 400)
y_dense = r6.(y_exact.(x_dense))

ax1 = Axis(fig[1, 1], title = "Soluções em [0, 10]",
           xlabel = "t", ylabel = "v(t)")
lines!(ax1, x_dense, y_dense, label = "Exata", color = :red, linewidth = 4)
scatterlines!(ax1, x_heun, y_heun, label = "Euler Melhorado", markersize = 8, color = :black)
scatterlines!(ax1, x_calc, y_calc, label = "Euler Explícito", markersize = 8, color = :yellow)
axislegend(ax1, position = :rb)

# 2) Erro absoluto
ax2 = Axis(fig[2, 1], title = "Erro absoluto", xlabel = "t",
           ylabel = "Diferença entre solução numérica e exata")
scatterlines!(ax2, x_heun, abs_err_heun, label = "Euler Melhorado", markersize = 8, color = :black)
scatterlines!(ax2, x_calc, abs_err_calc, label = "Euler Explícito", markersize = 8, color = :yellow)
axislegend(ax2, position = :rb)

# 3) Erro relativo
ax3 = Axis(fig[3, 1], title = "Erro relativo", xlabel = "t",
           ylabel = "Proporção entre erro e solução exata")
scatterlines!(ax3, x_heun, rel_err_heun, label = "Euler Melhorado", markersize = 8, color = :black)
scatterlines!(ax3, x_calc, rel_err_calc, label = "Euler Explícito", markersize = 8, color = :yellow)
axislegend(ax3, position = :rb)

display(fig)
save("problema_paraquedista.png", fig)
println("\nFigura salva em: problema_paraquedista.png\n")
