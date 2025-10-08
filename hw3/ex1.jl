# ====================== RC(q) e RL(i): AB4–AM4 + 2 figuras por problema ======================
using Printf, GLMakie, Dates
using DifferentialEquations  # Heun(), ODEProblem, solve

# ---------- Núcleo numérico mínimo ----------
function rk4_step(f, t, y, h, p)
    k1 = f(t,       y,           p)
    k2 = f(t+h/2,   y + (h/2)*k1, p)
    k3 = f(t+h/2,   y + (h/2)*k2, p)
    k4 = f(t+h,     y + h*k3,     p)
    return y + (h/6)*(k1 + 2k2 + 2k3 + k4)
end

function solve_abm4(f; tspan=(0.0,1.0), y0=0.0, h=1e-3, p=nothing)
    t0, tf = tspan
    N  = Int(ceil((tf - t0)/h))
    ts = collect(range(t0, step=h, length=N+1))
    ys = similar(ts); ys[1] = y0

    # bootstrap (y1..y3) por RK4
    for n in 1:3
        ys[n+1] = rk4_step(f, ts[n], ys[n], h, p)
    end
    fvals = [f(ts[k], ys[k], p) for k in 1:4]

    @inbounds for n in 4:N
        fn, fnm1, fnm2, fnm3 = fvals[n], fvals[n-1], fvals[n-2], fvals[n-3]
        # preditor AB4
        ypred = ys[n] + (h/24)*(55*fn - 59*fnm1 + 37*fnm2 - 9*fnm3)
        # corretor AM4 (uma iteração, usando f no predito)
        fpred = f(ts[n+1], ypred, p)
        ys[n+1] = ys[n] + (h/24)*(9*fpred + 19*fn - 5*fnm1 + fnm2)
        push!(fvals, f(ts[n+1], ys[n+1], p))
    end
    return ts, ys
end

# salvar rápido
savefig!(fig, stem; dir="figs", ext="png") = (mkpath(dir); path=joinpath(dir, "$stem.$ext"); save(path, fig); @info "Figura salva" path; path)

# ---------- EDOs no formato solicitado ----------
RC(R, C, E) = (; R, C, E)      # estado q
RL(R, L, E) = (; R, L, E)      # estado i

f_rc_q(t, q, p) = (p.E - q/p.C) / p.R
f_rl_i(t, i, p) = (p.E - p.R*i) / p.L

# ====================== Execução ======================
# RC
p_rc = RC(1e3, 10e-6, 1.0)
t_rc, q = solve_abm4(f_rc_q; tspan=(0.0, 2.0), y0=0.0, h=0.05, p=p_rc)
vC_num = q ./ p_rc.C

# ---- RC com Euler explícito (1ª ordem) no mesmo grid ----
q0 = 0.0
f_rc_ode(u, p, t) = f_rc_q(t, u, p)  # (u,p,t)
prob_rc = ODEProblem(f_rc_ode, q0, (first(t_rc), last(t_rc)), p_rc)
h_rc = t_rc[2] - t_rc[1]  # = 0.05
sol_rc = solve(prob_rc, Euler(); dt=h_rc, adaptive=false, saveat=t_rc)
q_euler = sol_rc.u
vC_euler = q_euler ./ p_rc.C


# RL
p_rl = RL(10.0, 50e-3, 1.0)
t_rl, i_num = solve_abm4(f_rl_i; tspan=(0.0, 2.0), y0=0.0, h=0.05, p=p_rl)

# ---- RL com Euler explícito (1ª ordem) no mesmo grid ----
i0 = 0.0
f_rl_ode(u, p, t) = f_rl_i(t, u, p)
prob_rl = ODEProblem(f_rl_ode, i0, (first(t_rl), last(t_rl)), p_rl)
h_rl = t_rl[2] - t_rl[1]  # = 0.05
sol_rl = solve(prob_rl, Euler(); dt=h_rl, adaptive=false, saveat=t_rl)
i_euler = sol_rl.u



# ====================== Soluções analíticas (E constante, q(0)=0, i(0)=0) ======================
vC_exact(t) = p_rc.E .+ (0 .- p_rc.E) .* exp.(-t ./ (p_rc.R*p_rc.C))
i_exact(t)  = (p_rl.E/p_rl.R) .+ (0 .- p_rl.E/p_rl.R) .* exp.(-(p_rl.R ./ p_rl.L) .* t)

# ====================== Figuras (só solução) ======================
# RC: ABM4 × Heun × Analítico
fig_rc_cmp = Figure(resolution=(900,420))
ax1 = Axis(fig_rc_cmp[1,1], title="RC (q→v_C): ABM4 × Euler × Analítico",
           xlabel="t (s)", ylabel="v_C(t) [V]")
h_abm = lines!(ax1, t_rc, vC_num)          # AB4–AM4
h_eul = lines!(ax1, t_rc, vC_euler)        # Euler (1ª ordem)
h_exa = lines!(ax1, t_rc, vC_exact(t_rc))  # Analítico
Legend(fig_rc_cmp[1,2], [h_abm, h_eul, h_exa],
       ["AB4–AM4", "Euler (1ª ord.)", "Analítico"])
savefig!(fig_rc_cmp, "rc_num_vs_analitico")


# RL: ABM4 × Heun × Analítico
fig_rl_cmp = Figure(resolution=(900,420))
ax2 = Axis(fig_rl_cmp[1,1], title="RL (i): ABM4 × Euler × Analítico",
           xlabel="t (s)", ylabel="i(t) [A]")
g_abm = lines!(ax2, t_rl, i_num)           # AB4–AM4
g_eul = lines!(ax2, t_rl, i_euler)         # Euler (1ª ordem)
g_exa = lines!(ax2, t_rl, i_exact(t_rl))   # Analítico
Legend(fig_rl_cmp[1,2], [g_abm, g_eul, g_exa],
       ["AB4–AM4", "Euler (1ª ord.)", "Analítico"])
savefig!(fig_rl_cmp, "rl_num_vs_analitico")


# ====================== ERROS (somente AB4–AM4) ======================

# RC: erro (analítico − numérico)
err_rc = vC_exact(t_rc) .- vC_num
fig_rc_err = Figure(resolution=(900,380))
ax_rc_err = Axis(fig_rc_err[1,1], title="RC: erro (analítico − numérico)",
                 xlabel="t (s)", ylabel="erro [V]")
lines!(ax_rc_err, t_rc, err_rc, label="AB4-AM4", color=:orange)
hlines!(ax_rc_err, [0])
axislegend(ax_rc_err, position=:rt)
savefig!(fig_rc_err, "rc_erro_analitico_menos_numerico")

# RL: erro (analítico − numérico)
err_rl = i_exact(t_rl) .- i_num
fig_rl_err = Figure(resolution=(900,380))
ax_rl_err = Axis(fig_rl_err[1,1], title="RL: erro (analítico − numérico)",
                 xlabel="t (s)", ylabel="erro [A]")
lines!(ax_rl_err, t_rl, err_rl, label="AB4-AM4", color=:orange)
hlines!(ax_rl_err, [0])
axislegend(ax_rl_err, position=:rt)
savefig!(fig_rl_err, "rl_erro_analitico_menos_numerico")

# ====================== ERRO DO EULER × ANALÍTICO ======================

# RC: erro (analítico − Euler)
err_rc_euler = vC_exact(t_rc) .- vC_euler
fig_rc_err_eul = Figure(resolution=(900,380))
ax_rc_err_eul = Axis(fig_rc_err_eul[1,1],
    title="RC: erro (analítico − Euler)",
    xlabel="t (s)", ylabel="erro [V]")
lines!(ax_rc_err_eul, t_rc, err_rc_euler, color=:orange, label="Euler (1ª ord.)")
hlines!(ax_rc_err_eul, [0])
axislegend(ax_rc_err_eul, position=:rt)
savefig!(fig_rc_err_eul, "rc_erro_analitico_menos_euler")

# RL: erro (analítico − Euler)
err_rl_euler = i_exact(t_rl) .- i_euler
fig_rl_err_eul = Figure(resolution=(900,380))
ax_rl_err_eul = Axis(fig_rl_err_eul[1,1],
    title="RL: erro (analítico − Euler)",
    xlabel="t (s)", ylabel="erro [A]")
lines!(ax_rl_err_eul, t_rl, err_rl_euler, color=:orange, label="Euler (1ª ord.)")
hlines!(ax_rl_err_eul, [0], color=:blue)
axislegend(ax_rl_err_eul, position=:rt)
savefig!(fig_rl_err_eul, "rl_erro_analitico_menos_euler")
