using Printf

# ----- Utilidades de formatação -----
r6(x::Real) = round(x, RoundNearestTiesAway; digits=6)
fmt(x) = @sprintf("%.6f", x)

# ----- Modelo: v'(t) = g - (c/m) v,  v(0)=0 -----
const P = (g = 9.8, c = 12.5, m = 68.0)
parachute(u, p, t) = p.g - (p.c/p.m) * u

u0 = 0.0
const T_MIN, T_MAX = 0.0, 100.0

y_exact(t) = (P.g * P.m / P.c) * (1 - exp(-(P.c / P.m) * t))

# ----- Heun até um t alvo -----
"""
    heun_until_t(f, p, u0, t0, ttarget; desired_dt=0.05, print_steps=false)

Integra v'(t)=f(v,p,t) de t0 até `ttarget` com Heun.
`desired_dt` é um passo alvo; o algoritmo ajusta o `dt` para bater exatamente em `ttarget`.
Retorna `(t_final, u_final)`.
"""
function heun_until_t(f, p, u0, t0, ttarget; desired_dt=1, print_steps=false)
    ttarget < t0 && error("ttarget deve ser >= t0")
    ttarget == t0 && return (t0, u0)

    nsteps = max(1, ceil(Int, (ttarget - t0)/desired_dt))
    dt = (ttarget - t0)/nsteps

    t, y = t0, u0
    if print_steps
        println("Heun: dt=$(fmt(dt)) em $nsteps passos")
        println("passo |   t_(n+1)   |   v_(n+1)")
        println("──────────────────────────────────────")
    end

    for k in 1:nsteps
        f1 = f(y, p, t)
        y_pred = y + dt*f1
        f2 = f(y_pred, p, t + dt)
        y += (dt/2) * (f1 + f2)
        t += dt
        print_steps && println(@sprintf("%4d | %11s | %11s", k, fmt(t), fmt(y)))
    end
    return (t, y)
end

function main()
    println("Queda com arrasto linear: v'(t) = g - (c/m) v,  v(0)=0")
    println("Parâmetros: g=$(P.g), c=$(P.c), m=$(P.m)")
    println("Intervalo permitido: [$(T_MIN), $(T_MAX)] s")
    print("Informe um t dentro desse intervalo para aproximar v(t): ")
    t_str = readline()
    print("Informe a largura do passo para simulação: ")
    desired_dt = readline()
    target_dt = tryparse(Float64, strip(desired_dt))
    t_target = tryparse(Float64, strip(t_str))
    t_target === nothing && return println("Valor inválido.")

    if !(T_MIN <= t_target <= T_MAX)
        println("t fora do intervalo [$(T_MIN), $(T_MAX)].")
        return
    end

    t, v_aprox = heun_until_t(parachute, P, u0, T_MIN, t_target; desired_dt=target_dt, print_steps=false)

    v_exata = y_exact(t_target)
    abs_err = abs(v_aprox - v_exata)
    rel_err = iszero(v_exata) ? NaN : abs_err/abs(v_exata)

    println("\nResultado em t=$(fmt(t)) s")
    println("v_aprox ≈ $(fmt(v_aprox)) m/s  (Heun)")
    println("v_exata = $(fmt(v_exata)) m/s")
    println("erro abs = $(fmt(abs_err))")
    println("erro rel = $(isnan(rel_err) ? "NaN" : fmt(rel_err))")
end

main()
