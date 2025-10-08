using Printf
using GLMakie

# ----- Utilidades de formatação -----
r6(x::Real) = round(x, RoundNearestTiesAway; digits=6)
fmt(x) = @sprintf("%.6f", x)

# ----- Modelo: v'(t) = g - (c/m) v,  v(0)=0 -----
const P = (g = 9.8, c = 12.5, m = 68.0, a = 8.3, v_max = 46.0, b = 2.2)  # Corrigido aqui
parachute(u, p, t) = p.g - (p.c/p.m) * (u + p.a*(u/p.v_max)^(p.b))

u0 = 0.0
const T_MIN, T_MAX = 0.0, 200.0

# ----- Heun até um t alvo -----
"""
    heun_until_t(f, p, u0, t0, ttarget; desired_dt=0.05, print_steps=false)

Integra v'(t)=f(v,p,t) de t0 até `ttarget` com Heun.
`desired_dt` é um passo alvo; o algoritmo ajusta o `dt` para bater exatamente em `ttarget`.
Retorna `(t_final, u_final)`.
"""
function heun_until_t(f, p, u0, t0, ttarget; desired_dt=0.05, print_steps=false)
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

    # Armazenar os valores para plotar depois
    t_values = [t]
    v_values = [y]

    for k in 1:nsteps
        f1 = f(y, p, t)
        y_pred = y + dt*f1
        f2 = f(y_pred, p, t + dt)
        y += (dt/2) * (f1 + f2)
        t += dt
        push!(t_values, t)
        push!(v_values, y)
        print_steps && println(@sprintf("%4d | %11s | %11s", k, fmt(t), fmt(y)))
    end
    return t_values, v_values
end

function main()
    println("Queda com arrasto linear: v'(t) = g - (c/m) v,  v(0)=0")
    println("Parâmetros: g=$(P.g), c=$(P.c), m=$(P.m), a=$(P.a), b=$(P.b)")
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

    # Executando o método de Heun
    t_values, v_values = heun_until_t(parachute, P, u0, T_MIN, t_target; desired_dt=target_dt, print_steps=true)

    # Plotando os resultados
    fig = Figure(resolution = (800, 600))
    ax = Axis(fig[1, 1], title = "Velocidade vs Tempo", xlabel = "Tempo (s)", ylabel = "Velocidade (m/s)")
    plot!(ax, t_values, v_values, label = "Solução Numérica (Heun)", color = :blue)
    
    # Salvando a figura
    save("figs/ex42.png", fig)
    println("\nFigura salva em: figs/ex42.png\n")
    # Mostrando o gráfico
    display(fig)
end

main()
