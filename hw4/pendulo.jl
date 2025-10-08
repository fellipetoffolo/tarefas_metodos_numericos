using DifferentialEquations
using GLMakie        # ou CairoMakie se preferir render offline
using LinearAlgebra  # <--- necessário para norm

# --- parâmetros e modelo ---
p = (c = 0.16, m = 0.5, L = 1.2, g = 9.81)

function pendulum!(du, u, p, t)
    du[1] = u[2]
    du[2] = (-p.c / p.m)*u[2] - (p.g / p.L)*sin(u[1])
end

u0 = [pi/2, 0.0]
tspan = (0.0, 20.0)

# solução de referência com solver adaptativo (alta precisão)
problem = ODEProblem(pendulum!, u0, tspan, p)
sol_ref = solve(problem, Tsit5(), reltol = 1e-8, abstol = 1e-8)

# wrapper que retorna um vetor (não in-place) para usar nos métodos explícitos
f_vec(t, y, p) = begin
    du = similar(y)
    pendulum!(du, y, p, t)
    du
end

# parâmetros da comparação
h = 0.01
t0, tf = tspan
N = Int(floor((tf - t0)/h))
ts = collect(range(t0, step=h, length=N+1))   # instantes onde vamos comparar

# ---------------- Métodos numéricos ----------------

# Heun (Euler melhorado) para sistema
function solve_heun(f, t0, y0, h, N, p)
    m = length(y0)
    ys = zeros(Float64, m, N+1)
    ys[:,1] .= y0
    t = t0
    for n in 1:N
        yview = view(ys, :, n)              # cria view da coluna atual
        k1 = f(t, yview, p)
        y_predict = yview .+ h .* k1
        k2 = f(t + h, y_predict, p)
        ys[:, n+1] = yview .+ (h/2).*(k1 .+ k2)
        t += h
    end
    return ts, ys
end

# RK4 passo fixo (vetorial)
function rk4_step_vec(f, t, y, h, p)
    k1 = f(t, y, p)
    k2 = f(t + h/2, y .+ (h/2).*k1, p)
    k3 = f(t + h/2, y .+ (h/2).*k2, p)
    k4 = f(t + h,   y .+ h.*k3, p)
    y .+ (h/6).*(k1 .+ 2k2 .+ 2k3 .+ k4)
end

function solve_rk4(f, t0, y0, h, N, p)
    m = length(y0)
    ys = zeros(Float64, m, N+1)
    ys[:,1] .= y0
    t = t0
    for n in 1:N
        yview = view(ys, :, n)
        ys[:, n+1] = rk4_step_vec(f, t, yview, h, p)
        t += h
    end
    return ts, ys
end

# ABM4 (Adams-Bashforth-Moulton 4) para vetores
function solve_abm4_vec(f, t0, y0, h, N, p)
    m = length(y0)
    ys = zeros(Float64, m, N+1)
    ys[:,1] .= y0

    # bootstrap com RK4 para os primeiros 3 passos
    t_local = t0
    for n in 1:3
        yview = view(ys, :, n)
        ys[:, n+1] = rk4_step_vec(f, t_local, yview, h, p)
        t_local += h
    end

    # preparar fvals como vetor de arrays
    fvals = Vector{Vector{Float64}}(undef, N+1)
    for k in 1:4
        yk = view(ys, :, k)
        fvals[k] = f(t0 + (k-1)*h, yk, p)
    end

    for n in 4:N
        fn   = fvals[n]
        fnm1 = fvals[n-1]
        fnm2 = fvals[n-2]
        fnm3 = fvals[n-3]

        # preditor AB4 (vetorial)
        ypred = ys[:, n] .+ (h/24).*(55 .* fn .- 59 .* fnm1 .+ 37 .* fnm2 .- 9 .* fnm3)
        # corretor AM4 (uma iteração)
        fpred = f(t0 + n*h, ypred, p)
        ys[:, n+1] = ys[:, n] .+ (h/24).*(9 .* fpred .+ 19 .* fn .- 5 .* fnm1 .+ fnm2)
        # armazenar para próximos passos
        fvals[n+1] = f(t0 + n*h, view(ys, :, n+1), p)
    end

    return ts, ys
end

# --- calcular soluções numéricas ---
ts_heun, ys_heun = solve_heun(f_vec, t0, u0, h, N, p)
ts_rk4, ys_rk4   = solve_rk4(f_vec, t0, u0, h, N, p)
ts_abm, ys_abm   = solve_abm4_vec(f_vec, t0, u0, h, N, p)

# --- obter solução de referência nos mesmos instantes (interpolada) ---
ys_ref = zeros(Float64, 2, length(ts))
for (i, ti) in enumerate(ts)
    uref = sol_ref(ti)           # sol_ref é chamável e devolve vetor
    ys_ref[:, i] .= uref
end

# --- calcular erros (norma euclidiana do vetor de erro em cada instante) ---
err_heun = [norm(ys_heun[:, i] .- ys_ref[:, i]) for i in 1:length(ts)]
err_rk4  = [norm(ys_rk4[:,  i] .- ys_ref[:, i]) for i in 1:length(ts)]
err_abm  = [norm(ys_abm[:,  i] .- ys_ref[:, i]) for i in 1:length(ts)]


# --- Figura com as soluções comparadas (θ e ω) em uma única figura ---
fig = Figure(resolution = (1000, 500))

# theta(t) comparativo (esquerda)
ax1 = Axis(fig[1, 1], title = "θ(t) — comparação", xlabel = "t", ylabel = "θ (rad)")
lines!(ax1, ts, ys_ref[1, :], label = "ref (Tsit5)", linewidth = 2)
lines!(ax1, ts_heun, ys_heun[1, :], linestyle = :dash, label = "Heun (h=0.05)")
lines!(ax1, ts_rk4, ys_rk4[1, :], linestyle = :dot, label = "RK4 (h=0.05)")
lines!(ax1, ts_abm, ys_abm[1, :], linestyle = :dashdot, label = "ABM4 (h=0.05)")
axislegend(ax1; position = :rt)

# omega(t) comparativo (direita)
ax2 = Axis(fig[1, 2], title = "ω(t) — comparação", xlabel = "t", ylabel = "ω (rad/s)")
lines!(ax2, ts, ys_ref[2, :], label = "ref (Tsit5)", linewidth = 2)
lines!(ax2, ts_heun, ys_heun[2, :], linestyle = :dash, label = "Heun")
lines!(ax2, ts_rk4, ys_rk4[2, :], linestyle = :dot, label = "RK4")
lines!(ax2, ts_abm, ys_abm[2, :], linestyle = :dashdot, label = "ABM4")
axislegend(ax2; position = :rt)

display(fig)
save("pendulum_solutions_comparison.png", fig)

# --- Salvar cada figura de erro separadamente ---
# Erro Heun
fig_heun = Figure(resolution = (800, 400))
ax_heun = Axis(fig_heun[1, 1], title = "Erro (norma) — Heun", xlabel = "t", ylabel = "||erro||")
lines!(ax_heun, ts, err_heun)
display(fig_heun)
save("pendulum_error_heun.png", fig_heun)

# Erro RK4
fig_rk4 = Figure(resolution = (800, 400))
ax_rk4 = Axis(fig_rk4[1, 1], title = "Erro (norma) — RK4", xlabel = "t", ylabel = "||erro||")
lines!(ax_rk4, ts, err_rk4)
display(fig_rk4)
save("pendulum_error_rk4.png", fig_rk4)

# Erro ABM4
fig_abm = Figure(resolution = (800, 400))
ax_abm = Axis(fig_abm[1, 1], title = "Erro (norma) — ABM4", xlabel = "t", ylabel = "||erro||")
lines!(ax_abm, ts, err_abm)
display(fig_abm)
save("pendulum_error_abm4.png", fig_abm)

using GLMakie

# vetores já existentes no seu script:
# ts, ys_ref (2 × N) onde ys_ref[1,:]=theta, ys_ref[2,:]=omega

t_vec     = ts
theta_vec = ys_ref[1, :]
omega_vec = ys_ref[2, :]

fig3 = Figure(resolution = (1000, 700))
ax3 = Axis3(fig3[1, 1],
            title = "Trajetória 3D: (t, θ(t), ω(t))",
            xlabel = "t (seg)", ylabel = "θ (rad)", zlabel = "ω (rad/s)")

# Curva paramétrica 3D com linha contínua
lines!(ax3, t_vec, theta_vec, omega_vec, linewidth = 2)

# Pontos coloridos pelo tempo para indicar direção
scatter!(ax3, t_vec, theta_vec, omega_vec; markersize = 6, color = t_vec, colormap = :viridis)

# Opcional: ajustar posição inicial da câmera — comente se der problema no seu backend
# try
#     cam3d!(ax3, Vec3f0(30, -30, 20))   # pode variar entre backends/versões; descomente se funcionar
# catch e
#     @warn "Não foi possível ajustar a câmera com cam3d! — ignorando."
# end

display(fig3)
save("pendulum_3d_t_theta_omega.png", fig3)





using GLMakie

# calcula erro absoluto apenas em theta

err_theta_rk4  = abs.(ys_rk4[1, :]  .- ys_ref[1, :])
err_theta_abm  = abs.(ys_abm[1, :]  .- ys_ref[1, :])

# controla quantos markers mostrar (evita excesso de pontos)
npts = length(ts)
marker_skip = max(1, Int(round(npts / 250)))  # ~250 markers por curva

fig = Figure(resolution = (1000, 420))
ax = Axis(fig[1, 1],
    title = "Erro absoluto em θ (|Δθ|) — comparação",
    xlabel = "t (s)", ylabel = "|Δθ| (rad)")

# linhas

lines!(ax, ts, err_theta_rk4,   linestyle = :dot,     linewidth = 2, label = "RK4")
lines!(ax, ts, err_theta_abm,   linestyle = :dashdot, linewidth = 2, label = "ABM4")

# markers amostrados para visual guia
scatter!(ax, ts[1:marker_skip:end], err_theta_rk4[1:marker_skip:end],  markersize = 5)
scatter!(ax, ts[1:marker_skip:end], err_theta_abm[1:marker_skip:end],  markersize = 5)

axislegend(ax; position = :rt)

display(fig)
save("pendulum_error_theta_only.png", fig)


























using Printf



# salva amostras a cada 1 segundo, começando em 0
function save_samples_every_second(filename::AbstractString;
        sol_ref,
        ts, ys_heun, ys_rk4, ys_abm,
        t0=0.0, tf=20.0, h=0.05)

    # helper: interpola linearmente coluna-wise entre pontos de ts
    function interp_sol(ts, ys, tq)
        # ys: m x N array, ts: length N
        if tq <= ts[1]
            return ys[:,1]
        elseif tq >= ts[end]
            return ys[:,end]
        else
            # find k such that ts[k] <= tq < ts[k+1]
            k = findlast(t -> t <= tq, ts)
            if k == length(ts)
                return ys[:,end]
            end
            t1, t2 = ts[k], ts[k+1]
            α = (tq - t1) / (t2 - t1)
            return (1-α).*ys[:,k] .+ α .* ys[:,k+1]
        end
    end

    tmax = floor(Int, tf)
    open(filename, "w") do io
        # cabeçalho
        println(io, "t,theta_ref,omega_ref,theta_heun,omega_heun,theta_rk4,omega_rk4,theta_abm,omega_abm")
        for tt in 0:tmax
            tq = float(tt)
            # referência via sol_ref (assume chamável)
            uref = sol_ref(tq)            # vetor [theta; omega]
            # interpola numéricos (assumindo mesmas dimensões 2 x N)
            u_heun = interp_sol(ts, ys_heun, tq)
            u_rk4  = interp_sol(ts, ys_rk4,  tq)
            u_abm  = interp_sol(ts, ys_abm,  tq)

            # escrever linha CSV
            @printf(io, "%.6f,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n",
                    tq,
                    uref[1], uref[2],
                    u_heun[1], u_heun[2],
                    u_rk4[1], u_rk4[2],
                    u_abm[1], u_abm[2])
        end
    end
    println("Amostras salvas em: $filename (t = 0:1:$tmax)")
end

# Exemplo de uso (ajuste nomes se necessário):
save_samples_every_second("pendulum_samples.csv";
    sol_ref = sol_ref,
    ts = ts, ys_heun = ys_heun, ys_rk4 = ys_rk4, ys_abm = ys_abm,
    t0 = ts[1], tf = ts[end], h = ts[2]-ts[1])




















using DifferentialEquations
using GLMakie
using LinearAlgebra

# ---------------- parâmetros físicos (seu p)
p = (c = 0.16, m = 0.5, L = 1.2, g = 9.81)
a = p.c / p.m
b = p.g / p.L

# modelo original (não-linear) usado apenas para referência
function pendulum!(du, u, p, t)
    du[1] = u[2]
    du[2] = (-p.c / p.m)*u[2] - (p.g / p.L)*sin(u[1])
end

# condições iniciais e resolução de referência (solução "verdadeiro" de alta precisão)
u0 = [pi/2, 0.0]
tspan = (0.0, 20.0)
prob = ODEProblem(pendulum!, u0, tspan, p)
sol_ref = solve(prob, Tsit5(), reltol = 1e-8, abstol = 1e-8)

# parâmetros numéricos (faça variar h e max_iters)
h = 0.05
t0, tf = tspan
N = Int(floor((tf - t0)/h))    # número de passos (t0..tN)
ts = collect(t0:h:(t0 + N*h))  # N+1 instantes

# ---------------- Construção por diferenças finitas (linearizado) ----------------
# Equação linearizada (small-angle): theta'' + a theta' + b theta = 0

# discretização escolhida:
# (theta_{n+1} - 2 theta_n + theta_{n-1}) / h^2  + a * (theta_n - theta_{n-1})/h + b * theta_n = 0
# rearranjando para obter relação envolvendo theta_{n+1}, theta_n, theta_{n-1}:
# Coeficientes: A_{n,n+1} * theta_{n+1} + A_{n,n} * theta_n + A_{n,n-1} * theta_{n-1} = 0
#
# Escrevemos sistema para incógnitas theta_1 .. theta_{N-1}
#
# Para n = 1..N-1 (equação centrada)
# (1/h^2) * theta_{n+1} + (-2/h^2 + b) * theta_n + (1/h^2 - a/h) * theta_{n-1} = 0
#
# Observação: o termo da derivada primeira usamos backward (theta_n - theta_{n-1})/h -> coef ( - a/h ) no termo de theta_{n-1}
#
# Vamos montar A (tridiagonal) tal que A * Theta_inner = d, onde Theta_inner = [theta_1 ... theta_{N-1}]'

# Preparar A e d
M = N - 1  # número de incógnitas (ignorando theta_0 e theta_N)
if M < 1
    error("N muito pequeno; aumente intervalo / diminua h")
end

# Prealocação
A = zeros(Float64, M, M)
d = zeros(Float64, M)

# Coeficientes (constantes)
c_plus  = 1.0 / h^2                         # coef de theta_{n+1}
c_zero  = -2.0 / h^2 + b                    # coef de theta_n
c_minus = 1.0 / h^2 - a / h                 # coef de theta_{n-1}

# montar as equações para n=1..M correspondendo a originais n=1..N-1
# lembrando que theta_0 é conhecido = u0[1]
theta0 = u0[1]
omega0 = u0[2]

# Aproximação inicial para theta_1 baseado em Taylor: theta_1 ≈ theta0 + h * omega0
# (usado para construir d[1] se preferirmos tratá-lo como conhecido)
theta1_guess = theta0 + h * omega0

for i in 1:M
    n = i  # corresponde ao índice original n = i
    # equação: c_plus * theta_{n+1} + c_zero * theta_n + c_minus * theta_{n-1} = 0
    # indices correspondem:
    # unknown vector Theta_inner has entries theta_1..theta_{N-1} -> index i corresponds to theta_n
    # theta_{n+1} is unknown for n <= N-2 (i <= M-1), else if n = N-1, theta_N is not in unknowns
    # theta_{n-1} is unknown for n >= 2 (i >= 2), else if n=1 theta_0 is known
    # we move known terms to RHS d

    # diagonal (theta_n)
    A[i, i] = c_zero

    # subdiagonal (theta_{n-1}) -> column i-1
    if i >= 2
        A[i, i-1] = c_minus
    else
        # theta_0 known -> move c_minus * theta_0 to RHS
        d[i] -= c_minus * theta0
    end

    # superdiagonal (theta_{n+1}) -> column i+1
    if i <= M-1
        A[i, i+1] = c_plus
    else
        # theta_N unknown (we don't have boundary condition at final time)
        # we will approximate theta_N using a single explicit step (RK4 one-step) from reference initial cond,
        # OR use the reference solution value at final time to close the system.
        # Aqui prefiro usar aproximação de uma etapa explícita simples:
        # usar a estimativa theta_N ≈ theta_{N-1} + h * omega_{N-1} (backward Euler-like)
        # mas omega_{N-1} unknown -> para simplicidade, usar a solução de referência para theta_N (fecho/prático).
        thetaN_approx = sol_ref(t0 + N*h)[1]  # usar referência apenas para fechar o sistema
        d[i] -= c_plus * thetaN_approx
    end
end

# Agora A * Theta_inner = d
# Resolver iterativamente com Gauss-Seidel (parâmetro: max_iters e tol)

using LinearAlgebra

"""
    gauss_seidel_relaxed(A, b; x0=zeros(length(b)), max_iters=10000, tol=1e-8, omega=1.0, record_res=true)

Gauss-Seidel com relaxação (atualiza x_i <- (1-ω)*x_i_old + ω*(b_i - sum_j≠i A[i,j]*x_j)/A[i,i]).
Retorna (x, niters, converged, residues).

- omega = 1.0 -> padrão GS
- omega in (0,1) -> under-relaxation (mais estável)
- omega > 1 -> over-relaxation (SOR), uso com cuidado
"""
function gauss_seidel_relaxed(A, b; x0 = zeros(length(b)),
                              max_iters::Int=5000,
                              tol::Float64=1e-8,
                              omega::Float64 = 1.0,
                              record_res::Bool = true)

    n = length(b)
    x = copy(x0)
    residues = record_res ? Float64[] : nothing

    # pre-check diagonal não-nula
    @assert all(abs.(diag(A)) .> 0) "Zeros na diagonal de A! Não é possível dividir."

    for k in 1:max_iters
        x_old = copy(x)
        # Gauss-Seidel sweep
        for i in 1:n
            sigma = 0.0
            # j < i usa x (já atualizado), j > i usa x_old
            @inbounds for j in 1:i-1
                sigma += A[i, j] * x[j]
            end
            @inbounds for j in i+1:n
                sigma += A[i, j] * x_old[j]
            end
            xi_gs = (b[i] - sigma) / A[i, i]
            # relaxação
            x[i] = (1 - omega) * x_old[i] + omega * xi_gs
        end

        # registrar residuo
        if record_res
            r = b - A*x
            push!(residues, norm(r))   # norma do resíduo
        end

        # critério de parada baseado em variação relativa do vetor solução
        if norm(x - x_old) < tol * max(1.0, norm(x))
            return x, k, true, residues
        end
    end

    return x, max_iters, false, residues
end


# Parâmetros do GS — deixe explícitos
omega = 0.5
max_iters = 20000
tol = 1e-9

# inicial guess (pode usar zeros ou uma extrapolação)
x0 = fill(theta0, M)
theta_inner, niters, converged, residues = gauss_seidel_relaxed(A, d;
    x0 = x0, max_iters = max_iters, tol = tol, omega = omega, record_res = true)


println("Gauss-Seidel: converged = ", converged, ", niters = ", niters)

# reconstruir vetor theta completo (0..N)
theta_fd = zeros(Float64, N+1)
theta_fd[1] = theta0
for i in 1:M
    theta_fd[i+1] = theta_inner[i]   # theta_1..theta_{N-1}
end
# theta_N aproximada (usada para fechamento)
theta_fd[N+1] = sol_ref(t0 + N*h)[1]   # apenas o valor de fechamento que usamos antes

# retornar também velocidade aproximada pelo backward difference (1ª ordem)
omega_fd = zeros(Float64, N+1)
omega_fd[1] = omega0
for n in 2:N+1
    omega_fd[n] = (theta_fd[n] - theta_fd[n-1]) / h
end

# --- erro em relação à solução de referência ---
ys_ref = zeros(2, N+1)
for (i, ti) in enumerate(ts)
    uref = sol_ref(ti)
    ys_ref[:, i] .= uref
end

err_fd = [norm([theta_fd[i] - ys_ref[1,i], omega_fd[i] - ys_ref[2,i]]) for i in 1:length(ts)]

# --- plot do resultado (comparação theta e erro) ---
fig = Figure(resolution = (1000, 700))

ax1 = Axis(fig[1, 1], title = "θ(t) — FD linearizado (central 2º, backward 1º) vs ref", xlabel = "t", ylabel = "θ (rad)")
lines!(ax1, ts, ys_ref[1, :], label = "ref (Tsit5)", linewidth = 2)
lines!(ax1, ts, theta_fd, linestyle = :dash, label = "FD linearizado (GS)")
axislegend(ax1)

ax2 = Axis(fig[2, 1], title = "Erro (norma) do esquema FD (GS)", xlabel = "t", ylabel = "||erro||")
lines!(ax2, ts, err_fd)
fig[2,1] = ax2

display(fig)
save("pendulum_fd_gs_comparison.png", fig)

println("max error (FD vs ref) = ", maximum(err_fd))
