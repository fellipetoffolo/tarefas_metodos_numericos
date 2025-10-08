
module Rk3

#--- Função para RUnge-Kutta de terceira ordem

function runge_kutta_3(f, u0, tspan, h, A, b, c)
    # f: função que define a equação diferencial dy/dt = f(t, y)
    # y0: condição inicial
    # tspan: intervalo de tempo [t0, tf]
    # h: passo de tempo
    # A: matriz dos coeficientes a (matriz de estágio)
    # b: vetor dos coeficientes b (pesos dos estágios)
    # c: vetor dos coeficientes c (pontos de avaliação)

    t0, tf = tspan
    t = t0
    u = u0
    u_vect = [ u0 ]
    # Loop de integração
    while t < tf
        # Estágios de Runge-Kutta de 3ª ordem
        k1 = h * f(t, u)
        k2 = h * f(t + b[2]*h, u + A[2,1]*k1)
        k3 = h * f(t + b[3]*h, u + A[3,1]*k1 + A[3,2]*k2)

        # Atualização do valor de y
        u += c[1]*k1 + c[2]*k2 + c[3]*k3
        push!(u_vect, u)
        # Atualização do tempo
        t += h
    end

    return u, u_vect
end
end