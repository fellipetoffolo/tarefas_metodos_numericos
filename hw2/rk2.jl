module Rk2

#--- Função para RUnge-Kutta de segunda ordem

function runge_kutta_2(f, u0, tspan, h, A, b, c)
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
        # Estágios de Runge-Kutta
        k1 = h * f(t, u)
        k2 = h * f(t + b[2]*h, u + A[2,1]*k1)

        # Atualização do valor de y
        u += c[1]*k1 + c[2]*k2
        push!(u_vect, u)
        # Atualização do tempo
        t += h
    end

    return u, u_vect
end

end