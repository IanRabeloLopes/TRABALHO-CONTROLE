function dxdt = vehicle_nonlinear(x, t, v, delta)
    a = 1.5;  % m (distância do CM ao eixo traseiro)
    b = 3.0;  % m (distância entre eixos)
    
    % Ângulo de deriva (alpha)
    alpha = atan(a * tan(delta) / b);
    
    % Equações de estado
    dxdt = zeros(3, 1);
    dxdt(1) = v * cos(x(3) + alpha);  % dx/dt
    dxdt(2) = v * sin(x(3) + alpha);  % dy/dt
    dxdt(3) = (v * cos(alpha) * tan(delta)) / b;  % dθ/dt
endfunction
