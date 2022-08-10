% Partially replicates Graphs (a) and (b) of Figure 2 of Crouzet and Tourre
% (2021), page 52. Programmed using Ben Moll's HACT toolikit as a guide.
% written by Rayhan Momin
%--------------------------------------------------------------------------

clear all; clc; close all;

%% PARAMETERS AND INITIAL OBJECTS:
tic

% Parameters
r = 0.05; % risk-free rate
kappa = r; % coupon rate of unsecured debt
m = 0.10; % debt amortization rate of unsecured debt
delta = 0.10; % depreciation rate
Theta = 0.35; % corporate income tax rate
sigma = 0.31; % volatility of productivity shock
gamma = 7.16; % curvature of capital adjustment cost: 7.16; with psi > 0
alpha_k = 0.33; % 1-deadweight losses
alpha_b = 0.15; % 1-debt haircut
a_avg = 0.24; % productivity parameter

% Endogeous state grid (debt to capital)
I = 250;
xmin = 1e-12;
xmax = 5; % CT'21 debt/ebitda = 2.13. APK = 0.24. debt/K=2.13*.24=.53.
x = linspace(xmin, xmax, I)'; % column vector
x_shift = x*(alpha_b/alpha_k);
x_mat = repmat(x,[1 length(x_shift)]);
[minValue,closestIndex] = min(abs(x_mat-x_shift'));

% Exogenous state grid (productivity)
J = 1; % number of grid points
amin = .24; % 5%tile CT'21 =.22, se = 0.01
amax = .24; % 95%tile CT'21=.25, se = 0.01
a = linspace(amin,amax,J); % row vector
%da = a(2)-a(1);
dx = x(2)-x(1);

% Optimal no-debt investment
g_star_vec = r -((2/gamma)*(delta + r + 0.5*gamma*r^2 - (1-Theta )*a)).^(0.5);
g_quo = ((2/gamma)*(delta + r + 0.5*gamma*r^2 - (1-Theta)*a));
if min(g_quo) < 0
    disp('Non-real solution to optimal investment rate.')
end

% Characteristic equation
xi_a = sigma^2/2;
xi_b = -(g_star_vec + m + sigma^2/2);
xi_c = -(r-g_star_vec);
xi = (-xi_b + sqrt(xi_b.^2-4*xi_a*xi_c))./(2*xi_a);

% Default boundary
x_ubar_star = (xi/(xi-1))*((r+m)/(kappa*(1-Theta )+m))*(((1-Theta )*a-(delta+g_star_vec+0.5*gamma*g_star_vec.^2))./(r-g_star_vec));

% Equity value function
e_coef = ((((1-Theta )*kappa+m)/(r+m))*x_ubar_star-(((1-Theta )*a-(delta+g_star_vec+0.5*gamma*g_star_vec.^2))./(r-g_star_vec)))/(x_ubar_star.^xi);
e_guess = ((1-Theta )*a-(delta+g_star_vec+0.5*gamma*g_star_vec.^2))./(r-g_star_vec)-(((1-Theta )*kappa+m)/(r+m))*x+e_coef*x.^xi;
e_guess(x>=x_ubar_star)=0;

%% Setup FDM
% Construct C matrix for exogenous state transitions
chi = (sigma^2*x.^2)/(2*dx^2);
varthe = - (sigma^2*x.^2)/(dx^2);
varrho = (sigma^2*x.^2)/(2*dx^2);

% Use if process is reflected (double-check)
centdiag_c = varthe;
centdiag_c(1) = varthe(1) + chi(1);
centdiag_c(I) = varthe(I) + varrho(I);

C =spdiags(centdiag_c,0,I,I)+spdiags(chi(2:I),-1,I,I)+spdiags([0;varrho(1:I-1)],1,I,I);

% Iteration set-up
Delta = 1000; % step size (for time) in the implicit method
maxit = 100;
tol = 1e-6;
mattol = 1e-6;
lcptol = 1e-3;
zeroapprox = 0;
iter = 0;
dist = zeros(maxit,1); % hold errors during iteration
ee = zeros(I,J);
gg = zeros(I,J);
xx = x*ones(1,J);
aa = ones(I,1)*a;

%% Solve with default option

for j=1:J
    e = e_guess(1:I,j);
    g_star = g_star_vec(1,j);
for n=1:maxit
    E = e;
    % Forward difference for the derivative of e wrt x;
    Exf(1:I-1) = (E(2:I)-E(1:I-1))/dx; 
    Exf(I) = (E(I)-gamma*(-m)-1)./xmax;
    Exf = reshape(Exf,I,1);
    
    % Backward difference for the derivative of e wrt x:
    Exb(2:I) = (E(2:I)-E(1:I-1))/dx; 
    Exb(1,:) = (E(1,:)-gamma*(g_star)-1)./xmin;
    Exb = reshape(Exb,I,1);
    
    I_convex = Exb <= Exf; %indicator whether value function is convex  
    
    % Investment rate and drift with forward difference
    gf = (1/gamma)*(E-x.*Exf-1);
    tempf = -(gf+m).*x;
    iotaf = tempf;
    iotaf(abs(iotaf)<1e-6) = 0;
    theta_f = a(j) - Theta*(a(j)-kappa*x)-(delta+gf+0.5*gamma*gf.^2)-(kappa + m)*x;
    Hf = theta_f + Exf.*iotaf;
    % Investment rate and drift with backward difference
    gb = (1/gamma)*(E-x.*Exb-1);
    tempb = -(gb+m).*x;
    iotab = tempb;
    iotab(abs(iotab)<1e-6) = 0;
    theta_b = a(j) - Theta*(a(j)-kappa*x)-(delta+gb+0.5*gamma*gb.^2)-(kappa + m)*x;
    Hb = theta_b + Exb.*iotab;
    % Investment rate at zero drift
    g0 = repelem(-m,I)';    
    g0(1) = g_star; % optimal investment for firm that takes on no debt
    Ex0 = (E - gamma*g0 - 1) ./ x;
    theta_0 = a(j) - Theta*(a(j)-kappa*x)-(delta+g0+0.5*gamma*g0.^2)-(kappa + m)*x;
    H0 = theta_0;

    % Make choice of forward or backward differences based on sign of drift.
    If = iotaf > zeroapprox; % positive drift
    Ib = iotab < zeroapprox; % negative drift
    I0 = (1-If-Ib); % zero drift
   
    % Policy update    
    g = gf.*If + gb.*Ib + g0.*I0;
    theta = a(j) - Theta*(a(j)-kappa*x)-(delta+g+0.5*gamma*g.^2)-(kappa + m)*x;

    % Construct discrete approximation of infinitesmal generator
    xi = -(min(iotab,0))/dx;
    beta =  -(max(iotaf,0))/dx + (min(iotab,0))/dx;
    zeta = (max(iotaf,0))/dx;
    
    centdiag = spdiags(beta,0,I,I);
    lowdiag = spdiags(xi(2:I),-1,I,I);
    updiag = spdiags([0;zeta(1:I-1)],1,I,I);
    
    % Check monotonicity
    if max(min(updiag >= 0))== 0
        disp('Updiag elements not >= 0')
    end
    if max(min(centdiag <= 0))== 0
        disp('Centdiag elements not <= 0')
    end
    if max(min(lowdiag >= 0))== 0
        disp('Lowdiag elements not <= 0')
    end

    A_tilde = spdiags(beta,0,I,I)+spdiags(xi(2:I),-1,I,I)+spdiags([0;zeta(1:I-1)],1,I,I);
    
    A = A_tilde + C;
    
    % Check transition matrix conditions
    if max(abs(sum(A,2))) > mattol
        disp('Improper Transition Matrix')
        disp(n)
        check_mat = full(A_tilde);
        check_sum = abs(sum(A_tilde,2));        
        break
    end
    
    % Solve HJBVI problem as LCP
    g_sparse = spdiags(reshape(g,I,1),0,I,I);
    B = (1/Delta + r)*speye(I) - g_sparse - A;
    theta_stacked = reshape(theta, I,1);
    e_stacked = reshape(e, I,1);
    b = theta_stacked + e_stacked/Delta;
    S = alpha_k*e_stacked(closestIndex);
    q = -b + B*S;
    z0 = e_stacked-S; lbnd = zeros(I,1); ubnd = Inf*ones(I,1); % set initial value
    z = LCP(B,q,lbnd,ubnd,z0,0);
    LCP_error = max(abs(z.*(B*z+q)));
    if LCP_error > lcptol
        disp('LCP not solved')
        disp(n)
    end
    E_stacked = z + S;
    E = reshape(E_stacked,I,1);
    Echange = E - e;
    e = E;

    dist(n) = max(max(abs(Echange)));
    if dist(n)<tol
        disp('Value Function Converged, Iteration = ')
        disp(n)
        break
    end
end
ee(1:I,j)=e;
gg(1:I,j)=g;
end

x_res = x(z<1e-6);
x_ubar = x_res(1);
dte = x/a_avg;

toc


%% Plot equity value function
plot(dte(x<=x_ubar),e(x<=x_ubar))
%surf(xx,aa,ee)
title('Equity Value per Unit of Capital across State Parameters')
xlabel('Debt to EBITDA (x/a)')
%ylabel('Productivity (A)')
%zlabel('Equity Value per Unit of Capital (e)')
ylabel('Equity Value per Unit of Capital (e)')

%% Plot investment policy function
plot(dte(x<=x_ubar),g(x<=x_ubar))
%surf(xx,aa,gg) 
title('Net Investment Rate across State Parameters')
xlabel('Debt to EBITDA (x/a)')
%ylabel('Productivity (A)')
%zlabel('Net Investment Rate')
ylabel('Net Investment Rate')

%% Plot gross investment policy function
g_gross = (delta+g+.5*gamma*g.^2);
%gg_gross = (delta+gg+.5*gamma*gg.^2);
plot(dte(x<=x_ubar),g_gross(x<=x_ubar))
%surf(xx,aa,gg_gross) 
title('Gross Investment Rate across State Parameters')
xlabel('Debt to EBITDA (x/a)')
ylabel('Gross Investment Rate')

