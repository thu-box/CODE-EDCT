function DCT_TOP_static(nelx, nely, DCTX, DCTY, volfrac, penal)
% DCT-based Topology Optimization for static compliance minimization.
% Inputs:
%   nelx   - Number of elements in x-direction
%   nely   - Number of elements in y-direction
%   DCTX   - Number of retained DCT modes in x-direction
%   DCTY   - Number of retained DCT modes in y-direction
%   volfrac- Volume fraction constraint
%   penal  - Penalization power (SIMP)

% ------------------------------
% 1. Material and element properties
% ------------------------------
E = 1;      % Young's modulus of solid material
E0 = 1;     % Reference Young's modulus
Emin = 1e-8; % Minimum Young's modulus (to avoid singularity)
nu = 0.3;   % Poisson's ratio

% Element stiffness matrix (plane stress quadrilateral element)
A11 = [12  3 -6 -3;  3 12  3  0; -6  3 12 -3; -3  0 -3 12];
A12 = [-6 -3  0  3; -3 -6 -3 -6;  0 -3 -6  3;  3 -6  3 -6];
B11 = [-4  3 -2  9;  3 -4 -9  4; -2 -9 -4 -3;  9  4 -3 -4];
B12 = [ 2 -3  4 -9; -3  2  9 -2;  4  9  2  3; -9 -2  3  2];
KE = E/(1-nu^2)/24*([A11 A12; A12' A11] + nu*[B11 B12; B12' B11]);

% ------------------------------
% 2. Finite element assembly
% ------------------------------
% Node numbering and element connectivity
eleN1 = repmat((1:nely)', 1, nelx) + kron(0:nelx-1, (nely+1)*ones(nely,1));
eleNode = repmat(eleN1(:), 1, 4) + repmat([0, nely+[1,2], 1], nelx*nely, 1);
edofMat = kron(eleNode, [2,2]) + repmat([-1,0], nelx*nely, 4);
nele = nely * nelx;

% Global stiffness matrix indexing
iK = reshape(kron(edofMat, ones(8,1))', 64*nelx*nely, 1);
jK = reshape(kron(edofMat, ones(1,8))', 64*nelx*nely, 1);

% Boundary conditions: left edge fixed, right edge free
fixeddofs = [1:2:2*(nely+1), 2*(nely+1)*(nelx+1)];
freedofs = setdiff(1:2*(nely+1)*(nelx+1), fixeddofs);

% External force: unit vertical load at top-right corner
F = zeros(2*(nely+1)*(nelx+1), 1);
F(2) = -1;
U = zeros(2*(nely+1)*(nelx+1), 1);

% ------------------------------
% 3. DCT basis initialization
% ------------------------------
Dx = dctmtx(nelx);  % Full DCT matrix in x-direction
Dy = dctmtx(nely);  % Full DCT matrix in y-direction
Dy_com = Dy(1:DCTY, :);  % Truncated basis in y-direction
Dx_com = Dx(1:DCTX, :);  % Truncated basis in x-direction

% Initialize physical density field
g = repmat(volfrac, [nely, nelx]);  % Initial density field
g_acti = (g >= 0) & (g <= 1);    % Active elements indicator

% DCT coefficient matrix (design variables)
x_full = Dy*g*Dx';
x = x_full(1:DCTY, 1:DCTX);

% Bounds for DCT coefficients
xmax_mat = zeros(DCTY, DCTX);
xmin_mat = zeros(DCTY, DCTX);
for i = 1:DCTY
    for j = 1:DCTX
        xmax_mat(i,j) = sum(sum(max(0, Dy(i,:)' * Dx(j,:))));
        xmin_mat(i,j) = sum(sum(min(0, Dy(i,:)' * Dx(j,:))));
    end
end


% Variables for MMA optimizer
m = 1;  % Number of constraints (volume constraint)
n = DCTY * DCTX;  % Number of DCT design variables
xold1 = x(:);
xold2 = x(:);
xmin = xmin_mat(:);
xmax = xmax_mat(:);
low = xmin;
upp = xmax;
c = 1e3;
d = zeros(m,1);
a0 = 1;
a = zeros(m,1);

% ------------------------------
% 4. Optimization loop
% ------------------------------
nloop = 200;    % Maximum iterations
tolx = 1e-3;    % Convergence tolerance
loop = 0;
change = 1;     % Design change
obj = zeros(nloop,1);  % Objective history

while change > tolx && loop < nloop
    loop = loop + 1;
    
    % a. FE analysis
    sK = reshape(KE(:) * (Emin + g(:)'.^penal * (E0 - Emin)), 64*nelx*nely, 1);
    K = sparse(iK, jK, sK); 
    K = (K + K')/2;  % Ensure symmetry
    U(freedofs) = K(freedofs, freedofs) \ F(freedofs);
    
    % b. Compliance and sensitivity analysis
    ce = reshape(sum((U(edofMat) * KE) .* U(edofMat), 2), nely, nelx);
    obj(loop) = sum(sum((Emin + g.^penal * (E0 - Emin)) .* ce));
    
    % Sensitivity in physical space
    dc_phy = -(E0 - Emin) * penal * g.^(penal-1) .* ce;
    
    % c. Project sensitivities to DCT space
    dc = Dy_com * (dc_phy.* g_acti) * Dx_com';
    dv = Dy_com * g_acti * Dx_com';
    
    % d. MMA update
    xval = x(:);
    f0val = obj(loop);
    df0dx = dc(:);
    fval = sum(g(:)) - volfrac * nele;
    dfdx = dv(:)';
    
    [xmma, ~, ~, ~, ~, ~, ~, ~, ~, low, upp] = ...
        mmasub(m, n, loop, xval, xmin, xmax, xold1, xold2, ...
               f0val, df0dx, fval, dfdx, low, upp, a0, a, c, d);
    
    % e. Update variables
    xnew = reshape(xmma, DCTY, DCTX);
    xold2 = xold1(:);
    xold1 = x(:);
    
    % Transform back to physical space
    gnew = Dy_com' * xnew * Dx_com;
    g_acti = (gnew >= 0) & (gnew <= 1);
    gnew = max(0, min(1, gnew));  % Project to [0,1]
    
    % f. Convergence check
    change = max(abs(gnew(:) - g(:)));
    g = gnew;
    x = xnew;
    
    % Display iteration info
    disp([' It.: ' sprintf('%4i',loop) ...
          ' Obj.: ' sprintf('%10.4f',obj(loop)) ...
          ' Vol.: ' sprintf('%6.3f',sum(sum(g))/(nelx*nely)) ...
          ' ch.: ' sprintf('%6.3f',change)])
    
    % Visualization
    figure(1);
    colormap(gray); 
    imagesc(-g); 
    axis equal; axis tight; axis off; 
    pause(1e-6);
end
end